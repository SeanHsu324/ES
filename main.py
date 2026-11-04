import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # 避免繁中 Windows 主控台 (cp950) 印 emoji 時崩潰
import time
import threading
import queue

import cv2
import numpy as np
import torch

# 導入你提供的環境定義檔案
from big_two_model import BigTwoA2C
from agents import ModelAgent
from action_space import get_action_mask_np, INDEX_TO_ACTION_VECTOR
from big_two_core import evaluate_hand_np, compare_hands_np
from robot_arm import RobotArm, HOME
from config import (
    AI_MODEL_PATH, COM_PORT, HAND_CAMERA_INDEX, PLAY_CAMERA_INDEX,
    NUM_SLOTS, PLAY_ROI, PICK_NAME, DROP_NAME, save_env_values,
)
from vision import (
    Camera, GeminiCardReader, scan_hand, crop,
    get_card_id, id_to_card_name, names_str,
)
from table_display import TableDisplay
from discord_bridge import DiscordBridge, TeeStdout

# =====================================================
# 1. 配置 (共用設定在 config.py：COM 埠、鏡頭編號、手牌分排、辨識區域、Discord)
# =====================================================
# --- 全自動：出牌區變化偵測 (決定何時才做影像辨識) ---
MOTION_THRESH = 4.0        # 相鄰兩幀平均差異 > 此值 = 有東西在動 (手伸進來)
CHANGE_THRESH = 10.0       # 穩定後與上次基準差異 > 此值 = 桌面真的變了 -> 辨識
STABLE_SEC = 1.0           # 畫面需靜止多久才判定放好牌了
PASS_TIMEOUT_SEC = 15.0    # 對手回合出牌區這麼久沒變化 -> 視為 Pass；設 None = 只用 PASS 按鈕 / /pass
PASS_BUTTON_DEBOUNCE_SEC = 0.5   # PASS 按鈕連按的最短間隔 (避免一次按壓算兩次 Pass)
ARM_SETTLE_SEC = 1.0       # 手臂動作結束後等一下再重設基準畫面
DISPLAY_FPS = 15 if os.name == "nt" else 8   # 觀眾畫面更新頻率 (樹莓派畫一張約 90ms，所以降到 8)

# 每打出一張牌要依序播放的動作，每張牌丟完都回初始點
# 例: 第 5 格 -> pick_05 -> drop -> home
CARD_ACTION_SEQUENCE = [PICK_NAME, DROP_NAME, HOME]

# 訓練環境 (big_two_env.py) 中「對手剩餘牌數」是原始張數，沒有除以 13
# 若你的 big_two_best.pth 是用除以 13 的版本訓練，改成 True
NORMALIZE_OPP_COUNTS = False

PLAYER_NAMES = {0: 'AI', 1: 'P2', 2: 'P3', 3: 'P4'}
SUIT_SYMBOL = {'C': '♣', 'D': '♦', 'H': '♥', 'S': '♠'}


def ids_to_vec(ids):
    v = np.zeros(52, dtype=np.int8)
    for c in ids: v[c-1] = 1
    return v


def pretty_cards(ids):
    """牌局頻道用：♠A ♥10 ♣3 (由小到大)"""
    out = []
    for c in sorted(ids, key=lambda c: ((c - 1) % 13, (c - 1) // 13)):
        n = id_to_card_name(c)
        out.append(SUIT_SYMBOL[n[0]] + ('10' if n[1] == 'T' else n[1]))
    return " ".join(out)


# =====================================================
# 2. 出牌區變化偵測 (決定什麼時候才做影像辨識)
# =====================================================
class PlayZoneWatcher:
    def __init__(self):
        self.baseline = None
        self.prev = None
        self.stable_since = None
        self.last_motion_time = time.time()

    @staticmethod
    def _small(frame):
        g = cv2.cvtColor(crop(frame, PLAY_ROI), cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (160, 90))
        return cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)

    def rebaseline(self, frame):
        self.baseline = self._small(frame)
        self.prev = self.baseline
        self.stable_since = None
        self.last_motion_time = time.time()

    def update(self, frame):
        """回傳 True 代表：出牌區有變化且已靜止，值得做一次辨識"""
        small = self._small(frame)
        if self.baseline is None:
            self.rebaseline(frame)
            return False
        motion = float(np.mean(np.abs(small - self.prev)))
        self.prev = small
        now = time.time()
        if motion > MOTION_THRESH:
            self.stable_since = None
            self.last_motion_time = now
            return False
        if self.stable_since is None:
            self.stable_since = now
        if now - self.stable_since < STABLE_SEC:
            return False
        if float(np.mean(np.abs(small - self.baseline))) > CHANGE_THRESH:
            self.baseline = small   # 不論辨識結果如何，同一個畫面只辨識一次
            return True
        return False

    def idle_seconds(self):
        return time.time() - self.last_motion_time


# =====================================================
# 3. 遊戲狀態
# =====================================================
class BigTwoGame:
    def __init__(self, agent):
        self.agent = agent
        self.on_event = None          # 牌局事件 (出牌、Pass、新回合、獲勝) 的通知函式
        self.reset()

    def reset(self):
        # 夾取位置對應: slot(1~13) -> card_id；牌被夾走後該格設為 None
        self.slots = {i: None for i in range(1, NUM_SLOTS + 1)}
        self.last_played_cards = set()
        self.played_history = set()   # 整局已出現在桌上的牌 (用來從出牌區找出「新」的牌)
        self.others_count = [13, 13, 13]
        self.current_turn_player = 0  # 0: AI, 1: P2, 2: P3, 3: P4
        self.last_action_player = 0
        self.pass_count = 0
        self.passed = set()           # 這一回合已經 Pass 的玩家 (觀眾畫面用)
        self.is_new_round = True
        self.game_started = False
        self.game_over = False
        self.winner = None
        self.turn_started_at = time.time()

    def _emit(self, *event):
        if self.on_event:
            self.on_event(*event)

    # --- 查詢 ---
    @property
    def hand_locked(self):
        return any(c is not None for c in self.slots.values())

    def hand_cards(self):
        return {c for c in self.slots.values() if c is not None}

    def slot_of(self, card_id):
        for s, c in self.slots.items():
            if c == card_id:
                return s
        return None

    def counts(self):
        """4 家剩餘張數 [AI, P2, P3, P4]"""
        mine = len(self.hand_cards()) if self.hand_locked or self.game_started else 13
        return [mine] + list(self.others_count)

    # --- 狀態轉移 ---
    def _next_turn(self, player):
        self.current_turn_player = player
        self.turn_started_at = time.time()

    def lock_hand(self, ids, first_player=None):
        self.slots = {i + 1: cid for i, cid in enumerate(ids)}
        if 1 in ids:
            self._next_turn(0)
        elif first_player is not None:
            self._next_turn(first_player)
        self._emit('start', self.current_turn_player)

    def _register_pass(self, who):
        self.pass_count += 1
        self.passed.add(who)
        print(f"[系統] {PLAYER_NAMES[who]} Pass，累計 Pass: {self.pass_count}")
        self._emit('pass', who)
        if self.pass_count >= 3:
            self.is_new_round = True
            self.last_played_cards.clear()
            self.pass_count = 0
            self.passed.clear()
            self._next_turn(self.last_action_player)
            print(f"[系統] 三人 Pass，新回合。輪到 {PLAYER_NAMES[self.current_turn_player]}")
            self._emit('new_round', self.current_turn_player)
        else:
            self._next_turn((who + 1) % 4)

    def opponent_pass(self):
        self._register_pass(self.current_turn_player)

    def is_legal_play(self, ids):
        vec = ids_to_vec(ids)
        hand_type, _ = evaluate_hand_np(vec)
        if hand_type == 0:
            return False
        if not self.last_played_cards:
            return True
        return compare_hands_np(vec, ids_to_vec(self.last_played_cards)) == 1

    def _after_play(self, p, ids, remaining):
        self.last_played_cards = set(ids)
        self.played_history |= set(ids)
        self.is_new_round = False
        self.game_started = True
        self.pass_count = 0
        self.passed.clear()
        self.last_action_player = p
        self._emit('play', p, set(ids), remaining)
        if remaining <= 0:
            self.game_over = True
            self.winner = p
            print(f"[系統] {PLAYER_NAMES[p]} 出完牌，遊戲結束")
            self._emit('win', p)
        else:
            self._next_turn((p + 1) % 4)

    def opponent_play(self, ids):
        p = self.current_turn_player
        self.others_count[p - 1] -= len(ids)
        print(f"[系統] {PLAYER_NAMES[p]} 打出 {names_str(ids)}，剩 {self.others_count[p-1]} 張")
        self._after_play(p, ids, self.others_count[p - 1])

    def decide_my_move(self):
        """AI 決策，回傳要出的 card_id 集合 (空集合 = Pass)"""
        hand = self.hand_cards()
        if self.is_new_round and self.last_action_player == 0:
            self.last_played_cards.clear()

        obs = np.zeros(113, dtype=np.float32)
        for c in hand: obs[c-1] = 1.0
        for c in self.last_played_cards: obs[52+c-1] = 1.0
        counts = np.array(self.others_count, dtype=np.float32)
        obs[104:107] = counts / 13.0 if NORMALIZE_OPP_COUNTS else counts
        obs[107 + self.last_action_player] = 1.0
        obs[111] = float(self.pass_count)
        obs[112] = float(self.is_new_round)

        hand_vec = ids_to_vec(hand)
        # 只有整局第一手 (桌面空且手上有梅花3) 才強制出梅花3
        is_start = (not self.game_started) and not self.last_played_cards and hand_vec[0] == 1
        mask, _ = get_action_mask_np(hand_vec, ids_to_vec(self.last_played_cards), is_start_of_game=is_start)
        action_idx = self.agent.choose_action(obs, mask)
        vec = INDEX_TO_ACTION_VECTOR[action_idx]
        return {i + 1 for i, v in enumerate(vec) if v == 1}

    def apply_my_move(self, ids):
        """更新狀態並回傳要夾取的格子 (由左到右)"""
        if not ids:
            self._register_pass(0)
            return []
        card_slots = sorted(self.slot_of(c) for c in ids)
        for s in card_slots:
            self.slots[s] = None
        self._after_play(0, ids, len(self.hand_cards()))
        return card_slots


# =====================================================
# 4. 主程序
# =====================================================
def main():
    bridge = DiscordBridge()
    if bridge.enabled:
        sys.stdout = TeeStdout(sys.stdout, bridge)   # 終端機的輸出同時送到 Discord 後臺
    bridge.start()

    device = torch.device("cpu")
    brain = BigTwoA2C()
    brain.load_state_dict(torch.load(AI_MODEL_PATH, map_location=device))
    agent = ModelAgent(brain, device, name="LiveAI")

    reader = GeminiCardReader()
    robot = RobotArm(COM_PORT)
    # 兩顆鏡頭放在同一個字典裡，/swapcam 互換時只要對調 (不用重開鏡頭)
    cams = {'hand': Camera(HAND_CAMERA_INDEX, "HAND CAM"), 'play': Camera(PLAY_CAMERA_INDEX, "PLAY CAM")}
    watcher = PlayZoneWatcher()
    game = BigTwoGame(agent)
    display = TableDisplay()

    st = {
        'auto': False,
        'running': True,
        'pending_hand': None,     # 辨識出、尚未確認的手牌
        'pending_first': None,    # 等 /first 指定誰有梅花3：要執行的函式
        'buffer': set(),          # 手動模式: /read 辨識到的對手出牌
        'need_rebaseline': False, # 手臂動作完後要重設出牌區基準畫面
        'scan_running': False,
        'scan_result': None,
        'scan_then_auto': False,
        'scan_first': None,
        'busy': None,             # 背景工作名稱 (辨識中)，None = 沒有
    }
    ui_tasks = queue.Queue()      # 背景完成後要回到主迴圈執行的函式
    frame_box = {'frame': None}

    # ---------- 牌局事件 -> 牌局頻道 (公開資訊，不含 AI 手牌) ----------
    def on_game_event(kind, *args):
        if kind == 'start':
            bridge.post_game(f"🃏 **牌局開始！** 由 **{PLAYER_NAMES[args[0]]}** 先出")
        elif kind == 'play':
            p, ids, remaining = args
            bridge.post_game(f"**{PLAYER_NAMES[p]}** 出牌　{pretty_cards(ids)}　（剩 {remaining} 張）")
        elif kind == 'pass':
            bridge.post_game(f"**{PLAYER_NAMES[args[0]]}** Pass")
        elif kind == 'new_round':
            bridge.post_game(f"🔄 新回合，由 **{PLAYER_NAMES[args[0]]}** 出牌")
        elif kind == 'win':
            bridge.post_game(f"🏆 **{PLAYER_NAMES[args[0]]} 獲勝！**")

    game.on_event = on_game_event

    # ---------- 背景工作 ----------
    def run_bg(label, work, done):
        """work() 在背景執行，完成後 done(結果) 回到主迴圈執行 (例外時結果為 None)"""
        def worker():
            try:
                result = work()
            except Exception as e:
                print(f"[{label}] 失敗: {e}")
                result = None
            ui_tasks.put(lambda: done(result))
        st['busy'] = label
        threading.Thread(target=worker, daemon=True).start()

    def start_scan(then_auto, first=None):
        """背景執行：手臂依序到各排拍照點 -> 手牌鏡頭拍照 -> 影像辨識 -> 回初始點"""
        def worker():
            try:
                result = scan_hand(robot, cams['hand'], reader)
            except Exception as e:
                print(f"[影像辨識] 失敗: {e}")
                result = (None, False)
                if robot.has_action(HOME) and not robot.abort.is_set():
                    robot.run_sequence([HOME], return_home=False)
                    robot.wait()
            st['scan_result'] = result
            st['scan_running'] = False

        st.update(scan_running=True, scan_result=None, scan_then_auto=then_auto, scan_first=first)
        print("[系統] 📷 開始拍照辨識手牌...")
        threading.Thread(target=worker, daemon=True).start()

    def lock_hand(names, first=None, then=None):
        ids = [get_card_id(n) for n in names]

        def finish(first_player):
            game.lock_hand(ids, None if 1 in ids else first_player - 1)
            st['pending_first'] = None
            print("[系統] 手牌已鎖定:")
            for s, c in game.slots.items():
                print(f"   第 {s:2d} 格 -> {id_to_card_name(c)}")
            print(f"[系統] 由 {PLAYER_NAMES[game.current_turn_player]} 先出")
            if then:
                then()

        if 1 in ids or first:
            finish(first or 1)
        else:
            st['pending_first'] = finish
            print("[系統] ❓ AI 沒有梅花3，請用 /first 指定是誰有梅花3")

    def enable_auto():
        missing = robot.missing_actions(
            [a.format(slot=s) for s in range(1, NUM_SLOTS + 1) for a in CARD_ACTION_SEQUENCE])
        if missing:
            print(f"[警告] 缺少動作檔 {missing}，請先用 record.py 錄製")
            return False
        if not robot.clamps_closed and not robot.close_clamps():
            print("[警告] 排夾無法關閉，全自動未啟動")
            return False
        # 先回初始點，到位後才開始偵測出牌區
        robot.run_sequence([HOME], return_home=False)
        st['need_rebaseline'] = True
        st['auto'] = True
        print("[系統] ✅ 全自動模式啟動")
        return True

    def play_my_turn():
        """AI 決策 + 叫手臂出牌"""
        ids = game.decide_my_move()
        if not ids:
            print("> AI: Pass")
            game.apply_my_move(ids)
            return

        slots_to_pick = game.apply_my_move(ids)
        print(f"> AI 出牌: {names_str(ids)}  (格子: {slots_to_pick})")
        seq = [a.format(slot=s) for s in slots_to_pick for a in CARD_ACTION_SEQUENCE]

        def on_done(ok):
            st['need_rebaseline'] = True
            if ok:
                print("[手臂] ✅ 出牌動作完成")
            else:
                st['auto'] = False
                print("[手臂] ⚠️ 動作中斷，自動模式已關閉，請確認牌桌")

        if not robot.run_sequence(seq, on_done=on_done):
            st['auto'] = False
            print("[系統] ❌ 手臂無法執行，自動模式已關閉，請手動出牌")

    def check_play_zone(frame):
        """出牌區有變化時做影像辨識 (背景)，找出對手新打出的牌"""
        turn = game.current_turn_player

        def done(names):
            if names is None:
                return
            if game.current_turn_player != turn or not st['auto']:
                print("[影像辨識] 等結果的期間狀態變了 (Pass / 關閉全自動)，忽略這次結果")
                return
            seen = {get_card_id(n) for n in names} - {None}
            new_ids = seen - game.played_history - game.hand_cards()
            print(f"[影像辨識] 出牌區: {names_str(seen)} -> 新的牌: {names_str(new_ids) or '(無)'}")
            if not new_ids:
                return
            if not (1 <= len(new_ids) <= 5) or not game.is_legal_play(new_ids):
                print(f"[警告] {names_str(new_ids)} 不是合法出牌，可能辨識錯誤；"
                      f"請整理牌後再放一次，或用 /read + /submit 手動輸入")
                return
            game.opponent_play(new_ids)

        run_bg('READING PLAY AREA', lambda: reader.read_played(frame), done)

    def opponent_pass_now(source):
        if not game.hand_locked:
            return f"[{source}] 還沒開局，忽略"
        if game.current_turn_player == 0:
            return f"[{source}] 現在是 AI 的回合，忽略"
        who = PLAYER_NAMES[game.current_turn_player]
        game.opponent_pass()
        st['buffer'] = set()
        if st['auto'] and frame_box['frame'] is not None:
            watcher.rebaseline(frame_box['frame'])
        return f"✅ {who} Pass"

    def busy_reason():
        if robot.busy:
            return "手臂動作中"
        if st['scan_running']:
            return "正在拍照辨識手牌"
        if st['busy']:
            return "影像辨識中"
        return None

    # ================= / 指令 =================
    def cmd_newgame():
        if robot.busy or st['scan_running']:
            return "⚠️ 手臂動作中，請稍後再開新局"
        game.reset()
        st.update(auto=False, pending_hand=None, pending_first=None, buffer=set())
        robot.open_clamps()
        print("[系統] 🆕 新局：狀態已重設、排夾已打開")
        bridge.post_game("🃏 **新牌局準備中…**")
        return "✅ 新局已就緒：排夾已打開。插好 13 張牌後用 /start"

    def cmd_start(first=None):
        b = busy_reason()
        if b:
            return f"⚠️ {b}，請稍後"
        if st['auto']:
            return "已經在全自動模式"
        if game.hand_locked:
            return "✅ 全自動模式啟動" if enable_auto() else "❌ 無法啟動全自動，詳情看後臺訊息"
        start_scan(then_auto=True, first=first)
        return "📷 開始拍照辨識手牌，完成後會自動開始"

    def cmd_first(player):
        if not st['pending_first']:
            return "目前不需要指定"
        st['pending_first'](player)
        return f"✅ 由 P{player} 先出"

    def cmd_scan():
        b = busy_reason()
        if b:
            return f"⚠️ {b}，請稍後"
        start_scan(then_auto=False)
        return "📷 開始拍照辨識手牌，完成後用 /status 查看、/fix 修正、/confirm 確認"

    def cmd_fix(slot, card):
        if st['pending_hand'] is None:
            st['pending_hand'] = [id_to_card_name(c) if c else '??' for c in game.slots.values()]
        if not get_card_id(card):
            return f"❌ 看不懂牌名「{card}」，格式例如 SA、H10、C3"
        st['pending_hand'][slot - 1] = card.strip().upper().replace('10', 'T')
        return f"✅ 第 {slot} 格改成 {st['pending_hand'][slot - 1]}，全部確認後用 /confirm\n" + \
            " ".join(f"{i+1}:{n}" for i, n in enumerate(st['pending_hand']))

    def cmd_confirm():
        ph = st['pending_hand']
        if ph is None:
            return "❌ 沒有待確認的手牌，請先 /scan"
        ids = [get_card_id(n) for n in ph]
        if None in ids or len(set(ids)) != NUM_SLOTS:
            return f"❌ 手牌不完整或有重複，請用 /fix 修正：\n" + " ".join(f"{i+1}:{n}" for i, n in enumerate(ph))
        st['pending_hand'] = None
        lock_hand(ph)
        if not robot.clamps_closed:
            robot.close_clamps()
        return "✅ 手牌已鎖定" + ("，請用 /first 指定誰有梅花3" if st['pending_first'] else "，用 /start 開始全自動")

    def cmd_auto(on):
        if not on:
            st['auto'] = False
            print("[系統] 全自動模式關閉")
            return "✅ 全自動模式關閉"
        return cmd_start()

    def cmd_pass():
        if robot.busy or st['scan_running']:
            return "⚠️ 手臂動作中，忽略"
        return opponent_pass_now("Discord")

    def cmd_read():
        b = busy_reason()
        if b:
            return f"⚠️ {b}，請稍後"
        frame = frame_box['frame']
        if frame is None:
            return "❌ 出牌區鏡頭沒有畫面"

        def got(names):
            if names is None:
                return
            st['buffer'] = {get_card_id(n) for n in names} - {None} - game.played_history
            print(f"[影像辨識] 對手出牌: {names_str(st['buffer']) or '(無)'} (用 /submit 提交)")

        run_bg('READING PLAY AREA', lambda f=frame: reader.read_played(f), got)
        return "🔍 辨識中，結果會顯示在後臺，之後用 /submit 提交"

    def cmd_submit():
        if game.current_turn_player == 0:
            return "❌ 現在是 AI 的回合，請用 /aiplay"
        if not st['buffer']:
            return "❌ 沒有辨識到的牌，請先 /read"
        ids, st['buffer'] = st['buffer'], set()
        game.opponent_play(ids)
        return f"✅ {names_str(ids)}"

    def cmd_aiplay():
        b = busy_reason()
        if b:
            return f"⚠️ {b}，請稍後"
        if not game.hand_locked:
            return "❌ 尚未鎖定手牌"
        if game.current_turn_player != 0:
            return f"❌ 現在是 {PLAYER_NAMES[game.current_turn_player]} 的回合"
        play_my_turn()
        return "✅ AI 出牌中"

    def cmd_stop():
        robot.emergency_stop()
        st['auto'] = False
        return "⛔ 已緊急停止，用 /resume 解除"

    def cmd_resume():
        robot.hold()
        if robot.has_action(HOME):
            robot.run_sequence([HOME], return_home=False)
        print("[系統] 已解除緊急停止，回到初始點")
        return "✅ 已解除停止，回到初始點 (全自動需要用 /start 重新開啟)"

    def cmd_clamp(action):
        if action == 'open' and st['auto']:
            return "❌ 全自動中不能開排夾，請先 /auto on:False"
        ok = robot.open_clamps() if action == 'open' else robot.close_clamps()
        return ("✅ 排夾已" + ("打開" if action == 'open' else "關閉")) if ok else "❌ 排夾沒有回應"

    def cmd_status():
        lines = [
            f"燈號：{light()}　模式：{'全自動' if st['auto'] else '手動'}　{busy_reason() or ''}",
            f"手臂：{robot.status}　排夾：{'關' if robot.clamps_closed else '開'}"
            f"　鏡頭：手牌 {cams['hand'].index} {'✅' if cams['hand'].ok else '❌'}"
            f" / 出牌區 {cams['play'].index} {'✅' if cams['play'].ok else '❌'}",
            f"輪到：{PLAYER_NAMES[game.current_turn_player]}　剩餘張數：" +
            " ".join(f"{PLAYER_NAMES[i]}={n}" for i, n in enumerate(game.counts())),
            f"上一手 ({PLAYER_NAMES[game.last_action_player]})：{names_str(game.last_played_cards) or '-'}",
            "AI 手牌：" + (" ".join(f"{s}:{id_to_card_name(c) if c else '--'}" for s, c in game.slots.items())
                          if game.hand_locked else "尚未鎖定"),
        ]
        if st['pending_hand']:
            lines.append("待確認：" + " ".join(f"{i+1}:{n}" for i, n in enumerate(st['pending_hand'])))
        if st['pending_first']:
            lines.append("❓ 等待 /first 指定誰有梅花3")
        return "\n".join(lines)

    def cmd_swapcam():
        b = busy_reason()
        if b:
            return f"⚠️ {b}，請稍後再換 (拍照中換鏡頭會拍錯)"
        cams['hand'], cams['play'] = cams['play'], cams['hand']
        cams['hand'].name, cams['play'].name = "HAND CAM", "PLAY CAM"
        st['need_rebaseline'] = True          # 出牌區畫面換了，偵測基準要重抓
        try:
            save_env_values({'HAND_CAMERA': cams['hand'].index, 'PLAY_CAMERA': cams['play'].index})
            saved = "，已存進 .env (下次開機也會用這個設定)"
        except OSError as e:
            saved = f"，但寫入 .env 失敗：{e}"
        print(f"[系統] 🔄 鏡頭互換：手牌 = {cams['hand'].index}，出牌區 = {cams['play'].index}")
        return f"✅ 鏡頭已互換：手牌 = {cams['hand'].index}，出牌區 = {cams['play'].index}{saved}"

    def cmd_shutdown():
        st['running'] = False
        return "👋 主程式關閉中"

    handlers = {
        'newgame': cmd_newgame, 'start': cmd_start, 'first': cmd_first, 'scan': cmd_scan,
        'fix': cmd_fix, 'confirm': cmd_confirm, 'auto': cmd_auto, 'pass': cmd_pass,
        'read': cmd_read, 'submit': cmd_submit, 'aiplay': cmd_aiplay, 'stop': cmd_stop,
        'resume': cmd_resume, 'clamp': cmd_clamp, 'status': cmd_status, 'swapcam': cmd_swapcam,
        'shutdown': cmd_shutdown,
    }

    # ================= 狀態燈 =================
    def light():
        """紅 = 需要人處理 (緊急停止、手臂/鏡頭故障)；黃 = 處理中或還沒開始全自動；綠 = 全自動正常進行"""
        arm_missing = robot.ser is None and COM_PORT is not None
        if robot.mode == 'stopped' or robot.status == 'STOPPED' or arm_missing or not cams['play'].ok:
            return 'red'
        if not st['auto'] or busy_reason() or st['pending_first'] or st['pending_hand'] is not None:
            return 'yellow'
        return 'green'

    # 啟動就移到初始點並停住
    if robot.has_action(HOME):
        print("[手臂] 移到初始點...")
        robot.run_sequence([HOME], return_home=False)
    else:
        print("[警告] 還沒有初始點 (actions/home.json)，請先用 record.py 設定")

    print("--- 大老二 AI + 機器手臂 (用 Discord / 指令控制；觀眾畫面按 Esc 可緊急停止) ---")
    last_pass_button = 0.0
    last_draw = 0.0

    while st['running']:
        frame = cams['play'].read()
        frame_box['frame'] = frame

        # --- 背景工作完成後的處理 ---
        while not ui_tasks.empty():
            st['busy'] = None
            ui_tasks.get()()

        # --- Discord / 指令 ---
        bridge.poll(handlers)

        # --- PASS 按鈕 ---
        while not robot.button_events.empty():
            t_press = robot.button_events.get()
            if t_press - last_pass_button < PASS_BUTTON_DEBOUNCE_SEC:
                continue
            last_pass_button = t_press
            if robot.busy or st['scan_running']:
                print("[PASS 按鈕] 手臂動作 / 拍照中，忽略")
            else:
                print(opponent_pass_now("PASS 按鈕"))   # 正在等的辨識結果會因為換人而自動忽略

        # --- 手牌辨識結果 ---
        if st['scan_result'] is not None:
            names, ok = st['scan_result']
            st['scan_result'] = None
            if names is not None:
                if ok and st['scan_then_auto']:
                    st['pending_hand'] = None
                    lock_hand(names, first=st['scan_first'], then=enable_auto)
                else:
                    st['pending_hand'] = names
                    print("[系統] 手牌辨識結果：" + " ".join(f"{i+1}:{n}" for i, n in enumerate(names)))
                    print("[系統] 確認無誤請用 /confirm" if ok else "[系統] 請用 /fix 修正後 /confirm，或 /scan 重拍")

        if frame is not None:
            # --- 手臂動作完成後重設基準 (避免手臂入鏡被誤判) ---
            if st['need_rebaseline'] and not robot.busy and time.time() - robot.finished_at > ARM_SETTLE_SEC:
                watcher.rebaseline(frame)
                game.turn_started_at = time.time()
                st['need_rebaseline'] = False

            # --- 全自動流程 ---
            if st['auto'] and not robot.busy and not st['need_rebaseline'] and not st['busy'] and not game.game_over:
                if game.current_turn_player == 0:
                    play_my_turn()
                elif watcher.update(frame):
                    check_play_zone(frame)
                elif (PASS_TIMEOUT_SEC is not None
                      and time.time() - game.turn_started_at > PASS_TIMEOUT_SEC
                      and watcher.idle_seconds() > PASS_TIMEOUT_SEC):
                    print(f"[系統] {PLAYER_NAMES[game.current_turn_player]} {PASS_TIMEOUT_SEC:.0f} 秒沒出牌，視為 Pass")
                    game.opponent_pass()
                if game.game_over:
                    st['auto'] = False

        # --- 觀眾畫面 (只顯示公開資訊) ---
        now = time.time()
        if now - last_draw >= 1.0 / DISPLAY_FPS:
            last_draw = now
            display.show({
                'status': light(),
                'counts': game.counts(),
                'played_total': len(game.played_history),
                'turn': game.current_turn_player if game.hand_locked and not game.game_over else None,
                'passed': game.passed,
                'winner': game.winner,
            }, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # Esc：現場緊急停止 (Discord 有延遲，留一個實體按鍵比較安全)
            print(cmd_stop())

    print("[系統] 主程式結束")
    time.sleep(2.5)     # 讓最後的訊息送到 Discord
    cams['hand'].release()
    cams['play'].release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[系統] 已中斷 (Ctrl+C)")
        cv2.destroyAllWindows()
