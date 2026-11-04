"""
Discord 機器人：完全用 / 指令控制，並把訊息送到兩個頻道
  - 後臺頻道：所有系統訊息 (終端機的輸出)、指令回覆
  - 牌局頻道：公開的出牌紀錄；開新局時清空

機器人跑在背景執行緒 (自己的 asyncio 迴圈)，畫面迴圈不會被卡住。
/ 指令收到後丟進佇列，由 main.py 的主迴圈執行 (遊戲狀態只在主執行緒改)，結果再回覆給使用者。
"""
import sys
import queue
import asyncio
import threading
from typing import Optional

import discord
from discord import app_commands

from config import (
    DISCORD_TOKEN, DISCORD_BACKEND_CHANNEL_ID, DISCORD_GAME_CHANNEL_ID, DISCORD_OWNER_IDS, NUM_SLOTS,
    DISCORD_PASS_BUTTON_ANYONE,
)

LOG_FLUSH_SEC = 2.0
MAX_MSG = 1900


def _err(msg):
    # 機器人自己的錯誤寫到 stderr，不要再送回 Discord (避免無限循環)
    print(f"[Discord] {msg}", file=sys.__stderr__)


class TeeStdout:
    """終端機照常顯示，同時把輸出複製一份送到 Discord 後臺"""

    def __init__(self, original, bridge):
        self.original = original
        self.bridge = bridge

    def write(self, text):
        self.original.write(text)
        self.bridge.log(text)
        return len(text)

    def flush(self):
        self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


class DiscordBridge:
    def __init__(self):
        self.enabled = bool(DISCORD_TOKEN)
        self.commands = queue.Queue()     # (指令名稱, 參數, future)
        self.connected = False
        self._log_lock = threading.Lock()
        self._log_buf = []
        self.loop = None
        self.client = None
        self._game_q = None           # 牌局頻道的訊息依序送 (避免「輪到誰」跑到出牌訊息前面)
        self._turn_msg = None         # 目前帶著 Pass 按鈕的那則訊息

    # ================= 給主執行緒用 =================
    def start(self):
        if not self.enabled:
            print("[Discord] ⚠️ .env 沒有 DISCORD_TOKEN，Discord 控制停用")
            return
        threading.Thread(target=self._thread_main, daemon=True).start()

    def log(self, text):
        if self.enabled and text:
            with self._log_lock:
                self._log_buf.append(text)

    def post_game(self, text):
        """送一則訊息到牌局頻道 (公開)"""
        self._enqueue_game(('text', text))

    def post_turn(self, text, turn_id, with_button):
        """公告輪到誰；with_button = 附上 Pass 按鈕 (上一則的按鈕會自動拿掉)"""
        self._enqueue_game(('turn', text, turn_id, with_button))

    def clear_turn_button(self):
        self._enqueue_game(('clear',))

    def _enqueue_game(self, item):
        if self.connected and self.loop and self._game_q is not None:
            self.loop.call_soon_threadsafe(self._game_q.put_nowait, item)

    def post_backend(self, text):
        self._post(DISCORD_BACKEND_CHANNEL_ID, text)

    def poll(self, handlers):
        """主迴圈每一輪呼叫：執行收到的 / 指令"""
        while not self.commands.empty():
            name, kwargs, fut = self.commands.get()
            try:
                result = handlers[name](**kwargs)
            except Exception as e:
                result = f"❌ 執行 /{name} 時發生錯誤：{e}"
            if self.loop and not fut.done():
                self.loop.call_soon_threadsafe(lambda f=fut, r=result: f.done() or f.set_result(r))

    # ================= Discord 執行緒 =================
    def _post(self, channel_id, text):
        if not (self.connected and self.loop):
            return

        async def send():
            ch = self.client.get_channel(channel_id)
            if ch is not None:
                for chunk in _chunks(text):
                    await ch.send(chunk)

        asyncio.run_coroutine_threadsafe(send(), self.loop)

    def _thread_main(self):
        try:
            asyncio.run(self._run())
        except Exception as e:
            _err(f"機器人停止：{e}")
            self.connected = False

    async def _call(self, name, **kwargs):
        fut = asyncio.get_running_loop().create_future()
        self.commands.put((name, kwargs, fut))
        try:
            return await asyncio.wait_for(fut, timeout=15)
        except asyncio.TimeoutError:
            return "⏳ 主程式忙碌中，指令已排入，稍後會執行"

    def can_press_pass(self, inter):
        if DISCORD_PASS_BUTTON_ANYONE:
            return True
        ids = {inter.user.id} | {r.id for r in getattr(inter.user, 'roles', [])}
        return bool(ids & DISCORD_OWNER_IDS)

    async def _drop_turn_button(self):
        msg, self._turn_msg = self._turn_msg, None
        if msg is not None:
            try:
                await msg.edit(view=None)
            except discord.HTTPException:
                pass        # 訊息已被刪除 (例如 /newgame 清空頻道)

    async def _game_sender(self):
        while True:
            item = await self._game_q.get()
            ch = self.client.get_channel(DISCORD_GAME_CHANNEL_ID)
            if ch is None:
                continue
            try:
                if item[0] == 'text':
                    for chunk in _chunks(item[1]):
                        await ch.send(chunk)
                elif item[0] == 'turn':
                    _, text, turn_id, with_button = item
                    await self._drop_turn_button()
                    if with_button:
                        self._turn_msg = await ch.send(text, view=PassView(self, turn_id))
                    else:
                        await ch.send(text)
                elif item[0] == 'clear':
                    await self._drop_turn_button()
            except Exception as e:
                _err(f"送出牌局訊息失敗：{e}")

    async def _flush_logs(self):
        while True:
            await asyncio.sleep(LOG_FLUSH_SEC)
            with self._log_lock:
                text = "".join(self._log_buf)
                self._log_buf.clear()
            text = text.strip()
            if not text:
                continue
            ch = self.client.get_channel(DISCORD_BACKEND_CHANNEL_ID)
            if ch is None:
                continue
            try:
                for chunk in _chunks(text, MAX_MSG - 10):
                    await ch.send(f"```\n{chunk}\n```")
            except Exception as e:
                _err(f"送出後臺訊息失敗：{e}")

    async def _run(self):
        self.loop = asyncio.get_running_loop()
        client = discord.Client(intents=discord.Intents.default())
        tree = app_commands.CommandTree(client)
        self.client = client
        self._register(tree)

        @tree.error
        async def on_app_command_error(inter, error):
            if isinstance(error, app_commands.CheckFailure):
                return          # 不是主人：已經回覆過「只有主人可以使用」
            _err(f"/{inter.command.name if inter.command else '?'} 錯誤：{error}")
            try:
                if inter.response.is_done():
                    await inter.followup.send(f"❌ {error}", ephemeral=True)
                else:
                    await inter.response.send_message(f"❌ {error}", ephemeral=True)
            except Exception:
                pass

        @client.event
        async def on_ready():
            backend = client.get_channel(DISCORD_BACKEND_CHANNEL_ID)
            if backend is None:
                _err(f"找不到後臺頻道 {DISCORD_BACKEND_CHANNEL_ID}，請確認機器人已加入伺服器")
                return
            guild = backend.guild
            # 只同步到這個伺服器 (不動全域指令，不會影響同一個 bot 的其他指令)
            tree.copy_global_to(guild=guild)
            synced = await tree.sync(guild=guild)
            self.connected = True
            print(f"[Discord] {client.user} 已上線，同步了 {len(synced)} 個 / 指令到「{guild.name}」")
            if not getattr(self, '_flusher', None):
                self._flusher = asyncio.create_task(self._flush_logs())
                self._game_q = asyncio.Queue()
                self._sender = asyncio.create_task(self._game_sender())

        await client.start(DISCORD_TOKEN)

    def _register(self, tree):
        bridge = self

        def owner_only():
            async def predicate(inter: discord.Interaction):
                ids = {inter.user.id} | {r.id for r in getattr(inter.user, 'roles', [])}
                if ids & DISCORD_OWNER_IDS:
                    return True
                await inter.response.send_message("⛔ 只有主人可以使用這個指令", ephemeral=True)
                return False
            return app_commands.check(predicate)

        async def reply(inter, name, **kwargs):
            await inter.response.defer(ephemeral=True, thinking=True)
            result = await bridge._call(name, **kwargs)
            await inter.followup.send(result or "✅", ephemeral=True)

        @tree.command(name="newgame", description="開新局：清空牌局頻道、重設狀態、打開排夾")
        @owner_only()
        async def newgame(inter: discord.Interaction):
            await inter.response.defer(ephemeral=True, thinking=True)
            game_ch = bridge.client.get_channel(DISCORD_GAME_CHANNEL_ID)
            cleared = 0
            if game_ch is not None:
                try:
                    deleted = await game_ch.purge(limit=None)
                    cleared = len(deleted)
                    bridge._turn_msg = None
                except discord.Forbidden:
                    await inter.followup.send("⚠️ 機器人沒有「管理訊息」權限，無法清空牌局頻道", ephemeral=True)
            result = await bridge._call("newgame")
            await inter.followup.send(f"🧹 已清空牌局頻道 {cleared} 則訊息\n{result}", ephemeral=True)

        @tree.command(name="start", description="拍照辨識手牌，完整就自動開始全自動對戰")
        @app_commands.describe(first="你沒有梅花3時，誰有梅花3 (先出)")
        @app_commands.choices(first=[app_commands.Choice(name=f"P{i}", value=i) for i in (2, 3, 4)])
        @owner_only()
        async def start(inter: discord.Interaction, first: Optional[app_commands.Choice[int]] = None):
            await reply(inter, "start", first=first.value if first else None)

        @tree.command(name="first", description="指定誰有梅花3 (先出)")
        @app_commands.choices(player=[app_commands.Choice(name=f"P{i}", value=i) for i in (2, 3, 4)])
        @owner_only()
        async def first(inter: discord.Interaction, player: app_commands.Choice[int]):
            await reply(inter, "first", player=player.value)

        @tree.command(name="scan", description="只拍照辨識手牌 (不自動開始)")
        @owner_only()
        async def scan(inter: discord.Interaction):
            await reply(inter, "scan")

        @tree.command(name="fix", description="修正辨識錯的某一格手牌")
        @app_commands.describe(slot=f"格子 1~{NUM_SLOTS}", card="牌名，例如 SA、H10、C3")
        @owner_only()
        async def fix(inter: discord.Interaction, slot: app_commands.Range[int, 1, NUM_SLOTS], card: str):
            await reply(inter, "fix", slot=slot, card=card)

        @tree.command(name="confirm", description="確認辨識結果，鎖定手牌")
        @owner_only()
        async def confirm(inter: discord.Interaction):
            await reply(inter, "confirm")

        @tree.command(name="auto", description="全自動對戰 開 / 關")
        @owner_only()
        async def auto(inter: discord.Interaction, on: bool):
            await reply(inter, "auto", on=on)

        @tree.command(name="read", description="辨識出牌區 (手動模式)")
        @owner_only()
        async def read(inter: discord.Interaction):
            await reply(inter, "read")

        @tree.command(name="submit", description="把 /read 辨識到的牌，當成現在對手出的牌 (手動模式)")
        @owner_only()
        async def submit(inter: discord.Interaction):
            await reply(inter, "submit")

        @tree.command(name="aiplay", description="輪到 AI 時，讓 AI 出牌 (手動模式)")
        @owner_only()
        async def aiplay(inter: discord.Interaction):
            await reply(inter, "aiplay")

        @tree.command(name="stop", description="⛔ 緊急停止手臂")
        @owner_only()
        async def stop(inter: discord.Interaction):
            await reply(inter, "stop")

        @tree.command(name="resume", description="解除緊急停止，回到初始點")
        @owner_only()
        async def resume(inter: discord.Interaction):
            await reply(inter, "resume")

        @tree.command(name="clamp", description="排夾 開 / 關")
        @app_commands.choices(action=[app_commands.Choice(name="開", value="open"),
                                      app_commands.Choice(name="關", value="close")])
        @owner_only()
        async def clamp(inter: discord.Interaction, action: app_commands.Choice[str]):
            await reply(inter, "clamp", action=action.value)

        @tree.command(name="status", description="查看系統狀態與 AI 手牌 (只有你看得到)")
        @owner_only()
        async def status(inter: discord.Interaction):
            await reply(inter, "status")

        @tree.command(name="swapcam", description="手牌鏡頭和出牌區鏡頭互換 (兩顆 USB 鏡頭讀反時用)")
        @owner_only()
        async def swapcam(inter: discord.Interaction):
            await reply(inter, "swapcam")

        @tree.command(name="shutdown", description="關閉主程式")
        @owner_only()
        async def shutdown(inter: discord.Interaction):
            await reply(inter, "shutdown")


class PassView(discord.ui.View):
    """「輪到 Px」訊息下面的 Pass 按鈕"""

    def __init__(self, bridge, turn_id):
        super().__init__(timeout=None)
        self.bridge = bridge
        self.turn_id = turn_id
        button = discord.ui.Button(label="Pass", emoji="✋", style=discord.ButtonStyle.secondary,
                                   custom_id=f"bigtwo_pass:{turn_id}")
        button.callback = self.on_click
        self.add_item(button)

    async def on_click(self, inter: discord.Interaction):
        if not self.bridge.can_press_pass(inter):
            await inter.response.send_message("⛔ 只有主人可以按 Pass", ephemeral=True)
            return
        await inter.response.defer(ephemeral=True, thinking=True)
        result = await self.bridge._call("pass_button", turn_id=self.turn_id)
        await inter.followup.send(result, ephemeral=True)


def _chunks(text, size=MAX_MSG):
    """把長文字切成 Discord 一則訊息放得下的大小 (盡量在換行處切)"""
    out, cur = [], ""
    for line in text.splitlines(keepends=True):
        if len(line) > size:                      # 單行就超過，硬切
            if cur:
                out.append(cur)
                cur = ""
            out += [line[i:i + size] for i in range(0, len(line), size)]
            continue
        if len(cur) + len(line) > size:
            out.append(cur)
            cur = ""
        cur += line
    if cur:
        out.append(cur)
    return [c for c in out if c.strip()]
