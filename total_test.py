import cv2
import numpy as np
import torch
from ultralytics import YOLO
from big_two_model import BigTwoA2C
from agents import ModelAgent
from action_space import get_action_mask_np, INDEX_TO_ACTION_VECTOR


DETECTOR_PATH = r'C:\Users\User\Desktop\ES\train7\weights\best.pt'
AI_MODEL_PATH = r'C:\Users\User\Desktop\ES\Ai_mod\big_two_a2c_ep50000.pth'

detector = YOLO(DETECTOR_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

brain = BigTwoA2C()
brain.load_state_dict(torch.load(AI_MODEL_PATH, map_location=device))
agent = ModelAgent(brain, device, name="LiveAI")

# 0-12: 梅花(C), 13-25: 方塊(D), 26-38: 紅心(H), 39-51: 黑桃(S)
SUIT_ORDER = ['C', 'D', 'H', 'S'] 
RANK_ORDER = ['3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', '2']

def get_card_id(name):
    try:
        s, r = name[0], name[1]
        return SUIT_ORDER.index(s) * 13 + RANK_ORDER.index(r) + 1
    except: return None

def id_to_card_name(card_id):
    if not (1 <= card_id <= 52): return "??"
    return f"{SUIT_ORDER[(card_id-1)//13]}{RANK_ORDER[(card_id-1)%13]}"


def main():
    cap = cv2.VideoCapture(1)
    
    # 狀態變數
    hand_cards = set()
    last_played_cards = set()
    temp_buffer = set()
    others_count = [13, 13, 13] # P2, P3, P4 的剩餘牌數
    
    # 輪到誰出牌 (0: P1(自己), 1: P2, 2: P3, 3: P4)
    current_turn_player = 0 
    # 最後一個出牌的人是誰
    last_action_player = 0
    
    pass_count = 0            
    is_new_round = True # 初始為新回合
    
    ai_suggestion = "Ready"

    print("--- 進階控制系統 ---")
    print("[H]手牌 [P]暫存 [Enter]提交 [Space]對手Pass [C]清空 [R]重設 [Q]離開")

    while True:
        ret, frame = cap.read()
        if not ret: break

        results = detector.predict(source=frame, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        curr_ids = [get_card_id(results[0].names[int(box.cls[0])]) for box in results[0].boxes if get_card_id(results[0].names[int(box.cls[0])])]

        # --- 更新 UI 顯示 ---
        hand_str = " ".join([id_to_card_name(c) for c in sorted(list(hand_cards))])
        last_str = " ".join([id_to_card_name(c) for c in sorted(list(last_played_cards))])
        
        y = 30
        cv2.putText(annotated_frame, f"HAND: {hand_str}", (20, y), 1, 1.2, (0, 255, 0), 2); y+=35
        cv2.putText(annotated_frame, f"LAST PLAY (P{last_action_player+1}): {last_str}", (20, y), 1, 1.2, (255, 200, 0), 2); y+=35
        cv2.putText(annotated_frame, f"TURN: P{current_turn_player+1} | PASS: {pass_count}", (20, y), 1, 1.0, (255, 255, 255), 1); y+=40
        cv2.putText(annotated_frame, f"AI ADVICE: {ai_suggestion}", (20, y), 1, 1.5, (0, 0, 255), 2)

        cv2.imshow("BigTwo AI Live Control", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): 
            break
            
        elif key == ord('h'): # 輸入手牌
            if curr_ids:
                for cid in curr_ids: hand_cards.add(cid)
                print(f"[系統] 已加入 {len(curr_ids)} 張牌到手牌")
            else:
                print("[警告] 畫面上沒偵測到牌，無法錄入手牌")

        elif key == ord('p'): # 暫存其他玩家出牌
            if curr_ids:
                before_count = len(temp_buffer)
                for cid in curr_ids: temp_buffer.add(cid)
                after_count = len(temp_buffer)
                print(f"[系統] 緩衝區新增了 {after_count - before_count} 張牌 (總計: {after_count})")
            else:
                print("[警告] 畫面上沒偵測到牌，請確保牌在框內且辨識標籤有出現")

        elif key == 32: # Space: 其他玩家 Pass
            if current_turn_player == 0:
                print("[警告] 現在是你的回合，你不能幫自己Pass。若要Pass請按Enter。")
                continue

            pass_count += 1
            print(f"[系統] 玩家 P{current_turn_player+1} Pass，累計 Pass: {pass_count}")
            current_turn_player = (current_turn_player + 1) % 4
            
            if temp_buffer:
                temp_buffer.clear()
                print("[系統] 緩衝區已清空。")
            
            if pass_count >= 3:
                is_new_round = True
                last_played_cards.clear()
                pass_count = 0
                current_turn_player = last_action_player
                print(f"[系統] 三人 Pass，開始新回合。輪到 P{current_turn_player+1}")
            
            if current_turn_player == 0:
                ai_suggestion = "Your Turn! Press Enter to get advice."
                print("[系統] 輪到你了，請按 Enter 獲取AI建議。")


        elif key == ord('c'): # 清空緩衝
            temp_buffer.clear()
            print("[系統] 緩衝區已清空")

        elif key == ord('r'): # 全重設
            hand_cards.clear(); last_played_cards.clear(); temp_buffer.clear()
            others_count = [13, 13, 13]
            pass_count = 0; is_new_round = True; current_turn_player = 0; last_action_player = 0
            ai_suggestion = "Ready"
            print("[系統] 遊戲狀態已完全重置")

        elif key == 13: # Enter
            if current_turn_player == 0:
               
                if is_new_round and last_action_player == 0:
                    last_played_cards.clear()
                    print("[系統] 你贏得了出牌權，開始新回合。")

                #113 維
                obs = np.zeros(113, dtype=np.float32)
                for c in hand_cards: obs[c-1] = 1.0
                for c in last_played_cards: obs[52+c-1] = 1.0
                obs[104:107] = np.array(others_count) / 13.0
                obs[107 + last_action_player] = 1.0
                obs[111] = float(pass_count)
                obs[112] = float(is_new_round)

            
                hand_vec = np.zeros(52, dtype=np.int8)
                for c in hand_cards: hand_vec[c-1] = 1
                last_vec = np.zeros(52, dtype=np.int8)
                for c in last_played_cards: last_vec[c-1] = 1
                
                mask, _ = get_action_mask_np(hand_vec, last_vec, is_start_of_game=is_new_round)
                action_idx = agent.choose_action(obs, mask)
                
                sugg_vec = INDEX_TO_ACTION_VECTOR[action_idx]
                res_ids = {i+1 for i, v in enumerate(sugg_vec) if v == 1}
                
                if not res_ids: 
                    pass_count += 1
                    ai_suggestion = "Pass"
                    print(f"> AI 建議: Pass。累計 Pass: {pass_count}")
                    if pass_count >= 3:
                        is_new_round = True; last_played_cards.clear(); pass_count = 0
                        current_turn_player = last_action_player
                        print(f"[系統] 三人 Pass，開始新回合。輪到 P{current_turn_player+1}")
                    else:
                        current_turn_player = (current_turn_player + 1) % 4
                else: 
                    ai_suggestion = " ".join([id_to_card_name(c) for c in sorted(list(res_ids))])
                    print(f"> AI 建議: {ai_suggestion}")
                    
                    hand_cards -= res_ids
                    last_played_cards = res_ids
                    is_new_round = False
                    pass_count = 0
                    last_action_player = 0
                    current_turn_player = (current_turn_player + 1) % 4
                    print(f"[系統] 你已出牌，剩餘 {len(hand_cards)} 張。輪到 P{current_turn_player+1}")

            elif temp_buffer:
                played_count = len(temp_buffer)
                player_idx = current_turn_player
                
                if 1 <= player_idx <= 3:
                    others_count[player_idx - 1] -= played_count
                
                last_played_cards = temp_buffer.copy()
                temp_buffer.clear()
                is_new_round = False
                pass_count = 0
                last_action_player = player_idx
                
                print(f"[系統] 玩家 P{player_idx+1} 打出了 {played_count} 張牌。")
                
                current_turn_player = (current_turn_player + 1) % 4
                print(f"[系統] 輪到 P{current_turn_player+1}")

                if current_turn_player == 0:
                    ai_suggestion = "Your Turn! Press Enter to get advice."
                    print("[系統] 輪到你了，請按 Enter 獲取AI建議。")
                else:
                    ai_suggestion = f"Waiting for P{current_turn_player+1}"
            else:
                print(f"[警告] 現在是 P{current_turn_player+1} 的回合，請先用 'P' 鍵掃描他們出的牌，再按 Enter。")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()