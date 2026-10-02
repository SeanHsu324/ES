'''
主訓練腳本，用於執行 A2C 演算法來訓練大老二 AI。
'''
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os # 導入 os 模組
from collections import deque
from gymnasium.vector import AsyncVectorEnv

from big_two_env import BigTwoEnv
from big_two_model import BigTwoA2C, STATE_DIM, ACTION_SPACE_SIZE
from action_space import INDEX_TO_ACTION_VECTOR

# --- 超參數設定 ---
# 檢查 GPU 是否可用，並使用 RTX 4060 Ti 8GB
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 5e-5
NUM_ENVS = 8 # 並行環境數量 (建議與 CPU 核心數一致，以充分利用 CPU 進行環境模擬)
GAMMA = 0.99  # 折扣因子
BETA = 0.01  # 熵係數 (鼓勵探索)

# 訓練設定
TOTAL_EPISODES = 500000  # 總訓練局數
ROLLOUT_STEPS = 256  # 每個 Rollout Batch 的步數
EVAL_INTERVAL = 1000  # 每隔多少局評估一次模型
EVAL_EPISODES = 500  # 評估時進行的局數

def make_env():
    """環境工廠函式，用於 AsyncVectorEnv"""
    return BigTwoEnv()

class A2CTrainer:
    def __init__(self):
        # 創建並行環境
        self.env = AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])
        print(f"環境初始化完成，並行環境數: {NUM_ENVS}")
        print(f"訓練設備: {DEVICE}")
        self.model = BigTwoA2C(action_space_size=ACTION_SPACE_SIZE).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.action_space_size = ACTION_SPACE_SIZE
        
        # 嘗試載入最新的模型權重 (實現斷點續訓)
        self.latest_episode = self._load_latest_model()
        
        # 訓練數據記錄
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.win_rates = deque(maxlen=100)
        
        # 確保狀態和動作空間維度正確
        assert self.env.single_observation_space.shape[0] == STATE_DIM
        assert self.env.single_action_space.n == ACTION_SPACE_SIZE

    def train(self):
        print(f"開始訓練 A2C 模型 (並行環境數: {NUM_ENVS})...")
        
        # 重置所有環境
        states, infos = self.env.reset()
        
        # 獲取初始合法動作遮罩
        legal_action_masks = infos["action_mask"]
        
        # 記錄每個環境的 episode 資訊
        episode_rewards_acc = np.zeros(NUM_ENVS)
        episode_lengths_acc = np.zeros(NUM_ENVS)
        
        total_episodes_completed = self.latest_episode # 從載入的最新 episode 開始
        total_steps = 0
        start_time = time.time()
        
        while total_episodes_completed < TOTAL_EPISODES:
            
            # --- Rollout 數據收集 ---
            log_probs = []
            values = []
            rewards = []
            masks = []
            
            for _ in range(ROLLOUT_STEPS):
                
                # --- 性能監控 ---
                if total_steps % 1000 == 0 and total_steps > 0:
                    elapsed_time = time.time() - start_time
                    steps_per_sec = total_steps / elapsed_time
                    print(f"Total Steps: {total_steps} | Episodes: {total_episodes_completed} | FPS: {steps_per_sec:.2f} steps/sec")
                total_steps += NUM_ENVS
                
                # 選擇動作 (批量處理)
                states_tensor = torch.from_numpy(states).float().to(DEVICE)
                
                # 模型前向傳播 (保留梯度)
                action_logits, values_tensor = self.model(states_tensor)
                
                # 應用合法動作遮罩 (批量處理)
                masks_tensor = torch.from_numpy(legal_action_masks).float().to(DEVICE)
                # 將非法動作的 Logits 設為極小值
                masked_logits = action_logits + (masks_tensor - 1) * 1e9
                
                # 動作機率分佈
                action_probs = F.softmax(masked_logits, dim=-1)
                
                # 採樣動作
                dist = torch.distributions.Categorical(action_probs)
                actions_index = dist.sample()
                
                # 計算 log_prob
                log_prob = dist.log_prob(actions_index)
                    
                # 將模型選擇的索引轉換回遊戲動作 (批量處理)
                actions_index_np = actions_index.cpu().numpy()
                
                # 由於我們在 BigTwoEnv.step 中直接使用 action_index，
                # 這裡我們只需要將 actions_index_np 傳遞給 env.step
                # AsyncVectorEnv 會自動將 action_index 傳遞給 BigTwoEnv.step
                
                # 執行動作 (批量處理)
                new_states, step_rewards, terminateds, truncateds, infos = self.env.step(actions_index_np)
                dones = terminateds | truncateds
                
                # 儲存數據
                # 確保所有張量的形狀都是 (NUM_ENVS, 1)
                log_probs.append(log_prob.view(NUM_ENVS, 1))
                values.append(values_tensor.view(NUM_ENVS, 1))
                rewards.append(torch.from_numpy(step_rewards).float().view(NUM_ENVS, 1).to(DEVICE))
                masks.append(torch.from_numpy(1.0 - dones).float().view(NUM_ENVS, 1).to(DEVICE))
                
                # 處理 Episode 結束
                episode_rewards_acc += step_rewards
                episode_lengths_acc += 1
                
                # 處理 Episode 結束
                for i, done in enumerate(dones):
                    if done:
                        self.episode_rewards.append(episode_rewards_acc[i])
                        self.episode_lengths.append(episode_lengths_acc[i])
                        episode_rewards_acc[i] = 0
                        episode_lengths_acc[i] = 0
                        total_episodes_completed += 1
                
                states = new_states
                # 獲取下一狀態的合法動作遮罩
                legal_action_masks = infos["action_mask"]
                
                # 檢查是否達到總 Episode 數
                if total_episodes_completed >= TOTAL_EPISODES:
                    actual_rollout_steps = i + 1 # 記錄實際的 Rollout 步數
                    break
            
            # 計算 Rollout 結束時的最終價值 V(S_T))
            with torch.no_grad():
                next_states_tensor = torch.from_numpy(states).float().to(DEVICE)
                _, next_values_tensor = self.model(next_states_tensor)
                
            # 由於 Rollout 步數可能不足 ROLLOUT_STEPS，我們將實際步數傳遞給 update_model
            if 'actual_rollout_steps' not in locals():
                actual_rollout_steps = ROLLOUT_STEPS
                
            self.update_model(rewards, log_probs, values, masks, next_values_tensor, actual_rollout_steps)

            # --- 記錄與評估 ---
            if total_episodes_completed % 100 < NUM_ENVS and len(self.episode_rewards) > 0:
                print(f"Episode {total_episodes_completed}/{TOTAL_EPISODES} | " 
                      f"Avg Reward: {np.mean(self.episode_rewards):.2f} | " 
                      f"Avg Length: {np.mean(self.episode_lengths):.2f}")

            if total_episodes_completed % EVAL_INTERVAL < NUM_ENVS and total_episodes_completed > 0:
                self.evaluate()
                # 保存模型到 'models' 資料夾
                model_dir = "models"
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                model_path = os.path.join(model_dir, f"big_two_a2c_ep{total_episodes_completed}.pth")
                torch.save(self.model.state_dict(), model_path)
                print(f"模型已保存到: {model_path}")
                
    def update_model(self, rewards, log_probs, values, masks, next_values, actual_rollout_steps):
        
        # --- 關鍵修復：重新計算實際 Rollout 步數 ---
        # 由於多進程環境可能導致 actual_rollout_steps 計算錯誤，我們使用實際的元素數量來計算
        total_elements = torch.cat(log_probs).numel()
        # 確保總元素數量是 NUM_ENVS 的倍數
        if total_elements % NUM_ENVS != 0:
            raise RuntimeError(f"Rollout data size ({total_elements}) is not a multiple of NUM_ENVS ({NUM_ENVS}).")
            
        actual_rollout_steps = total_elements // NUM_ENVS
        # --- 關鍵修復結束 ---
        
        # 重新組織數據
        # 為了確保維度正確，我們強制將其扁平化後再重塑
        log_probs = torch.cat(log_probs).flatten().view(actual_rollout_steps, NUM_ENVS, 1)
        values = torch.cat(values).flatten().view(actual_rollout_steps, NUM_ENVS, 1)
        rewards = torch.cat(rewards).flatten().view(actual_rollout_steps, NUM_ENVS, 1)
        masks = torch.cat(masks).flatten().view(actual_rollout_steps, NUM_ENVS, 1)
        
        # 計算折扣回報 (Discounted Returns)
        returns = torch.zeros_like(rewards)
        R = next_values.view(1, NUM_ENVS, 1) # (1, NUM_ENVS, 1)
              # 循環上限應為實際的 Rollout 步數
        for t in reversed(range(actual_rollout_steps)):
            R = rewards[t] + GAMMA * R * masks[t]
            returns[t] = R
            
        # 扁平化數據
        log_probs = log_probs.view(-1, 1)
        values = values.view(-1, 1)
        returns = returns.view(-1, 1)
        
        # 計算優勢函數 (Advantage)
        advantage = returns - values
        
        # 計算損失
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        
        # 熵損失 (鼓勵探索)
        # 重新計算 action_probs 以獲得熵
        # 這裡需要重新執行 Rollout 步驟，或者將 action_probs 儲存下來
        # 為了簡化，我們使用一個近似的熵損失
        entropy_loss = -(log_probs.exp() * log_probs).mean()
        
        # 總損失
        loss = actor_loss + 0.5 * critic_loss + BETA * entropy_loss
        
        # 反向傳播與優化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪 (可選)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

    def evaluate(self):
        # 評估邏輯 (簡化版，只運行單個環境)
        eval_env = BigTwoEnv()
        win_count = 0
        total_opp_cards = 0
        
        for _ in range(EVAL_EPISODES):
            state, info = eval_env.reset()
            terminated = False
            truncated = False
            
            while not terminated and not truncated:
                # 獲取合法動作遮罩
                mask = info["action_mask"]
                
                # 選擇動作 (貪婪策略)
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
                action_logits, _ = self.model(state_tensor)
                
                # 應用遮罩
                mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).to(DEVICE)
                masked_logits = action_logits + (mask_tensor - 1) * 1e9
                
                # 選擇最大機率的動作
                action = torch.argmax(masked_logits, dim=-1).item()
                
                # 執行動作
                state, reward, terminated, truncated, info = eval_env.step(action)
                
            # 檢查是否勝利 (玩家 0)
            if reward > 0: # 勝利獎勵為 100.0
                win_count += 1
            
            # 計算對手剩餘牌數
            # 狀態向量的 52+52 到 52+52+3 是對手剩餘牌數 (歸一化)
            opp_cards_normalized = state[104:107]
            # 由於歸一化因子是 13.0，我們需要反歸一化
            total_opp_cards += np.sum(opp_cards_normalized * 13.0)
            
        win_rate = win_count / EVAL_EPISODES
        avg_opp_cards = total_opp_cards / EVAL_EPISODES
        
        print(f"\n--- 評估結果 ---")
        print(f"勝率: {win_rate:.2%}")
        print(f"平均對手剩餘牌數: {avg_opp_cards:.2f}")
        
        return win_rate, avg_opp_cards

    def _load_latest_model(self):
        """查找並載入最新的模型權重"""
        model_dir = "models"
        if not os.path.exists(model_dir):
            return 0
            
        model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
        if not model_files:
            return 0
            
        # 找出最新的模型文件 (基於 episode 數字)
        latest_file = max(model_files, key=lambda f: int(f.split('ep')[1].split('.')[0]))
        latest_path = os.path.join(model_dir, latest_file)
        
        try:
            self.model.load_state_dict(torch.load(latest_path, map_location=DEVICE))
            print(f"成功載入最新模型: {latest_path}")
            # 從文件名中提取 episode 數
            latest_episode = int(latest_file.split('ep')[1].split('.')[0])
            return latest_episode
        except Exception as e:
            print(f"載入模型失敗: {e}")
            return 0

if __name__ == '__main__':
    trainer = A2CTrainer()
    trainer.train()
