import gymnasium as gym
from gymnasium import spaces
import numpy as np
from big_two_core import BigTwoGameNP, DECK_SIZE, get_hand_indices
from action_space import ACTION_SPACE_SIZE, INDEX_TO_ACTION_VECTOR, get_action_mask_np

# 狀態空間維度:
# 1. 當前玩家手牌 (52)
# 2. 最後出的牌 (52)
# 3. 其他玩家剩餘牌數 (3)
# 4. 最後出牌的玩家 (4) - One-hot encoding
# 5. 連續 Pass 次數 (1)
# 6. 是否為新一輪 (1)
# 總計: 52 + 52 + 3 + 4 + 1 + 1 = 113

class BigTwoEnv(gym.Env):
    """
    大老二強化學習環境 (NumPy 向量化版本)
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.game = BigTwoGameNP(num_players=4)
        
        # 動作空間: 固定的 4551 個動作 + Pass (0)
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        
        # 狀態空間: 113 維向量
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(113,), dtype=np.float32
        )
        
        self.render_mode = render_mode

    def _get_obs(self, game_state):
        """
        將遊戲狀態轉換為 113 維的觀察向量。
        """
        current_player = game_state['current_player']
        
        # 1. 當前玩家手牌 (52)
        hand_vector = game_state['hands'][current_player]
        
        # 2. 最後出的牌 (52)
        last_played_hand = game_state['last_played_hand']
        
        # 3. 其他玩家剩餘牌數 (3)
        remaining_cards = np.array([np.sum(h) for h in game_state['hands']], dtype=np.float32)
        # 移除當前玩家的牌數
        other_remaining_cards = np.delete(remaining_cards, current_player)
        
        # 4. 最後出牌的玩家 (4) - One-hot encoding
        last_player_one_hot = np.zeros(4, dtype=np.float32)
        last_played_player = game_state['last_played_player']
        if last_played_player != -1:
            last_player_one_hot[last_played_player] = 1.0
            
        # 5. 連續 Pass 次數 (1)
        players_passed = np.array([game_state['players_passed']], dtype=np.float32)
        
        # 6. 是否為新一輪 (1)
        is_new_round = np.array([1.0 if np.sum(last_played_hand) == 0 else 0.0], dtype=np.float32)
        
        # 組合觀察向量
        obs = np.concatenate([
            hand_vector,
            last_played_hand,
            other_remaining_cards,
            last_player_one_hot,
            players_passed,
            is_new_round
        ]).astype(np.float32)
        
        return obs

    def _get_info(self, game_state):
        """
        獲取額外資訊，包括合法動作遮罩。
        """
        current_player = game_state['current_player']
        hand_vector = game_state['hands'][current_player]
        last_played_hand = game_state['last_played_hand']
        
        # 檢查是否為遊戲開始 (必須出梅花 3)
        is_start_of_game = self.game.is_start_of_game()
        
        # 獲取合法動作遮罩
        mask, _ = get_action_mask_np(hand_vector, last_played_hand, is_start_of_game)
        
        return {"action_mask": mask}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 重置遊戲
        game_state = self.game.reset()
        
        # 模擬其他玩家的動作直到輪到 AI 玩家 (玩家 0)
        while game_state['current_player'] != 0 and not game_state['is_game_over']:
            # 隨機選擇一個合法動作 (Pass 或出牌)
            mask, _ = get_action_mask_np(
                game_state['hands'][game_state['current_player']],
                game_state['last_played_hand'],
                self.game.is_start_of_game()
            )
            
            # 找出所有合法的動作索引
            legal_actions = np.where(mask == 1)[0]
            
            if len(legal_actions) == 0:
                # 理論上不應發生，但如果發生，強制 Pass
                action_index = 0
            else:
                # 隨機選擇一個合法動作
                action_index = np.random.choice(legal_actions)
                
            # 獲取動作向量
            action_vector = INDEX_TO_ACTION_VECTOR[action_index]
            
            # 執行動作
            game_state, _, terminated, truncated, _ = self.game.step(action_vector)
            
            if terminated or truncated:
                # 遊戲結束，重新開始
                return self.reset(seed=seed, options=options)
                
        observation = self._get_obs(game_state)
        info = self._get_info(game_state)
        
        return observation, info

    def step(self, action):
        # 1. 檢查動作是否合法 (在訓練循環中，這應該由模型處理)
        info = self._get_info(self.game._get_state())
        if info["action_mask"][action] == 0:
            # 非法動作，給予極大懲罰並結束遊戲
            reward = -100.0
            terminated = True
            truncated = False
            info["is_legal"] = False
            return self._get_obs(self.game._get_state()), reward, terminated, truncated, info
            
        # 2. 獲取動作向量
        action_vector = INDEX_TO_ACTION_VECTOR[action]
        
        # 3. 執行動作
        game_state, reward, terminated, truncated, step_info = self.game.step(action_vector)
        
        # 4. 模擬其他玩家的動作直到輪到 AI 玩家 (玩家 0)
        while game_state['current_player'] != 0 and not terminated and not truncated:
            # 隨機選擇一個合法動作 (Pass 或出牌)
            mask, _ = get_action_mask_np(
                game_state['hands'][game_state['current_player']],
                game_state['last_played_hand'],
                self.game.is_start_of_game()
            )
            
            legal_actions = np.where(mask == 1)[0]
            
            if len(legal_actions) == 0:
                # 理論上不應發生，但如果發生，強制 Pass
                action_index = 0
            else:
                action_index = np.random.choice(legal_actions)
                
            action_vector = INDEX_TO_ACTION_VECTOR[action_index]
            
            # 執行動作
            game_state, step_reward, terminated, truncated, _ = self.game.step(action_vector)
            reward += step_reward # 累積獎勵
            
        observation = self._get_obs(game_state)
        info = self._get_info(game_state)
        
        return observation, reward, terminated, truncated, info

    def render(self):
        # 渲染邏輯 (可選)
        pass

    def close(self):
        # 清理資源 (可選)
        pass

# --- 測試 ---
if __name__ == '__main__':
    env = BigTwoEnv()
    obs, info = env.reset()
    
    print(f"初始觀察向量維度: {obs.shape}")
    print(f"初始合法動作數量: {np.sum(info['action_mask'])}")
    
    # 選擇一個合法的動作
    legal_actions = np.where(info['action_mask'] == 1)[0]
    action = np.random.choice(legal_actions)
    
    # 執行動作
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"執行動作 {action} 後的獎勵: {reward}")
    print(f"遊戲是否結束: {terminated}")
    print(f"下一輪合法動作數量: {np.sum(info['action_mask'])}")
