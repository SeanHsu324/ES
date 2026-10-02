import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn.functional as F
from big_two_core import BigTwoGameNP, DECK_SIZE, get_hand_indices
from action_space import ACTION_SPACE_SIZE, INDEX_TO_ACTION_VECTOR, get_action_mask_np

class BigTwoEnv(gym.Env):
    """
    大老二強化學習環境 - 對抗性微調版本
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, opponent_model=None, device='cpu'):
        super().__init__()
        self.game = BigTwoGameNP(num_players=4)
        self.opponent_model = opponent_model
        self.device = device
        
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(113,), dtype=np.float32
        )

    def _get_obs(self, game_state):
        current_player = max(0, min(3, game_state['current_player']))
        hand_vector = game_state['hands'][current_player]
        last_played_hand = game_state['last_played_hand']
        remaining_cards = np.array([np.sum(h) for h in game_state['hands']], dtype=np.float32)
        other_remaining_cards = np.delete(remaining_cards, current_player) / 13.0
        
        last_player_one_hot = np.zeros(4, dtype=np.float32)
        last_played_player = game_state['last_played_player']
        if last_played_player != -1:
            last_player_one_hot[last_played_player] = 1.0
            
        players_passed = np.array([game_state['players_passed'] / 3.0], dtype=np.float32)
        is_new_round = np.array([1.0 if np.sum(last_played_hand) == 0 else 0.0], dtype=np.float32)
        
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
        current_player = max(0, min(3, game_state['current_player']))
        hand_vector = game_state['hands'][current_player]
        last_played_hand = game_state['last_played_hand']
        is_start_of_game = self.game.is_start_of_game()
        mask, _ = get_action_mask_np(hand_vector, last_played_hand, is_start_of_game)
        
        if np.sum(last_played_hand) == 0:
            mask[0] = 0
        return {"action_mask": mask}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        game_state = self.game.reset()
        
        while game_state['current_player'] != 0 and not game_state['is_game_over']:
            info_others = self._get_info(game_state)
            mask = info_others["action_mask"]
            
            if self.opponent_model is not None:
                obs = self._get_obs(game_state)
                with torch.no_grad():
                    action_index, _, _ = self.opponent_model.get_action(obs, mask, device=self.device)
            else:
                legal_actions = np.where(mask == 1)[0]
                action_index = np.random.choice(legal_actions) if len(legal_actions) > 0 else 0
            
            action_vector = INDEX_TO_ACTION_VECTOR[action_index]
            game_state, _, terminated, truncated, _ = self.game.step(action_vector)
            if terminated or truncated:
                game_state = self.game.reset()
                continue
        return self._get_obs(game_state), self._get_info(game_state)

    def step(self, action):
        info = self._get_info(self.game._get_state())
        if info["action_mask"][action] == 0:
            return self._get_obs(self.game._get_state()), -500.0, True, False, {"is_legal": False}
            
        game_state, reward, terminated, truncated, _ = self.game.step(INDEX_TO_ACTION_VECTOR[action])
        
        while game_state['current_player'] != 0 and not terminated and not truncated:
            info_others = self._get_info(game_state)
            mask = info_others["action_mask"]
            
            if self.opponent_model is not None:
                obs = self._get_obs(game_state)
                with torch.no_grad():
                    action_index, _, _ = self.opponent_model.get_action(obs, mask, device=self.device)
            else:
                legal_actions = np.where(mask == 1)[0]
                action_index = np.random.choice(legal_actions) if len(legal_actions) > 0 else 0
                
            game_state, _, terminated, truncated, _ = self.game.step(INDEX_TO_ACTION_VECTOR[action_index])
            
        # --- 獎勵優化 (Reward Shaping) ---
        if terminated or truncated:
            final_hands = game_state['hands']
            ai_hand = final_hands[0]
            num_cards = np.sum(ai_hand)
            
            if num_cards == 0:
                opponents_cards = sum([np.sum(final_hands[i]) for i in range(1, 4)])
                # 極大化贏牌獎勵
                reward = 200.0 + opponents_cards * 10.0
            else:
                # 極大化輸牌懲罰
                reward = -num_cards * 10.0
                if np.any(ai_hand[48:52]):
                    reward -= 100.0
                if num_cards >= 10:
                    reward -= 200.0
        else:
            # 鼓勵出牌而不是 Pass
            reward = 0.1 if action != 0 else -0.1
            
        return self._get_obs(game_state), reward, terminated, truncated, self._get_info(game_state)

    def render(self):
        pass

    def close(self):
        pass
