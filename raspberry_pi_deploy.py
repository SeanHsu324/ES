import torch
import numpy as np
from big_two_model import BigTwoA2C, ACTION_SPACE_SIZE

# 部署配置
MODEL_PATH = "models/big_two_best.pth"
DEVICE = torch.device("cpu")

class BigTwoAI:
    def __init__(self, model_path=MODEL_PATH):
        self.model = BigTwoA2C(action_space_size=ACTION_SPACE_SIZE).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        print(f"AI 模型已加載: {model_path}")

    def get_action(self, state, action_mask):
        """
        給機器手臂調用的接口
        :param state: 當前遊戲狀態 (186 維向量)
        :param action_mask: 合法動作掩碼 (1692 維向量，1 為合法，0 為非法)
        :return: 動作索引
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        mask_tensor = torch.from_numpy(action_mask).float().to(DEVICE)
        
        with torch.no_grad():
            action_logits, _ = self.model(state_tensor)
            # 應用掩碼
            masked_logits = action_logits + (mask_tensor - 1) * 1e9
            action = torch.argmax(masked_logits, dim=-1).item()
            
        return action

# 範例用法
if __name__ == "__main__":
    ai = BigTwoAI()
    # 這裡假設 state 和 mask 是從機器手臂的視覺系統或遊戲引擎獲取的
    # dummy_state = np.zeros(186)
    # dummy_mask = np.ones(1692)
    # action = ai.get_action(dummy_state, dummy_mask)
    # print(f"AI 建議動作索引: {action}")
