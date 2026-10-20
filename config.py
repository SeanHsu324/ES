"""
共用設定 (main.py / record.py 都讀這裡)
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AI_MODEL_PATH = os.path.join(BASE_DIR, 'Ai_mod', 'big_two_best.pth')
SNAPSHOT_DIR = os.path.join(BASE_DIR, 'webcam_snapshots')   # 每次拍手牌都存一張，方便除錯

# --- 機器手臂 ---
COM_PORT = 'COM5'          # 設成 None 則為模擬模式 (不接手臂，只印出動作)

# --- 鏡頭 (用 record.py 的 cams 指令查有哪些編號) ---
HAND_CAMERA_INDEX = 1      # 看手牌架的鏡頭
PLAY_CAMERA_INDEX = 2      # 看出牌區的鏡頭
CAMERA_WIDTH = 1280        # 解析度越高，Gemini 越容易看清牌角
CAMERA_HEIGHT = 720

# --- 手牌架分區 ---
# 手臂先移到 pose (拍照點) 再用手牌鏡頭拍照，該排由左到右依序是接下來 count 個格子
# 例：第一排 7 張 = 格子 1~7，第二排 6 張 = 格子 8~13
HAND_ZONES = [
    {'label': 'Row 1', 'pose': 'photo_1', 'count': 7},
    {'label': 'Row 2', 'pose': 'photo_2', 'count': 6},
]
NUM_SLOTS = sum(z['count'] for z in HAND_ZONES)
assert NUM_SLOTS == 13, "HAND_ZONES 的 count 加總必須是 13"
PHOTO_SETTLE_SEC = 0.8     # 手臂到拍照點後等晃動停止再拍
HAND_REVERSED = False      # 若鏡頭畫面左右與格子順序相反，改成 True (每一排都會反轉)

# 辨識區域 (x, y, w, h，皆為畫面比例 0~1)。None = 整張畫面
HAND_ROI = None            # 手牌鏡頭畫面中，牌架那一排的位置
PLAY_ROI = None            # 出牌鏡頭畫面中，出牌區的位置

# --- 錄製動作名稱 ---
PICK_NAME = 'pick_{slot:02d}'   # 夾第 N 格
DROP_NAME = 'drop'              # 往前丟牌

# --- Gemini ---
# API Key 請設在環境變數 GEMINI_API_KEY (PowerShell: $env:GEMINI_API_KEY="你的key")
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')
