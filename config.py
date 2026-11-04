"""
共用設定 (main.py / record.py 都讀這裡)
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path):
    """讀取 .env (KEY=VALUE，每行一筆)；系統環境變數已經有的不會被覆蓋"""
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8-sig') as f:   # utf-8-sig: 記事本存檔可能帶 BOM
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            if key.startswith('export '):
                key = key[len('export '):].strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv(os.path.join(BASE_DIR, '.env'))


def save_env_values(values):
    """把設定寫回 .env (已有的那一行就改掉，沒有就加在最後)，其他行原封不動"""
    path = os.path.join(BASE_DIR, '.env')
    lines = []
    if os.path.exists(path):
        with open(path, encoding='utf-8-sig') as f:
            lines = f.read().splitlines()
    todo = {k: str(v) for k, v in values.items()}
    for i, line in enumerate(lines):
        key = line.split('=', 1)[0].strip()
        if key in todo:
            lines[i] = f"{key}={todo.pop(key)}"
    lines += [f"{k}={v}" for k, v in todo.items()]
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write("\n".join(lines) + "\n")
    for k, v in values.items():
        os.environ[k] = str(v)

AI_MODEL_PATH = os.path.join(BASE_DIR, 'Ai_mod', 'big_two_best.pth')
SNAPSHOT_DIR = os.path.join(BASE_DIR, 'webcam_snapshots')   # 每次拍手牌都存一張，方便除錯

IS_WINDOWS = os.name == 'nt'


def _env_device(name, default):
    """每台電腦不同的裝置設定，可在該台的 .env 覆蓋 (數字 = 編號，其他 = 路徑，none = 不用)"""
    v = os.environ.get(name, '').strip()
    if not v:
        return default
    if v.lower() == 'none':
        return None
    return int(v) if v.isdigit() else v


# --- 機器手臂 ---
# 'auto' = 自動找 ESP32 (USB 轉序列晶片 CH340 / CP210x / ESP32 內建 USB)
# 也可指定，例如 Windows 'COM5'、樹莓派 '/dev/ttyUSB0'；None = 模擬模式 (不接手臂)
# 可在 .env 寫 COM_PORT=COM5 覆蓋
#   樹莓派預設 'i2c' = 不用 ESP32，樹莓派 I2C 直接驅動 PCA9685 (接線見 pi_servo.py)
COM_PORT = _env_device('COM_PORT', 'auto' if IS_WINDOWS else 'i2c')
I2C_BUS = 1                # 樹莓派 GPIO2/3 的 I2C
PCA9685_ADDR = 0x40
PASS_BUTTON_GPIO = 17      # 樹莓派版 PASS 按鈕 (GPIO17 = pin 11，另一腳接 GND)

# --- 排夾 (J7 夾第一排、J8 夾第二排，接 PCA9685 通道 6、7) ---
# 開機預設開啟；拍完兩排、手臂回初始點後關閉；按 R 重設新局時打開
CLAMP_OPEN = (0, 90)       # J7: 0 開 / J8: 90 開
CLAMP_CLOSED = (90, 0)     # J7: 90 關 / J8: 0 關
CLAMP_SETTLE_SEC = 0.5     # 排夾動作後等多久再繼續

# --- 鏡頭 (用 record.py 的 cams 指令查有哪些編號) ---
# Windows 是編號 (0 通常是筆電內建鏡頭)；樹莓派建議用 /dev/v4l/by-id/... 路徑，插拔換孔也不會對調
# 可在 .env 寫 HAND_CAMERA=... / PLAY_CAMERA=... 覆蓋
HAND_CAMERA_INDEX = _env_device('HAND_CAMERA', 1 if IS_WINDOWS else 0)   # 看手牌架的鏡頭
PLAY_CAMERA_INDEX = _env_device('PLAY_CAMERA', 2 if IS_WINDOWS else 2)   # 看出牌區的鏡頭
CAMERA_WIDTH = 1280        # 解析度越高，Gemini 越容易看清牌角
CAMERA_HEIGHT = 720

# --- 手牌架分區 ---
# 手臂先移到 pose (拍照點) 再用手牌鏡頭拍照，該排由左到右依序是接下來 count 個格子
# 第一排 8 張 = 格子 1~8，第二排 5 張 = 格子 9~13
HAND_ZONES = [
    {'label': 'Row 1', 'pose': 'photo_1', 'count': 8},
    {'label': 'Row 2', 'pose': 'photo_2', 'count': 5},
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
# API Key 放在 ES/.env：GEMINI_API_KEY=你的key (也可以用環境變數，環境變數優先)
# 依序嘗試：前一個忙線 (503/504) 或超時就自動換下一個，成功的會記住下次優先用
# 2026-10-27 實測 (認 5 張牌全對)：3.5-flash 5~14s、3.1-flash-lite 3~4s、3.5-flash-lite 4~15s；
# 3.8-flash / 3.7-flash 當時伺服器過載一直超時，所以沒放進來
# 也可以在 .env 寫 GEMINI_MODEL=模型名稱，會排在最前面
GEMINI_MODELS = ['gemini-3.5-flash', 'gemini-3.1-flash-lite', 'gemini-3.5-flash-lite']
if os.environ.get('GEMINI_MODEL'):
    GEMINI_MODELS.insert(0, os.environ['GEMINI_MODEL'].strip())
GEMINI_TIMEOUT_SEC = 25    # 單一模型等多久沒回應就換下一個

# --- Discord (完全用 / 指令控制) ---
# Bot Token 放在 .env：DISCORD_TOKEN=你的token (沒有的話程式照常跑，只是不能用 Discord 控制)
DISCORD_TOKEN = os.environ.get('DISCORD_TOKEN', '').strip()
DISCORD_BACKEND_CHANNEL_ID = 1567535190211829822   # 後臺：系統訊息、指令回覆
DISCORD_GAME_CHANNEL_ID = 1567535349456961647      # 牌局對話：公開的出牌紀錄 (開新局時會清空)
# 能用控制指令的人：擁有這個身分組 (或使用者 ID 剛好是這個) 的成員
DISCORD_OWNER_IDS = {1567531351337345044}

# --- 觀眾顯示 (外接螢幕：紅黃綠狀態燈 + 牌局畫面 + 出牌區鏡頭) ---
TABLE_FULLSCREEN = not IS_WINDOWS   # 樹莓派預設全螢幕；Windows 開一般視窗方便測試
TABLE_SCREEN_X = int(os.environ.get('TABLE_SCREEN_X', '0'))   # 有兩個螢幕時，外接螢幕的左上角 X 座標
TABLE_SCREEN_Y = int(os.environ.get('TABLE_SCREEN_Y', '0'))
# 畫面上 4 家的位置 (玩家編號 0=AI, 1=P2, 2=P3, 3=P4)：下、右、上、左
TABLE_SEATS = {0: 'bottom', 1: 'right', 2: 'top', 3: 'left'}
