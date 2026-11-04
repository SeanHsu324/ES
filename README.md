# 大老二 AI 機器手臂

用機器手臂實際上桌打大老二：Gemini 看牌、A2C 模型（`big_two_best.pth`）決定要出哪幾張、六軸手臂從手牌架夾牌並往前丟出。

```
手牌鏡頭 ──(開局拍一次)──► Gemini ──► 13 格手牌對照表 ─┐
                                                      ├─► A2C 決策 ──► 手臂 pick_XX → drop → home
出牌鏡頭 ──(畫面有變化才拍)─► Gemini ──► 對手出了什麼 ──┘
```

---

## 硬體

| 項目 | 說明 |
|---|---|
| ESP32 + PCA9685 | 通道 0~5 是手臂 J1 底座 ~ J6 夾爪，通道 6、7 是排夾 J7、J8。序列埠 115200 |
| 示範臂 | 6 顆可變電阻，用手擺動來教從手臂動作 |
| 手牌架 | 手臂前方，分**第一排、第二排**兩區，共 13 格 |
| 排夾 J7 / J8 | 用伺服馬達夾住第一排 / 第二排的牌。J7：0° 開、90° 關；J8：90° 開、0° 關 |
| PASS 按鈕 | 自回彈按鈕，一腳接 ESP32 **GPIO 27**，另一腳接 **GND**（使用內建上拉，不用另外加電阻） |
| 手牌鏡頭 | 拍手牌架。手臂移到拍照點後拍照，一排拍一次 |
| 出牌鏡頭 | 拍架子前方的出牌區 |

### ESP32 韌體（firmware/esp32_arm/esp32_arm.ino）
用 Arduino IDE 開啟，板子選 **ESP32 Dev Module**，需要安裝函式庫 **Adafruit PWM Servo Driver Library**。

| 指令 | 作用 |
|---|---|
| `P,j1,...,j6` | 播放模式：直接設定六軸角度 |
| `s` | 緊急停止：手臂停在原地，**忽略所有 `P` 指令** |
| `r` | 解除停止並回到即時連動。⚠️ **手臂會移向示範臂當下的姿勢**，送出前先把示範臂擺好 |
| `h` | 解除停止但停在原位，不跟隨示範臂（main.py 的 `G` 用這個） |
| `C,j7,j8` / `C,OPEN` / `C,CLOSE` | 設定排夾角度 / 全開 / 全關。開機預設全開，緊急停止時不會動 |
| `?` | 查詢狀態，回傳 `STATE,<模式>,j1,...,j6,j7,j8` |

- **即時連動模式**（開機預設）：從手臂跟著示範臂動，並回傳 `DATA,j1,...,j6`。**只有這個模式能錄製。**
- **播放模式**：J1~J5 每個封包最多變化 10°。程式在兩個動作之間也會自動插值，每幀最多 8°。夾爪 J6 不限速，角度範圍和即時連動一樣是 5~180°。
- 格式錯誤的 `P` 封包會被拒絕，回傳 `ERR:BAD_PACKET`。舊版韌體遇到格式錯誤會把該軸當成 0°，讓手臂暴衝。
- Python 每播完一組動作，都會用 `?` 確認 ESP32 真的輸出到最後的角度。對不上就中止，並關閉全自動模式。
- 用 `record.py` 的 `check` 指令可以做自我測試：狀態查詢 → J1 轉 5° 再轉回 → 確認緊急停止時不會動 → 回到即時連動。
- ⚠️ 這裡的「確認到位」是確認 **ESP32 有送出對應的 PWM 訊號**。伺服馬達本身沒有角度回授，所以馬達卡住或沒電時偵測不到。

---

## 檔案結構

```
ES/
├── main.py              主程式（全自動對戰）
├── record.py            錄製動作 / 設定拍照點 / 測試辨識
├── config.py            ★ 共用設定都在這裡改
├── firmware/esp32_arm/  ESP32 韌體
├── pi/                  樹莓派 5 安裝與部署腳本
├── pi_servo.py          樹莓派版：I2C 直接驅動 PCA9685 (取代 ESP32)
├── table_display.py     觀眾畫面 (狀態燈、4 家剩餘張數、出牌區鏡頭)
├── discord_bridge.py    Discord 機器人 (/ 指令、後臺與牌局頻道)
├── vision.py            鏡頭、Gemini 辨識、分排拍照流程
├── robot_arm.py         手臂控制（播放、插值、回初始點、緊急停止）
├── big_two_model.py     A2C 網路架構
├── agents.py            ModelAgent（推論）
├── action_space.py      4551 個動作的定義與合法動作遮罩
├── big_two_core.py      牌型判斷 / 比大小
├── big_two_core_wrapper.py   C 語言加速核心的介面
├── big_two_c_lib.dll (Windows) / big_two_core_lib.so (樹莓派)，原始碼在 c_core/
├── Ai_mod/big_two_best.pth                        決策模型
├── actions/             錄好的手臂動作（record.py 會自動建立）
├── webcam_snapshots/    每次拍手牌的照片（除錯用）
└── 訓練資料/             訓練程式、YOLO 舊版、文件（上場不需要）
```

---

## 安裝

```powershell
python -m venv venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate
pip install -r requirements.txt
```

上場會用到的套件：`torch`、`numpy`、`opencv-python`、`pyserial`、`google-genai`。

### Gemini API Key
在 `ES/` 資料夾建一個 `.env` 檔，寫入一行：
```
GEMINI_API_KEY=你的key
```
程式啟動時會自動讀取。`.env` 裡是你的金鑰，**不要上傳或分享這個檔案**。

**模型**：程式會依序嘗試 `config.py` 的 `GEMINI_MODELS`（目前是 `gemini-3.5-flash` → `gemini-3.1-flash-lite` → `gemini-3.5-flash-lite`）。
- 某個模型忙線（503）或超過 `GEMINI_TIMEOUT_SEC`（25 秒）沒回應，就自動換下一個。成功的模型會記住，下次優先用。
- 想指定模型，可以在 `.env` 加一行 `GEMINI_MODEL=模型名稱`，它會排在最前面。
- 舊的 `gemini-2.5-flash` 已經不開放給新用戶。

---

## 設定（config.py）

| 設定 | 說明 |
|---|---|
| `COM_PORT` | ESP32 的連接埠。設成 `None` 是模擬模式，不接手臂也能測流程 |
| `HAND_CAMERA_INDEX` / `PLAY_CAMERA_INDEX` | 兩顆鏡頭的編號，用 `record.py` 的 `cams` 指令查 |
| `HAND_ZONES` | 手牌架分排：每排的拍照點名稱和張數，**加總必須是 13** |
| `HAND_ROI` / `PLAY_ROI` | 只辨識畫面中的某個區域 `(x, y, w, h)`，數值是 0~1 的比例。`None` 代表整張畫面 |
| `HAND_REVERSED` | 鏡頭畫面左右和格子順序相反時，改成 `True` |
| `PHOTO_SETTLE_SEC` | 手臂到拍照點後，等多久再拍照 |
| `CLAMP_OPEN` / `CLAMP_CLOSED` | 排夾 J7、J8 開和關的角度 |

**格子編號規則**：第一排由左到右是格子 1、2、3…，第二排接著往下編。「左右」是以**手牌鏡頭畫面**為準。

```python
HAND_ZONES = [
    {'label': 'Row 1', 'pose': 'photo_1', 'count': 7},   # 格子 1~7
    {'label': 'Row 2', 'pose': 'photo_2', 'count': 6},   # 格子 8~13
]
```

---

## 第一次設定：record.py

```powershell
python record.py
```
執行後會開兩個預覽視窗（HAND CAM / PLAY CAM），終端機用來輸入指令。

### 建議步驟
1. `cams`：查鏡頭編號，填進 `config.py` 後重新執行。
2. 擺好示範臂，輸入 `h`，存成**初始點**。
3. 輸入 `b`，跑批次設定：
   1. **拍照點** `photo_1`、`photo_2`：一邊看 HAND CAM，一邊移動示範臂，直到整排牌都入鏡、牌角清楚，再按 Enter。
   2. **夾牌** `pick_01` ~ `pick_13`：從初始點出發 → 到該格夾緊 → 提起，停在「拿牌姿勢」。13 格最好都停在同一個姿勢。
   3. **丟牌** `drop`：從拿牌姿勢開始，夾爪保持夾緊 → 往前甩 → 張開夾爪，讓牌落到出牌區。
   - 每個動作都是按 Enter 開始、按 Enter 結束。錄完可以 `p` 試播、`r` 重錄。開頭和結尾手臂沒在動的空檔，程式會自動剪掉。
4. 測試：
   - `t 5`：測試出第 5 格的牌
   - `v 1`：拍第一排並用 Gemini 辨識
   - `v all`：跑一次完整的手牌辨識
   - `v p`：辨識出牌區
5. `3`：檢查 main.py 需要的動作是否都齊全。

### 指令一覽
| 指令 | 作用 |
|---|---|
| `h` | 把目前姿勢設為初始點 |
| `c 排號` | 把目前姿勢設為該排的拍照點（例：`c 1`） |
| `b` | 批次設定：拍照點 → pick_01~13 → drop |
| `1 名稱` | 錄製單一動作（例：`1 pick_05`） |
| `2 名稱` | 刪除動作 |
| `3` | 列出動作，並檢查是否齊全 |
| `4 名稱` | 播放動作，播完自動回初始點 |
| `home` | 回到初始點 |
| `go 排號` | 移到該排拍照點並停住 |
| `t 格號` | 測試出某格牌：pick → drop → home |
| `v 排號` / `v all` / `v p` | 測試 Gemini 辨識 |
| `snap` | 兩顆鏡頭各存一張照片 |
| `clamp open` / `clamp close` | 排夾開 / 關。也可以用 `clamp 45 45` 直接指定 J7、J8 的角度 |
| `check` | ESP32 自我測試（J1 會轉 5° 再轉回） |
| （按 PASS 按鈕） | 終端機會印出「🔘 PASS 按鈕被按下」，可用來確認接線 |
| `cams` | 列出可用的鏡頭編號 |
| `s` / `r` | 緊急停止 / 解除停止並回到即時連動 |
| `exit` | 離開 |

### 需要的動作檔（actions/）
| 名稱 | 內容 |
|---|---|
| `home` | 初始點（單一姿勢） |
| `photo_1`, `photo_2` | 各排的拍照點（單一姿勢） |
| `pick_01` ~ `pick_13` | 夾第 N 格的牌 |
| `drop` | 往前丟牌 |

---

## 上場：main.py

```powershell
python main.py
```
樹莓派上直接點**桌面的「大老二 AI」**捷徑。

執行後：
- 手臂會**直接移到初始點並停住**，不會跟隨示範臂。
- 螢幕全螢幕顯示**觀眾畫面**（見下方）。
- 所有操作都用 **Discord 的 `/` 指令**。

### 觀眾畫面（外接螢幕）
只顯示公開資訊，像遊戲畫面一樣給所有人看：
- **狀態燈**：🟢 全自動正常進行 / 🟡 處理中、還沒開始全自動、等你確認 / 🔴 需要人處理（緊急停止、手臂或出牌區鏡頭沒接上）
- **4 家剩餘張數**（牌背 + 數字）。輪到誰，誰的數字就會發光；這一回合 Pass 的人會標 `PASS`；獲勝的人有皇冠
- **出牌區已出過的總張數**（中間）
- **出牌區鏡頭**即時畫面（右下）

畫面**不會顯示任何人的手牌**，AI 的手牌只能在 Discord 用 `/status` 看（只有你自己看得到）。

### Discord 設定
1. 在 `.env` 加一行 `DISCORD_TOKEN=你的bot token`（Windows 和樹莓派各自的 `.env` 都要加）。
2. 機器人在伺服器需要這些權限：**傳送訊息、讀取訊息歷史、管理訊息**（清空牌局頻道用），邀請時要勾 `applications.commands`。
3. 頻道和身分組在 `config.py` 設定：
   - 後臺頻道：`1567535190211829822`，收所有系統訊息與指令回覆。
   - 牌局頻道：`1567535349456961647`，收公開的出牌紀錄，開新局時會清空。
   - 主人身分組：`1567531351337345044`，只有擁有這個身分組的人才能用指令。
4. 指令只會同步到這個伺服器，不會動到同一個 bot 的全域指令。

### 開局流程
1. `/newgame`：清空牌局頻道、重設狀態、打開排夾。
2. 13 張牌插進手牌架（第一排、第二排）。
3. `/start`：手臂依序到 `photo_1`、`photo_2` 拍照並辨識每一格的牌，回初始點後**排夾關閉**。辨識完整就自動進入全自動。
   - AI 沒有梅花 3 時，可以直接用 `/start first:P2` 指定先出的人，或之後再用 `/first`。
   - 辨識不完整時，用 `/status` 查看、`/fix` 修正、`/confirm` 確認，再 `/start`。

### 全自動流程
- **輪到 AI**：AI 決策 → 手臂依序把每一張牌 `pick_XX → drop → home`。
- **輪到對手**：出牌區畫面有變化並靜止 1 秒後，才做一次影像辨識。看到的牌扣掉整局已經出過的牌，剩下的就是新出的牌。新的牌必須壓得過上一手才會採用，避免辨識錯誤把狀態弄亂。
- **對手 Pass**：對手按 **PASS 按鈕**（或用 `/pass`）立即判定。出牌區 15 秒沒有動靜也會自動判定 Pass。只想用按鈕的話，把 `PASS_TIMEOUT_SEC` 設成 `None`。
  - 按鈕只在對手回合有效；輪到 AI、手臂動作中或拍照中按下都會被忽略。0.5 秒內連按只算一次。
- 每一手出牌、Pass、新回合、獲勝都會公開貼到牌局頻道，例如「P2 出牌 ♠9 ♥9（剩 11 張）」。

### 對手須知
- 牌要**正面朝上**放進出牌區，疊在上一手上面也可以。
- 放好後手要移開，畫面靜止後才會開始辨識。

### / 指令
| 指令 | 作用 |
|---|---|
| `/newgame` | 開新局：清空牌局頻道、重設、打開排夾 |
| `/start [first]` | 拍照辨識手牌並開始全自動（已鎖定手牌時直接開始） |
| `/first player` | AI 沒有梅花 3 時，指定誰先出 |
| `/scan` | 只拍照辨識手牌，不自動開始 |
| `/fix slot card` | 修正某一格，例如 `slot:5 card:SA` |
| `/confirm` | 確認手牌 |
| `/auto on` | 全自動開 / 關 |
| `/pass` | 現在輪到的對手 Pass |
| `/read`、`/submit` | 手動模式：辨識出牌區 → 當成對手這一手 |
| `/aiplay` | 手動模式：輪到 AI 時讓 AI 出牌 |
| `/stop` | **緊急停止** |
| `/resume` | 解除停止並回初始點 |
| `/clamp action` | 排夾開 / 關（全自動中不能開） |
| `/status` | 系統狀態與 AI 手牌（只有你看得到） |
| `/shutdown` | 關閉主程式 |

所有回覆都只有下指令的人看得到。

**現場緊急停止**：Discord 會有延遲，所以觀眾畫面保留一個實體按鍵：按 **`Esc`** 立即緊急停止。

### 影像辨識的紀錄
終端機和 Discord 只顯示「[影像辨識]」的結果。辨識服務的技術細節（回應時間、錯誤、重試）寫在本機的 `logs/vision.log`，需要除錯時再看。

### 可調參數（main.py 開頭）
| 參數 | 預設 | 說明 |
|---|---|---|
| `MOTION_THRESH` | 4.0 | 畫面差異大於這個值，代表有東西在動 |
| `CHANGE_THRESH` | 10.0 | 畫面靜止後，和上次比差異大於這個值，才呼叫 Gemini |
| `STABLE_SEC` | 1.0 | 畫面要靜止多久，才判定牌已放好 |
| `PASS_TIMEOUT_SEC` | 15.0 | 對手回合多久沒動靜，判定為 Pass。設 `None` 就只用 PASS 按鈕 / `Space` |
| `NORMALIZE_OPP_COUNTS` | False | 對手剩餘牌數要不要除以 13（要和訓練時的設定一致） |

---

## 牌名規則

花色字母 + 點數字母，固定 2 個字元。Gemini 只能回傳這 52 種名稱。

- 花色：`C` 梅花、`D` 方塊、`H` 紅心、`S` 黑桃
- 點數（小 → 大）：`3 4 5 6 7 8 9 T J Q K A 2`（`T` = 10）
- 例：`C3` 梅花 3、`DT` 方塊 10、`SA` 黑桃 A、`H2` 紅心 2

內部編號 1~52 = 花色 × 13 + 點數 + 1（梅花 3 = 1，黑桃 2 = 52）。

---

## 常見問題

| 狀況 | 處理 |
|---|---|
| 鏡頭開不起來 | 用 `record.py` 的 `cams` 查編號，再改 `config.py` |
| 手牌少算或多算一張 | 到 `webcam_snapshots/` 看照片，調整拍照點（`c 1`）或 `HAND_ROI`，讓牌角清楚 |
| 手臂一入鏡就被當成有人出牌 | 設定 `PLAY_ROI`，只框出牌區 |
| 對手出牌沒被偵測到 | 調低 `CHANGE_THRESH`，或改用 `P` + `Enter` 手動輸入 |
| 還沒出牌就被判 Pass | 調高 `PASS_TIMEOUT_SEC` |
| 錄製時沒有錄到點位 | ESP32 要在即時連動模式。record.py 錄製前會自動送 `r` |
| 手臂抖或動作銜接太猛 | 調低 `robot_arm.py` 的 `MAX_STEP_DEG` |
| 牌丟不夠遠 | 重錄 `drop` 時甩快一點，或改成伸到出牌區上方再放開 |

---

## 在樹莓派 5 上執行

樹莓派版**不用 ESP32 和示範臂**：樹莓派用 I2C 直接驅動 PCA9685，PASS 按鈕直接接樹莓派的 GPIO。
動作檔、角度換算、限速、緊急停止、排夾，都和 ESP32 版完全相同（[pi_servo.py](pi_servo.py)）。
- 動作要在 **Windows + ESP32 + 示範臂**錄好，再複製過來。樹莓派版不能錄製。
- 同一份程式碼兩邊都能跑：C 核心會自動選 `.dll` 或 `.so`，手臂在 Windows 走 ESP32 序列埠，在樹莓派走 I2C。

### 接線（樹莓派 40pin）
| PCA9685 / 按鈕 | 樹莓派 |
|---|---|
| VCC | 3.3V（pin 1） |
| GND | GND（pin 6） |
| SDA | GPIO2（pin 3） |
| SCL | GPIO3（pin 5） |
| V+（伺服電源） | **外接 5~6V 電源**，電源的 GND 要和樹莓派共地。不可以從樹莓派取電 |
| PASS 按鈕 | GPIO17（pin 11）和 GND（pin 9），使用內建上拉 |

### 每台電腦各自的設定（寫在該台的 `.env`）
```
GEMINI_API_KEY=你的key
COM_PORT=i2c                  # 樹莓派預設 i2c；Windows 預設 auto (自動找 ESP32)
HAND_CAMERA=/dev/v4l/by-id/usb-xxxx-video-index0
PLAY_CAMERA=/dev/v4l/by-id/usb-yyyy-video-index0
```
樹莓派的鏡頭建議用 `/dev/v4l/by-id/...` 路徑，插拔或換 USB 孔也不會對調。路徑用 `record.py` 的 `cams` 指令查。

### 第一次安裝
1. 在 Windows 的 ES 資料夾執行，把程式複製過去：
   ```powershell
   powershell -File pi\deploy.ps1 -PiHost 10.193.1.201
   ```
   - 樹莓派上還沒有 `.env` 時會一併複製。之後要更新金鑰，加上 `-CopyEnv`。
2. **系統設定**（要 sudo，在樹莓派上自己執行一次）：開啟 I2C、風扇一開機就全速、安裝系統套件。
   ```bash
   sudo bash ~/ES/pi/system_setup.sh
   sudo reboot
   ```
   重開後執行 `i2cdetect -y 1`，PCA9685 應該出現在 `40`。原本的 `config.txt` 會先備份成 `config.txt.bak-日期`。
3. **Python 環境**（不用 sudo）：
   ```bash
   cd ~/ES && bash pi/setup_pi.sh
   ```
   這會安裝 Python 套件（含 CPU 版 PyTorch）、重新編譯 C 核心，最後做一次檢查。

### 第一次上電要注意
- 伺服板**剛上電時**，程式不知道手臂在哪。第一個動作（回初始點）會**直接跳到定位**，啟動前手要離開手臂。
- 之後只要伺服板一直有電，程式重開時會讀回上次的角度，從原位平滑出發。

### 執行
點**樹莓派桌面上的「大老二 AI」**捷徑：會開一個終端機視窗，觀眾畫面則全螢幕顯示。程式結束後，按 Enter 關閉終端機。
也可以在樹莓派桌面的終端機執行：
```bash
bash ~/ES/pi/start.sh
```
之後改了程式，在 Windows 再跑一次 `pi\deploy.ps1` 就會更新（桌面捷徑也會一起更新）。

---

## 重新訓練 AI

訓練相關檔案都在 `訓練資料/AI決策訓練/`，需要另外安裝 `gymnasium`：
```powershell
pip install gymnasium
python 訓練資料/AI決策訓練/train_a2c.py
```
訓練出來的模型會存成 `big_two_a2c_epXXXX.pth`。挑表現最好的一個，複製成 `Ai_mod/big_two_best.pth`。

> 狀態向量為 113 維：手牌 52、上一手 52、對手剩餘牌數 3、最後出牌者 4、連續 Pass 數 1、是否新回合 1。
