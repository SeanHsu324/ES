"""
樹莓派直接驅動 PCA9685 (I2C)，取代 ESP32。

PiServoLink 模擬 ESP32 韌體的序列埠協定 (P / s / r / h / ? / C / BTN:PASS)，
所以 robot_arm.py 的到位確認、緊急停止、排夾、自我測試都不用改。
角度 -> 脈衝的換算、角度限制、播放限速都和韌體 (firmware/esp32_arm) 完全相同，錄好的動作可以直接用。

接線 (樹莓派 40pin)：
  PCA9685 VCC -> 3.3V (pin 1)    GND -> GND (pin 6)
  PCA9685 SDA -> GPIO2 (pin 3)   SCL -> GPIO3 (pin 5)
  PCA9685 V+  -> 伺服馬達外接電源 (5~6V)，電源 GND 要和樹莓派共地
  PASS 按鈕   -> GPIO17 (pin 11) 與 GND (pin 9)，使用內建上拉電阻
"""
import time
import queue
import threading

# ---- 與韌體相同的參數 ----
SERVOMIN, SERVOMAX = 125, 590
MIN_ANGLE, MAX_ANGLE = 5, 175
J6_OUTPUT_MAX = 180
PLAYBACK_MAX_STEP = 10
CLAMP_CH = (6, 7)
CLAMP_OPEN = (0, 90)
CLAMP_CLOSE = (90, 0)
PWM_FREQ = 50
OSC_FREQ = 25_000_000


def angle_to_pulse(angle):
    # 和 Arduino map(angle, 0, 180, SERVOMIN, SERVOMAX) 一樣用整數除法
    return angle * (SERVOMAX - SERVOMIN) // 180 + SERVOMIN


def pulse_to_angle(pulse):
    return round((pulse - SERVOMIN) * 180 / (SERVOMAX - SERVOMIN))


class PCA9685:
    MODE1, PRESCALE, LED0_ON_L = 0x00, 0xFE, 0x06
    SLEEP, AI, RESTART = 0x10, 0x20, 0x80

    def __init__(self, bus=1, address=0x40):
        from smbus2 import SMBus
        self.bus = SMBus(bus)
        self.addr = address
        self.prescale = int(OSC_FREQ / (PWM_FREQ * 4096.0) + 0.5) - 1   # 50Hz -> 121 (和 Adafruit 函式庫相同)

    def _r(self, reg):
        return self.bus.read_byte_data(self.addr, reg)

    def _w(self, reg, val):
        self.bus.write_byte_data(self.addr, reg, val)

    def is_running(self):
        """PCA9685 已經在輸出 50Hz (上次程式結束後伺服板一直有電)"""
        mode1 = self._r(self.MODE1)
        return not (mode1 & self.SLEEP) and self._r(self.PRESCALE) == self.prescale

    def begin(self):
        old = self._r(self.MODE1) & ~self.RESTART
        self._w(self.MODE1, (old & 0x7F) | self.SLEEP)       # 改頻率前要先睡眠
        self._w(self.PRESCALE, self.prescale)
        self._w(self.MODE1, old & ~self.SLEEP)
        time.sleep(0.005)
        self._w(self.MODE1, (old & ~self.SLEEP) | self.RESTART | self.AI)

    def set_pwm(self, ch, on, off):
        self.bus.write_i2c_block_data(self.addr, self.LED0_ON_L + 4 * ch,
                                      [on & 0xFF, on >> 8, off & 0xFF, off >> 8])

    def get_off(self, ch):
        d = self.bus.read_i2c_block_data(self.addr, self.LED0_ON_L + 4 * ch, 4)
        if d[1] & 0x10 or d[3] & 0x10:     # full on / full off 旗標
            return None
        return d[2] | (d[3] << 8)


class PiServoLink:
    """假裝成 ESP32 的序列埠：write() 收指令、readline() 回訊息"""

    def __init__(self, bus=1, address=0x40, pass_button_gpio=17):
        self.pca = PCA9685(bus, address)
        self.out = queue.Queue()
        self.lock = threading.Lock()
        self.emergency = False
        self.playback = True          # 沒有示範臂，永遠是播放模式
        self.stop_reported = False
        self.angles = [None] * 6      # None = 不知道手臂目前在哪 (伺服板剛上電)

        if self.pca.is_running():
            # 伺服板一直有電：讀回上次輸出的角度，手臂可以從原位平滑出發
            for i in range(6):
                off = self.pca.get_off(i)
                self.angles[i] = pulse_to_angle(off) if off else None
        else:
            self.pca.begin()
        if None in self.angles:
            self.angles = [None] * 6
        self.clamps = list(CLAMP_OPEN)
        self._write_clamps(CLAMP_OPEN)  # 和韌體一樣：開機排夾全開

        self.button = None
        if pass_button_gpio is not None:
            try:
                from gpiozero import Button
                self.button = Button(pass_button_gpio, pull_up=True, bounce_time=0.03)
                self.button.when_pressed = lambda: self.out.put(b"BTN:PASS\n")
            except Exception as e:
                print(f"[樹莓派] ⚠️ PASS 按鈕 (GPIO{pass_button_gpio}) 無法使用: {e}")
        self.out.put(b"READY\n")

    # ---------------- 序列埠介面 ----------------
    def readline(self):
        try:
            return self.out.get(timeout=1)
        except queue.Empty:
            return b""

    def write(self, data):
        for line in data.decode('utf-8', 'ignore').splitlines():
            with self.lock:
                self._handle(line.strip())

    def close(self):
        if self.button:
            self.button.close()
        self.pca.bus.close()

    def query(self):
        """給 RobotArm 直接查狀態 (不經過佇列)：(模式, 角度或 None)"""
        with self.lock:
            mode = 'EMERGENCY_STOP' if self.emergency else 'PLAYBACK'
            angles = None if self.angles[0] is None else list(self.angles)
            return mode, angles, list(self.clamps)

    # ---------------- 指令 ----------------
    def _say(self, text):
        self.out.put((text + "\n").encode())

    def _handle(self, cmd):
        if not cmd:
            return
        low = cmd.lower()
        if low in ('s', 'stop'):
            self.emergency = True
            self.stop_reported = False
            self._say("STATUS:EMERGENCY_STOP")
        elif low in ('r', 'run', 'h', 'hold'):
            # 沒有示範臂可以跟隨，r 和 h 一樣：解除停止、停在原位
            self.emergency = False
            self._say("STATUS:PLAYBACK")
        elif cmd == '?':
            mode, angles, clamps = self.query()
            vals = angles or [0] * 6
            self._say("STATE," + mode + "," + ",".join(map(str, vals + clamps)))
        elif cmd.upper().startswith('P,'):
            self._playback(cmd)
        elif cmd.upper().startswith('C,'):
            self._clamp(cmd)
        else:
            self._say("ERR:UNKNOWN " + cmd)

    @staticmethod
    def _parse(cmd, n):
        parts = cmd.split(',')[1:]
        if len(parts) != n:
            return None
        try:
            return [int(p) for p in parts]
        except ValueError:
            return None

    def _playback(self, cmd):
        if self.emergency:
            if not self.stop_reported:
                self._say("ERR:STOPPED")
                self.stop_reported = True
            return
        a = self._parse(cmd, 6)
        if a is None:
            self._say("ERR:BAD_PACKET " + cmd)
            return
        first = self.angles[0] is None
        for i in range(6):
            v = max(MIN_ANGLE, min(J6_OUTPUT_MAX if i == 5 else MAX_ANGLE, a[i]))
            if i < 5 and not first:   # 不知道起點時 (剛上電) 只能直接到位
                last = self.angles[i]
                v = max(last - PLAYBACK_MAX_STEP, min(last + PLAYBACK_MAX_STEP, v))
            self.pca.set_pwm(i, 0, angle_to_pulse(v))
            self.angles[i] = v

    def _write_clamps(self, a):
        for i in range(2):
            self.clamps[i] = max(0, min(180, a[i])) if hasattr(self, 'clamps') else a[i]
            self.pca.set_pwm(CLAMP_CH[i], 0, angle_to_pulse(self.clamps[i]))

    def _clamp(self, cmd):
        if self.emergency:
            self._say("ERR:STOPPED")
            return
        up = cmd.upper()
        if up == 'C,OPEN':
            a = list(CLAMP_OPEN)
        elif up == 'C,CLOSE':
            a = list(CLAMP_CLOSE)
        else:
            a = self._parse(cmd, 2)
            if a is None:
                self._say("ERR:BAD_PACKET " + cmd)
                return
        self._write_clamps(a)
        self._say(f"STATUS:CLAMP,{self.clamps[0]},{self.clamps[1]}")
