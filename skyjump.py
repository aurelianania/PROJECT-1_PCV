"""
Game lompat awan dengan kontrol tangan bersarung tangan biru (setengah layar kanan kamera).
Tombol: r = restart, q = keluar.
File gambar (satu folder dengan file ini): player.png, hell.png, land.jpg, sky.jpg,
space.jpg, heaven.jpg
"""

import os
import random
import time

import cv2
import numpy as np

# ============================================================== KONFIGURASI
CAM_INDEX = 0
CAM_W, CAM_H = 640, 480
LOWER_BLUE = np.array([100, 80, 60])      # rentang HSV sarung tangan biru
UPPER_BLUE = np.array([125, 255, 255])
MIN_AREA = 500                            # kontur lebih kecil dari ini dianggap noise
SMOOTH = 0.3                              # makin kecil makin halus tapi makin delay
EDGE_MARGIN = 0.15                        # tangan tidak perlu mentok tepi area

GAME_W, GAME_H = 480, 640
STEP = 1 / 60.0                           # fisika 60 langkah/detik
GRAVITY = 0.6
JUMP_V = -15.0
SPRING_V = -25.0
SCROLL_LINE = GAME_H * 0.4
GROUND_Y = GAME_H - 40

# (file background, nama, skor mulai) dari bawah ke atas
STAGES = [
    ("hell.png",   "HELL",   0),
    ("land.jpg",   "LAND",   300),
    ("sky.jpg",    "SKY",    600),
    ("space.jpg",  "SPACE",  900),
    ("heaven.jpg", "HEAVEN", 1200),
]
GATE_SCORE = 1500                         # skor saat pintu surga muncul
GATE_WORLD = GATE_SCORE * 10 + (GROUND_Y - SCROLL_LINE)
BANNER_STEPS = 120                        # lama tulisan nama stage (2 detik)

PLAYER_IMG = "player.png"
PLAYER_H = 60                             # tinggi pemain di layar (lebar ikut rasio)
DOOR_W, DOOR_H = 60, 90

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def ramp(v, start, full):
    return clamp((v - start) / float(full - start), 0.0, 1.0)


# ================================================================== GAMBAR
def load_image(name):
    img = cv2.imread(os.path.join(BASE_DIR, name), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError("File '%s' tidak ada di folder %s" % (name, BASE_DIR))
    return img


def load_background(name):
    """Skala + crop tengah supaya pas ukuran layar game tanpa penyet."""
    img = load_image(name)[:, :, :3]
    h, w = img.shape[:2]
    s = max(GAME_W / float(w), GAME_H / float(h))
    img = cv2.resize(img, (int(np.ceil(w * s)), int(np.ceil(h * s))))
    y0, x0 = (img.shape[0] - GAME_H) // 2, (img.shape[1] - GAME_W) // 2
    return np.ascontiguousarray(img[y0:y0 + GAME_H, x0:x0 + GAME_W])


def load_player(name, height):
    img = load_image(name)
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(round(w * height / float(h)))), height),
                      interpolation=cv2.INTER_AREA)


def overlay(img, sprite, x, y):
    """Tempel sprite (BGR/BGRA) dengan pojok kiri-atas di (x, y); aman di tepi layar."""
    sh, sw = sprite.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, img.shape[1]), min(y + sh, img.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    s = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = img[y0:y1, x0:x1]
    if s.shape[2] == 4:
        a = s[:, :, 3:4].astype(np.float32) / 255.0
        roi[:] = (s[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)
    else:
        roi[:] = s[:, :, :3]


# ============================================================ DETEKSI TANGAN
def detect_hand(frame, kernel):
    """
    Cari sarung tangan biru di setengah kanan frame.
    Return (posisi X 0..1 di area kanan atau None kalau tidak ada, mask).
    """
    h, w = frame.shape[:2]
    x0 = w // 2
    cv2.rectangle(frame, (x0, 0), (w - 1, h - 1), (0, 255, 255), 2)   # kotak area aktif

    roi = frame[:, x0:]
    hsv = cv2.cvtColor(cv2.GaussianBlur(roi, (7, 7), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
    if not contours:
        return None, mask

    c = max(contours, key=cv2.contourArea)
    m = cv2.moments(c)
    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    bx, by, bw, bh = cv2.boundingRect(c)
    cv2.rectangle(frame, (bx + x0, by), (bx + bw + x0, by + bh), (0, 255, 0), 2)
    cv2.circle(frame, (cx + x0, cy), 8, (0, 0, 255), -1)

    pos = (cx / float(w - x0) - EDGE_MARGIN) / (1 - 2 * EDGE_MARGIN)
    return clamp(pos, 0.0, 1.0), mask


# ==================================================================== AWAN
class Platform:
    def __init__(self, x, y, w, kind="normal", vx=0.0):
        self.x, self.y, self.w = x, y, w   # x = pusat, y = permukaan atas awan
        self.kind = kind                   # normal / moving / breakable / spring / gate
        self.vx = vx

    def update(self):
        if self.kind == "moving":
            self.x += self.vx
            half = self.w / 2
            if self.x < half or self.x > GAME_W - half:
                self.x = clamp(self.x, half, GAME_W - half)
                self.vx = -self.vx


CLOUD_COLORS = {                      # (warna badan, warna outline) BGR
    "normal":    ((255, 255, 255), (215, 215, 215)),
    "moving":    ((150, 210, 255), (110, 170, 230)),
    "breakable": ((170, 175, 185), (130, 135, 145)),
    "spring":    ((255, 255, 255), (215, 215, 215)),
    "gate":      ((190, 235, 255), (90, 190, 240)),
}


def draw_door(img, x, yb):
    hw = DOOR_W // 2
    cy = yb - DOOR_H + hw
    gold, pale = (60, 190, 240), (200, 245, 255)
    cv2.rectangle(img, (x - hw, cy), (x + hw, yb), gold, -1)
    cv2.ellipse(img, (x, cy), (hw, hw), 0, 180, 360, gold, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x - hw + 6, cy), (x + hw - 6, yb), pale, -1)
    cv2.ellipse(img, (x, cy), (hw - 6, hw - 6), 0, 180, 360, pale, -1, cv2.LINE_AA)
    cv2.line(img, (x, cy - hw + 6), (x, yb), gold, 2, cv2.LINE_AA)


def draw_cloud(img, p):
    x, y, w = int(p.x), int(p.y), int(p.w)
    col, edge = CLOUD_COLORS[p.kind]
    if p.kind == "gate":
        draw_door(img, x, y + 4)
    # outline lalu badan awan
    cv2.ellipse(img, (x, y + 6), (w // 2 + 2, 10), 0, 0, 360, edge, -1, cv2.LINE_AA)
    cv2.circle(img, (x - w // 5, y + 3), 11, edge, -1, cv2.LINE_AA)
    cv2.circle(img, (x + w // 6, y + 2), 13, edge, -1, cv2.LINE_AA)
    cv2.ellipse(img, (x, y + 7), (w // 2, 8), 0, 0, 360, col, -1, cv2.LINE_AA)
    cv2.circle(img, (x - w // 5, y + 3), 9, col, -1, cv2.LINE_AA)
    cv2.circle(img, (x + w // 6, y + 2), 11, col, -1, cv2.LINE_AA)

    if p.kind == "breakable":      # garis retak
        for a, b in (((-8, -4), (-2, 6)), ((-2, 6), (5, 0)), ((5, 0), (10, 8))):
            cv2.line(img, (x + a[0], y + a[1]), (x + b[0], y + b[1]),
                     (80, 90, 100), 2, cv2.LINE_AA)
    elif p.kind == "spring":       # per merah
        pts = np.array([(x - 6, y), (x + 6, y - 3), (x - 6, y - 6), (x + 6, y - 9)], np.int32)
        cv2.polylines(img, [pts], False, (60, 60, 60), 2, cv2.LINE_AA)
        cv2.rectangle(img, (x - 9, y - 13), (x + 9, y - 9), (0, 0, 220), -1)


# ===================================================================== GAME
class Game:
    def __init__(self):
        self.best = 0
        self.sprite = load_player(PLAYER_IMG, PLAYER_H)
        self.pw, self.ph = self.sprite.shape[1], PLAYER_H
        self.backgrounds = [load_background(s[0]) for s in STAGES]
        self.reset()

    def reset(self):
        self.started = self.over = self.won = self.gate_spawned = False
        self.height, self.score, self.stage, self.banner = 0.0, 0, 0, 0
        self.px, self.py, self.vy = GAME_W / 2, GROUND_Y, JUMP_V   # py = posisi kaki
        self.platforms = [Platform(GAME_W / 2, GROUND_Y, 150)]
        self.spawn()

    def spawn(self):
        """Buat awan baru di atas layar. Setelah pintu surga dibuat, tidak ada awan lagi."""
        if self.gate_spawned:
            return
        top = min(p.y for p in self.platforms)
        while top > -60:
            level = clamp(self.score / float(GATE_SCORE), 0.0, 1.0)
            new_top = top - random.uniform(55 + 25 * level, 85 + 55 * level)

            if self.height + GROUND_Y - new_top >= GATE_WORLD:      # waktunya pintu surga
                self.platforms.append(Platform(random.uniform(120, GAME_W - 120),
                                               new_top, 200, "gate"))
                self.gate_spawned = True
                return
            top = new_top

            w = random.randint(int(90 - 30 * level), int(100 - 25 * level))
            p_spring = 0.07
            p_break = 0.25 * ramp(self.score, 200, 1000)
            p_move = 0.35 * ramp(self.score, 300, 1000)
            r = random.random()
            kind = ("spring" if r < p_spring else
                    "breakable" if r < p_spring + p_break else
                    "moving" if r < p_spring + p_break + p_move else "normal")
            vx = random.choice([-1, 1]) * (1.2 + 1.8 * level) if kind == "moving" else 0.0
            self.platforms.append(Platform(random.uniform(w / 2, GAME_W - w / 2),
                                           top, w, kind, vx))

    def step(self, hand_x, found):
        if not self.started:
            if not found:
                return
            self.started, self.banner = True, BANNER_STEPS
        if self.over or self.won:
            return
        self.banner = max(0, self.banner - 1)

        for p in self.platforms:
            p.update()

        # X pemain mengikuti tangan, Y dengan gravitasi
        self.px = clamp(hand_x * GAME_W, self.pw / 2, GAME_W - self.pw / 2)
        prev = self.py
        self.vy += GRAVITY
        self.py += self.vy

        # mendarat di awan (hanya saat jatuh, naik bisa tembus dari bawah)
        if self.vy > 0:
            for p in self.platforms:
                if (abs(self.px - p.x) <= p.w / 2 + self.pw * 0.25
                        and prev <= p.y + 1 and self.py >= p.y):
                    self.py = p.y
                    self.vy = SPRING_V if p.kind == "spring" else JUMP_V
                    if p.kind == "breakable":
                        self.platforms.remove(p)       # langsung hilang
                    break

        # menang kalau menyentuh pintu surga
        for p in self.platforms:
            if (p.kind == "gate" and abs(self.px - p.x) < (DOOR_W + self.pw * 0.6) / 2
                    and p.y - DOOR_H < self.py and self.py - self.ph < p.y):
                self.won = True
                return

        # scroll: pemain tidak naik melewati SCROLL_LINE, dunia yang turun
        if self.py < SCROLL_LINE:
            dy = SCROLL_LINE - self.py
            self.py = SCROLL_LINE
            for p in self.platforms:
                p.y += dy
            self.height += dy
            self.score = int(self.height / 10)
            self.best = max(self.best, self.score)

        # ganti stage sesuai skor
        stage = max(i for i, s in enumerate(STAGES) if self.score >= s[2])
        if stage != self.stage:
            self.stage, self.banner = stage, BANNER_STEPS

        self.platforms = [p for p in self.platforms if p.y < GAME_H + 40]
        self.spawn()

        if self.py - self.ph > GAME_H:                 # jatuh keluar layar
            self.over = True

    # ------------------------------------------------------------- gambar
    @staticmethod
    def text(img, s, y, scale=0.8, color=(255, 255, 255), x=None):
        (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        org = ((GAME_W - tw) // 2 if x is None else x, y)
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    def draw(self, found):
        img = self.backgrounds[self.stage].copy()
        for p in self.platforms:
            draw_cloud(img, p)
        overlay(img, self.sprite, int(self.px - self.pw / 2), int(self.py - self.ph))

        self.text(img, "Skor: %d" % self.score, 28, 0.7, x=10)
        self.text(img, "Terbaik: %d" % self.best, 28, 0.6, x=GAME_W - 160)

        if not self.started:
            self.text(img, "Tunjukkan tangan biru", GAME_H // 2 - 10)
            self.text(img, "untuk mulai", GAME_H // 2 + 25)
        elif not (self.over or self.won):
            if self.banner > 0:
                self.text(img, "STAGE %d: %s" % (self.stage + 1, STAGES[self.stage][1]),
                          90, 0.9, (60, 220, 255))
            if not found:
                self.text(img, "Tangan tidak terdeteksi", 60, 0.6, (80, 80, 255))

        if self.over or self.won:
            img = (img * 0.5).astype(np.uint8)
            title, col = (("GAME OVER", (80, 80, 255)) if self.over
                          else ("KAMU MENANG!", (60, 220, 255)))
            self.text(img, title, GAME_H // 2 - 30, 1.2, col)
            self.text(img, "Skor: %d" % self.score, GAME_H // 2 + 15, 0.9)
            self.text(img, "Tekan R untuk main lagi", GAME_H // 2 + 55, 0.7)
        return img


# ===================================================================== MAIN
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print("Kamera tidak bisa dibuka. Coba ubah CAM_INDEX.")
        return

    game = Game()
    kernel = np.ones((5, 5), np.uint8)
    hand_x, acc, prev_t = 0.5, 0.0, time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(cv2.resize(frame, (CAM_W, CAM_H)), 1)   # mirror

        raw, mask = detect_hand(frame, kernel)
        found = raw is not None
        if found:
            hand_x += SMOOTH * (raw - hand_x)

        now = time.perf_counter()
        acc += min(now - prev_t, 0.1)
        prev_t = now
        while acc >= STEP:
            game.step(hand_x, found)
            acc -= STEP

        cv2.imshow("Game", game.draw(found))
        cv2.imshow("Kamera", frame)
        cv2.imshow("Mask Biru", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            game.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()