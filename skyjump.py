import os
import random
import time

import cv2
import numpy as np

#konfig
CAM_INDEX = 0
LOWER_BLUE = np.array([100, 80, 60])
UPPER_BLUE = np.array([125, 255, 255])
MIN_AREA = 500
SMOOTH = 0.3
EDGE_MARGIN = 0.15

GAME_W, GAME_H = 480, 640
GRAVITY, JUMP, SPRING = 0.6, -15.0, -25.0
SCROLL_LINE = GAME_H * 0.4
GROUND_Y = GAME_H - 40

#(file background, nama stage, skor mulai)
STAGES = [
    ("underground.jpg",   "UNDERGROUND",   0),
    ("ground.jpg",   "GROUND",   300),
    ("sky.jpg",    "SKY",    600),
    ("space.jpg",  "SPACE",  900),
    ("heaven.jpeg", "HEAVEN", 1200),
]
GATE_SCORE = 1500
GATE_WORLD = GATE_SCORE * 10 + (GROUND_Y - SCROLL_LINE)
FADE_STEPS = 60
BANNER_STEPS = 120
PLAYER_H, DOOR_H = 60, 90

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()



def clamp(v, low, high):
    return max(low, min(high, v))


def load(name, height=None):
    img = cv2.imread(os.path.join(BASE_DIR, name), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError("File '%s' tidak ada di folder %s" % (name, BASE_DIR))
    if height:
        img = cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height))
    return img


def load_background(name):
    img = load(name)[:, :, :3]
    h, w = img.shape[:2]
    s = max(GAME_W / w, GAME_H / h)
    img = cv2.resize(img, (int(w * s) + 1, int(h * s) + 1))
    y, x = (img.shape[0] - GAME_H) // 2, (img.shape[1] - GAME_W) // 2
    return img[y:y + GAME_H, x:x + GAME_W].copy()


def overlay(canvas, sprite, x, y):
    sh, sw = sprite.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, canvas.shape[1]), min(y + sh, canvas.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    part = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    area = canvas[y0:y1, x0:x1]
    if part.shape[2] == 4:
        alpha = part[:, :, 3:4] / 255.0
        area[:] = part[:, :, :3] * alpha + area * (1 - alpha)
    else:
        area[:] = part[:, :, :3]


#deteksi tangan
def detect_hand(frame, kernel):
    h, w = frame.shape[:2]
    x0 = w // 2
    cv2.rectangle(frame, (x0, 0), (w - 1, h - 1), (0, 255, 255), 2)

    roi = frame[:, x0:]
    hsv = cv2.cvtColor(cv2.GaussianBlur(roi, (7, 7), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
    if not contours:
        return None, mask

    biggest = max(contours, key=cv2.contourArea)
    m = cv2.moments(biggest)
    cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    bx, by, bw, bh = cv2.boundingRect(biggest)
    cv2.rectangle(frame, (bx + x0, by), (bx + bw + x0, by + bh), (0, 255, 0), 2)
    cv2.circle(frame, (cx + x0, cy), 8, (0, 0, 255), -1)

    pos = (cx / (w - x0) - EDGE_MARGIN) / (1 - 2 * EDGE_MARGIN)
    return clamp(pos, 0.0, 1.0), mask


#awan
#warna awan per jenis (warna isi, warna tepi) format BGR
COLORS = {
    "normal":    ((255, 255, 255), (215, 215, 215)),
    "moving":    ((150, 210, 255), (110, 170, 230)),
    "breakable": ((170, 175, 185), (130, 135, 145)),
    "spring":    ((255, 255, 255), (215, 215, 215)),
    "gate":      ((190, 235, 255), (90, 190, 240)),
}


class Platform:

    def __init__(self, x, y, width, kind="normal", speed=0.0):
        self.x, self.y, self.width = x, y, width
        self.kind, self.speed = kind, speed

    def update(self):
        if self.kind == "moving":
            self.x += self.speed
            half = self.width / 2
            if self.x < half or self.x > GAME_W - half:
                self.x = clamp(self.x, half, GAME_W - half)
                self.speed = -self.speed

    def draw(self, canvas):
        x, y, w = int(self.x), int(self.y), int(self.width)
        fill, edge = COLORS[self.kind]
        for color, grow in ((edge, 2), (fill, 0)):
            cv2.ellipse(canvas, (x, y + 6), (w // 2 + grow, 8 + grow), 0, 0, 360, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x - w // 5, y + 3), 9 + grow, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x + w // 6, y + 2), 11 + grow, color, -1, cv2.LINE_AA)

        if self.kind == "breakable":                # garis retak
            crack = np.array([(x - 8, y - 4), (x - 2, y + 6), (x + 5, y), (x + 10, y + 8)], np.int32)
            cv2.polylines(canvas, [crack], False, (80, 90, 100), 2, cv2.LINE_AA)
        elif self.kind == "spring":                 # per merah
            coil = np.array([(x - 6, y), (x + 6, y - 3), (x - 6, y - 6), (x + 6, y - 9)], np.int32)
            cv2.polylines(canvas, [coil], False, (60, 60, 60), 2, cv2.LINE_AA)
            cv2.rectangle(canvas, (x - 9, y - 13), (x + 9, y - 9), (0, 0, 220), -1)


#game
class Game:
    def __init__(self):
        self.best = 0
        self.player_img = load("player.png", PLAYER_H)
        self.door_img = load("door.png", DOOR_H)
        self.backgrounds = [load_background(s[0]) for s in STAGES]
        self.pw = self.player_img.shape[1]
        self.dw = self.door_img.shape[1]
        self.reset()

    def reset(self):
        self.started = self.over = self.won = self.gate_spawned = False
        self.height = 0.0
        self.score = 0
        self.stage, self.prev_stage = 0, 0
        self.fade = self.banner = 0
        self.px, self.py, self.vy = GAME_W / 2, GROUND_Y, JUMP
        self.platforms = [Platform(GAME_W / 2, GROUND_Y, 150)]
        self.spawn()

    def spawn(self):
        #Buat awan baru di atas layar. Setelah awan pintu surga dibuat, tidak ada awan lagi.
        if self.gate_spawned:
            return
        top = min(p.y for p in self.platforms)
        while top > -60:
            level = min(self.score / GATE_SCORE, 1.0)
            top -= random.uniform(55 + 25 * level, 85 + 55 * level)

            if self.height + GROUND_Y - top >= GATE_WORLD:
                self.platforms.append(Platform(GAME_W / 2, top, 200, "gate"))
                self.gate_spawned = True
                return

            width = random.randint(int(90 - 30 * level), int(100 - 25 * level))
            x = random.uniform(width / 2, GAME_W - width / 2)
            kind, speed = "normal", 0.0
            r = random.random()
            if r < 0.07:
                kind = "spring"
            elif r < 0.07 + 0.2 * level:
                kind = "breakable"
            elif r < 0.07 + 0.5 * level:
                kind = "moving"
                speed = random.choice([-1, 1]) * (1.2 + 1.8 * level)
            self.platforms.append(Platform(x, top, width, kind, speed))

    def step(self, hand_x, found):
        if not self.started:
            if not found:
                return
            self.started, self.banner = True, BANNER_STEPS
        if self.over or self.won:
            return

        self.fade = max(0, self.fade - 1)
        self.banner = max(0, self.banner - 1)
        for p in self.platforms:
            p.update()

        #gerakkan pemain, X ikut tangan, Y kena gravitasi
        self.px = clamp(hand_x * GAME_W, self.pw / 2, GAME_W - self.pw / 2)
        old_py = self.py
        self.vy += GRAVITY
        self.py += self.vy

        #mendarat di awan (hanya saat jatuh, saat naik bisa tembus dari bawah)
        if self.vy > 0:
            for p in self.platforms:
                above_cloud = abs(self.px - p.x) <= p.width / 2 + self.pw * 0.25
                crossed_top = old_py <= p.y + 1 and self.py >= p.y
                if above_cloud and crossed_top:
                    self.py = p.y
                    self.vy = SPRING if p.kind == "spring" else JUMP
                    if p.kind == "breakable":
                        self.platforms.remove(p) #awan langsung hilang
                    break

        #menang kalau badan pemain menyentuh pintu
        for p in self.platforms:
            if (p.kind == "gate" and abs(self.px - p.x) < (self.dw + self.pw * 0.6) / 2
                    and self.py > p.y - DOOR_H and self.py - PLAYER_H < p.y):
                self.won = True
                return

        #kalau pemain lewat garis, semua awan digeser turun
        if self.py < SCROLL_LINE:
            shift = SCROLL_LINE - self.py
            self.py = SCROLL_LINE
            for p in self.platforms:
                p.y += shift
            self.height += shift
            self.score = int(self.height / 10)
            self.best = max(self.best, self.score)

        #ganti stage sesuai skor
        new_stage = 0
        for i, stage in enumerate(STAGES):
            if self.score >= stage[2]:
                new_stage = i
        if new_stage != self.stage:
            self.prev_stage, self.stage = self.stage, new_stage
            self.fade, self.banner = FADE_STEPS, BANNER_STEPS

        #buang awan di bawah layar, buat yang baru di atas
        self.platforms = [p for p in self.platforms if p.y < GAME_H + 40]
        self.spawn()

        if self.py - PLAYER_H > GAME_H: #jatuh keluar layar = kalah
            self.over = True

    def text(self, canvas, s, y, scale=0.8, color=(255, 255, 255), x=None):
        font = cv2.FONT_HERSHEY_SIMPLEX
        if x is None:
            x = (GAME_W - cv2.getTextSize(s, font, scale, 2)[0][0]) // 2
        cv2.putText(canvas, s, (x, y), font, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(canvas, s, (x, y), font, scale, color, 2, cv2.LINE_AA)

    def draw(self, found):
        #background, dicampur dengan background lama saat baru ganti stage (fade)
        canvas = self.backgrounds[self.stage].copy()
        if self.fade > 0:
            old = self.fade / FADE_STEPS
            canvas = cv2.addWeighted(self.backgrounds[self.prev_stage], old, canvas, 1 - old, 0)

        for p in self.platforms:
            if p.kind == "gate":
                overlay(canvas, self.door_img, int(p.x - self.dw / 2), int(p.y + 4 - DOOR_H))
            p.draw(canvas)
        overlay(canvas, self.player_img, int(self.px - self.pw / 2), int(self.py - PLAYER_H))

        self.text(canvas, "Score: %d" % self.score, 28, 0.7, x=10)
        self.text(canvas, "Best Score: %d" % self.best, 28, 0.6, x=GAME_W - 160)

        if not self.started:
            self.text(canvas, "Tunjukkan objek biru", GAME_H // 2 - 10)
            self.text(canvas, "untuk mulai", GAME_H // 2 + 25)
        elif not (self.over or self.won):
            if self.banner > 0:
                self.text(canvas, "STAGE %d: %s" % (self.stage + 1, STAGES[self.stage][1]),
                          90, 0.9, (60, 220, 255))
            if not found:
                self.text(canvas, "Tangan tidak terdeteksi", 60, 0.6, (80, 80, 255))

        if self.over or self.won:
            canvas = (canvas * 0.5).astype(np.uint8)
            if self.over:
                self.text(canvas, "GAME OVER", GAME_H // 2 - 30, 1.2, (80, 80, 255))
            else:
                self.text(canvas, "YOU WIN!", GAME_H // 2 - 30, 1.2, (60, 220, 255))
            self.text(canvas, "Score: %d" % self.score, GAME_H // 2 + 15, 0.9)
            self.text(canvas, "Press R to Play Again", GAME_H // 2 + 55, 0.7)
        return canvas


#main
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Kamera tidak bisa dibuka. Coba ubah CAM_INDEX.")
        return

    game = Game()
    kernel = np.ones((5, 5), np.uint8)
    hand_x = 0.5
    timer, last = 0.0, time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(cv2.resize(frame, (640, 480)), 1)

        raw_x, mask = detect_hand(frame, kernel)
        found = raw_x is not None
        if found:
            hand_x += SMOOTH * (raw_x - hand_x)

        now = time.perf_counter()
        timer += min(now - last, 0.1)
        last = now
        while timer >= 1 / 60:
            game.step(hand_x, found)
            timer -= 1 / 60

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