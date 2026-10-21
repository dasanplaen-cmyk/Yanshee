"""Single window: camera video + panel with the state and the arm angles.

All OpenCV (no matplotlib): drawing the panel costs a few ms per frame and does not open a second
window with another GUI toolkit.
"""
import unicodedata

import cv2
import numpy as np

# One colour per servo (BGR), the same for its bar and its line in the plot
COLORS = [(245, 135, 66), (0, 165, 255), (80, 200, 80), (60, 60, 230), (220, 120, 220), (200, 200, 60)]
STATE_COLORS = {"ARMED": (40, 40, 200), "SYNC": (0, 190, 220), "DISARMED": (90, 90, 90)}
BG, GRID, TEXT, DIM = (28, 28, 28), (60, 60, 60), (235, 235, 235), (150, 150, 150)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _ascii(text: str) -> str:
    """OpenCV fonts are ASCII only: 'déjà vu' -> 'deja vu'."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _wrap(text: str, width: int):
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])


class Dashboard:
    def __init__(self, servos, calib, height=720, panel_width=480, history_s=10.0):
        self.servos, self.calib = list(servos), calib
        self.h, self.pw, self.history_s = height, panel_width, history_s
        self.t = []
        self.values = {s: [] for s in self.servos}
        self.fps, self._last_t = 0.0, None

    def add(self, t: float, smoothed):
        """One processed frame. smoothed: {servo: angle}, or None if the person is not visible (breaks the line)."""
        self.t.append(t)
        for s in self.servos:
            self.values[s].append((smoothed or {}).get(s, np.nan))
        while self.t and self.t[0] < t - self.history_s:
            self.t.pop(0)
            for v in self.values.values():
                v.pop(0)
        if self._last_t is not None and t > self._last_t:
            inst = 1.0 / (t - self._last_t)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self._last_t = t

    def render(self, frame, state: str, status: str, smoothed, sent, notes=()) -> np.ndarray:
        """Full window image: video scaled to the panel height + panel on the right.

        smoothed: target of each servo (or None); sent: last angle sent to the robot, per servo.
        notes: extra lines (gestures, recording...).
        """
        scale = self.h / frame.shape[0]
        video = cv2.resize(frame, (int(round(frame.shape[1] * scale)), self.h),
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        panel = np.full((self.h, self.pw, 3), BG, np.uint8)
        y = self._header(panel, state)
        lines = _wrap(_ascii(status), 58) if status and status != state else []
        for line in lines + [_ascii(n) for n in notes]:
            cv2.putText(panel, line, (12, y), FONT, 0.45, TEXT, 1, cv2.LINE_AA)
            y += 18
        y += 8
        for i, s in enumerate(self.servos):
            self._servo_row(panel, y, s, COLORS[i % len(COLORS)], (smoothed or {}).get(s), sent.get(s))
            y += 44
        self._plot(panel, y + 6, self.h - 12)
        return np.hstack([video, panel])

    def _header(self, panel, state):
        cv2.rectangle(panel, (0, 0), (self.pw, 38), STATE_COLORS.get(state, (90, 90, 90)), -1)
        cv2.putText(panel, state, (12, 27), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panel, f"{self.fps:4.1f} fps", (self.pw - 105, 26), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return 60

    def _servo_row(self, panel, y, name, color, target, sent):
        c = self.calib[name]
        x0, x1 = 12, self.pw - 12
        px = lambda a: int(x0 + (x1 - x0) * float(np.clip(a, 0, 180)) / 180.0)
        values = f"target {target:5.1f}" if target is not None else "target  --"
        values += f"   sent {sent:4.0f}" if sent is not None else "   sent  --"
        cv2.putText(panel, name, (x0, y), FONT, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(panel, values, (x1 - 195, y), FONT, 0.45, TEXT, 1, cv2.LINE_AA)
        top, bottom = y + 8, y + 22
        cv2.rectangle(panel, (x0, top), (x1, bottom), GRID, -1)                       # 0..180
        cv2.rectangle(panel, (px(c["min"]), top), (px(c["max"]), bottom), (85, 85, 85), -1)  # allowed range
        cv2.line(panel, (px(c["home"]), top - 3), (px(c["home"]), bottom + 3), DIM, 1)    # home
        if sent is not None:
            cv2.line(panel, (px(sent), top - 4), (px(sent), bottom + 4), (255, 255, 255), 2)
        if target is not None:
            cv2.circle(panel, (px(target), (top + bottom) // 2), 6, color, -1, cv2.LINE_AA)

    def _plot(self, panel, top, bottom):
        x0, x1 = 34, self.pw - 12
        if bottom - top < 60:
            return
        cv2.putText(panel, f"Last {self.history_s:.0f} s (targets)", (12, top), FONT, 0.45, DIM, 1, cv2.LINE_AA)
        top += 10
        cv2.rectangle(panel, (x0, top), (x1, bottom), GRID, 1)
        py = lambda a: int(bottom - (bottom - top) * a / 180.0)
        for a in (0, 45, 90, 135, 180):
            cv2.line(panel, (x0, py(a)), (x1, py(a)), GRID, 1)
            cv2.putText(panel, str(a), (4, py(a) + 4), FONT, 0.35, DIM, 1, cv2.LINE_AA)
        if len(self.t) < 2:
            return
        t_end = self.t[-1]
        xs = [int(x1 - (x1 - x0) * (t_end - t) / self.history_s) for t in self.t]
        for i, s in enumerate(self.servos):
            segment = []
            for x, v in zip(xs, self.values[s]):
                if np.isnan(v):  # no person: break the line
                    if len(segment) > 1:
                        cv2.polylines(panel, [np.array(segment, np.int32)], False, COLORS[i % len(COLORS)], 1, cv2.LINE_AA)
                    segment = []
                else:
                    segment.append((x, py(v)))
            if len(segment) > 1:
                cv2.polylines(panel, [np.array(segment, np.int32)], False, COLORS[i % len(COLORS)], 1, cv2.LINE_AA)


def overlay(frame, state: str, status: str, notes=()) -> np.ndarray:
    """Video only (no --vis): state and notices in a semi-transparent band at the top left."""
    lines = [status if status and status != state else state] + list(notes)
    lines = [_ascii(l)[:90] for l in lines]
    h = 8 + 18 * len(lines)
    w = min(frame.shape[1], 12 + 8 * max(len(l) for l in lines))
    band = frame[:h, :w]
    frame[:h, :w] = (band * 0.4).astype(np.uint8)  # darken so the text is readable
    color = {"ARMED": (60, 60, 255), "SYNC": (0, 220, 255)}.get(state, (255, 255, 255))
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (6, 20 + 18 * i), FONT, 0.5, color if i == 0 else TEXT, 1, cv2.LINE_AA)
    return frame
