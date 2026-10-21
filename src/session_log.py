"""Session recording (--log) to analyse detection and filtering afterwards.

Writes two CSV files:
  <name>.csv       one row per processed frame: raw landmarks, angles, targets, commands
  <name>_sent.csv  one row per command sent to the robot: response and latency
With the raw landmarks any variant of the retargeting can be recomputed without repeating the test.
"""
import csv
import os
import threading

from retarget import L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, R_ELBOW, R_HIP, R_SHOULDER, R_WRIST, HAND_POINTS

LANDMARKS = {
    "l_sh": L_SHOULDER, "r_sh": R_SHOULDER, "l_el": L_ELBOW, "r_el": R_ELBOW,
    "l_wr": L_WRIST, "r_wr": R_WRIST, "l_hip": L_HIP, "r_hip": R_HIP,
    # hand (pinky, index, thumb): for the robust wrist z (DepthSign)
    **{f"{s[0]}_{n}": i for s, pts in HAND_POINTS.items() for n, i in zip(("pinky", "index", "thumb"), pts)},
}
HUMAN = ("abduction", "pitch", "elbow")


class SessionLog:
    def __init__(self, path, servos):
        root, ext = os.path.splitext(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)  # e.g. logs/ in a fresh clone
        self.servos = list(servos)
        self.mark = 0
        self._lock = threading.Lock()  # the sender thread also writes
        self._closed = False

        self._f = open(root + (ext or ".csv"), "w", newline="")
        self._w = csv.writer(self._f)
        header = ["t", "state", "mark", "pose"]  # pose: MediaPipe detected someone (even if not fully visible)
        for name in LANDMARKS:
            header += [f"{name}_{k}" for k in ("wx", "wy", "wz", "ix", "iy", "vis")]
        for side in ("left", "right"):
            header += [f"{side}_{q}" for q in HUMAN]
        for kind in ("target", "smooth", "cmd"):
            header += [f"{kind}_{s}" for s in self.servos]
        self._w.writerow(header)

        self._fs = open(root + "_sent.csv", "w", newline="")
        self._ws = csv.writer(self._fs)
        self._ws.writerow(["t", "runtime_ms", "latency_ms", "code", "msg"] + self.servos)

    def frame(self, t, state, world=None, image=None, human=None, targets=None, smoothed=None, cmd=None):
        """world/image: MediaPipe landmark lists (or None if there is no person)."""
        row = [f"{t:.4f}", state, self.mark, int(world is not None)]
        for idx in LANDMARKS.values():
            if world is None:
                row += [""] * 6
            else:
                w, i = world[idx], image[idx]
                row += [f"{w.x:.4f}", f"{w.y:.4f}", f"{w.z:.4f}", f"{i.x:.4f}", f"{i.y:.4f}", f"{i.visibility or 0:.3f}"]
        for side in ("left", "right"):
            arm = human.get(side) if human else None
            row += [f"{getattr(arm, q):.2f}" if arm else "" for q in HUMAN]
        for d in (targets, smoothed, cmd):
            row += [f"{d[s]:.2f}" if d and s in d else "" for s in self.servos]
        with self._lock:
            if not self._closed:
                self._w.writerow(row)

    def sent(self, t, angles, runtime_ms, latency_s, res):
        code = res.get("code", "") if isinstance(res, dict) else ""
        msg = res.get("msg", "") if isinstance(res, dict) else str(res)
        with self._lock:
            if self._closed:
                return  # the sender thread may finish its last command after closing
            self._ws.writerow([f"{t:.4f}", runtime_ms, f"{latency_s * 1000:.1f}", code, msg]
                              + [angles.get(s, "") for s in self.servos])

    def close(self):
        with self._lock:
            self._closed = True
            self._f.close()
            self._fs.close()
