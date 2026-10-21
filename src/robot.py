"""Yanshee client: real YanAPI or simulation (--dry-run)."""
import os
import sys
import threading
import time
import config as C
import random

# YanAPI.py and lib_ukit/ live in the project root, not in src/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# open_vision_stream returns this code ("Resource is not availble") if the stream is already open
STREAM_ALREADY_OPEN = 20001


class DryRunRobot:
    """Simulates the servos. If given `stream` (a YanAPIRobot), it is used ONLY to open and close
    the real camera; the servos are never touched."""

    def __init__(self, servos, stream=None):
        self.angles = {s: 90 for s in servos}
        self.stream = stream

    def open_stream(self, resolution):
        if self.stream:
            self.stream.open_stream(resolution)
        else:
            print(f"[dry-run] open_vision_stream({resolution})")

    def close_stream(self):
        if self.stream:
            self.stream.close_stream()
        else:
            print("[dry-run] close_vision_stream()")

    def get_angles(self, names):
        return {n: self.angles[n] for n in names}

    def set_angles(self, angles, runtime_ms):
        print(f"[dry-run] set_servos_angles({angles}, runtime={runtime_ms})")
        self.angles.update(angles)
        return {"code": 0, "data": {n: True for n in angles}, "msg": "dry-run"}

    def home(self, angles, runtime_ms):
        print(f"[dry-run] home: set_servos_angles({angles}, runtime={runtime_ms})")
        self.angles.update(angles)

    def rest(self):
        print("[dry-run] start_play_motion('reset')")


class YanAPIRobot:
    def __init__(self, ip, music):
        import YanAPI  # UBTECH's Python module (ships with the robot)
        self.api = YanAPI
        self.api.yan_api_init(ip)
        self.api.set_robot_volume_value(C.ROBOT_VOL)

        if music:
            song_list = sorted(os.listdir(os.path.join(ROOT, "music")))  # music/ in the project root
            self.N_songs = len(song_list)
            self.i = 0
            self.songs = random.sample(song_list, self.N_songs)
            self.play_song()

    def play_song(self):
        if self.i < self.N_songs:
            self.api.start_play_music(self.songs[self.i])
            self.i += 1
        else:
            self.i = 0
            self.songs = random.sample(self.songs, self.N_songs)
            self.api.start_play_music(self.songs[self.i])
            self.i += 1
        
    def open_stream(self, resolution):
        res = self.api.open_vision_stream(resolution=resolution)
        if res.get("code") == STREAM_ALREADY_OPEN:
            # Left open by another program (or a previous run that did not close it): reuse it,
            # but with the resolution it was opened with, not necessarily `resolution`.
            print("[robot] camera stream was already open; reusing it")
        elif res.get("code") != 0:
            raise RuntimeError(f"open_vision_stream failed: {res}")

    def close_stream(self):
        self.api.close_vision_stream()

    def get_angles(self, names):
        res = self.api.get_servos_angles(names)
        return res.get("data", res)

    def set_angles(self, angles, runtime_ms):
        return self.api.set_servos_angles(angles, runtime_ms)
    
    def home(self, angles, runtime_ms):
        """angles: {servo: home angle}. Slow move to avoid a jerk."""
        return self.api.set_servos_angles(angles, runtime_ms)

    def rest(self):
        self.api.start_play_motion(name="reset")


class CommandSender(threading.Thread):
    """Thread that sends only the latest pending command to the robot, at a fixed rate.

    This way detection never waits for the network and no queue of stale commands builds up.
    """

    def __init__(self, robot, hz, on_sent=None):
        """on_sent(t, angles, runtime_ms, latency_s, response): called after every send (log)."""
        super().__init__(daemon=True)
        self.robot, self.period = robot, 1.0 / hz
        self.on_sent = on_sent
        self._lock = threading.Lock()
        self._pending = None  # (angles, runtime_ms)
        self._stop = threading.Event()
        self._last_error = None

    def submit(self, angles, runtime_ms):
        if not angles:
            return
        with self._lock:
            # Merge, do not replace: each command only carries the servos that changed on that frame.
            # If replaced, a servo that changed two frames ago and was not sent yet would be lost (in
            # sesion2.csv, LeftShoulderFlex got 4 commands during 15 s of fast movement).
            if self._pending:
                merged = {**self._pending[0], **angles}
                runtime_ms = max(runtime_ms, self._pending[1])
            else:
                merged = dict(angles)
            self._pending = (merged, runtime_ms)

    def clear(self):
        with self._lock:
            self._pending = None

    def run(self):
        while not self._stop.is_set():
            t0 = time.monotonic()
            with self._lock:
                job, self._pending = self._pending, None
            if job:
                try:
                    res = self.robot.set_angles(*job)
                except Exception as e:  # the network may fail: don't kill the loop
                    res = {"code": "exception", "msg": str(e)}
                self._report(job, res, t0)
            self._stop.wait(max(0.0, self.period - (time.monotonic() - t0)))

    def _report(self, job, res, t0):
        # set_servos_angles replies {"code": 0, "data": {servo: true/false}}; if the robot rejects
        # the command or some servo, warn (once per kind of error, so as not to flood the console).
        data = res.get("data") if isinstance(res, dict) else None
        failed = [n for n, ok in data.items() if ok is False] if isinstance(data, dict) else []  # sometimes data is the string "{}"
        error = None
        if not isinstance(res, dict) or res.get("code") != 0:
            error = f"command rejected: {res}"
        elif failed:
            error = f"servos that did not accept the command: {failed}"
        if error and error != self._last_error:
            print(f"[robot] {error}  (command: {job[0]})")
        self._last_error = error
        if self.on_sent:
            self.on_sent(t0, job[0], job[1], time.monotonic() - t0, res)

    def stop(self):
        self._stop.set()
