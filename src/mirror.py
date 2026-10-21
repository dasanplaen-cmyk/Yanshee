"""Mirror teleoperation: camera -> MediaPipe (Jetson or PC) -> Yanshee arm servos.

Usage:
  python src/mirror.py --ip 192.168.1.62 --source 0 --vis   # real robot, local camera, angle panel
  python src/mirror.py --ip 192.168.1.62 --dry-run  # uses the robot camera but does not move servos
  python src/mirror.py --dry-run --source 0          # local webcam, no robot

Gestures (held for GESTURE_HOLD_S): thumb up = arm, thumb down = disarm, middle finger = quit,
victory = next song (with --music).
Keys (window or terminal): x arms/disarms (fallback for the gestures), m adds a mark to the log,
n next song, q quits (robot back to rest).
Starts DISARMED with the robot at home (arms in a T-pose). Arming goes to SYNC: also stand in a
T-pose with straight elbows and hold it for HOME_HOLD_S; then it arms and the robot follows you.
"""
import argparse
import os
import queue
import select
import sys
import termios
import threading
import time
import tty

from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

import cv2
import mediapipe as mp
import numpy as np

import config as C
from filters import JointSmoother
from retarget import (L_ELBOW, L_SHOULDER, L_WRIST, R_ELBOW, R_SHOULDER, R_WRIST,
                      HomeSync, arm_angles, default_reference, driving_sides, home_error, human_to_robot,
                      DepthSign, rebuild_arms, segment_lengths)
from robot import CommandSender, DryRunRobot, YanAPIRobot
from session_log import SessionLog
from dashboard import Dashboard, overlay

# Paths relative to the project root, not to the folder the program is launched from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANDMARKER_MODEL = os.path.join(ROOT, "config", "pose_landmarker_full.task")
GESTURE_MODEL = os.path.join(ROOT, "config", "gesture_recognizer.task")
QUIT_KEY, ARM_KEY, MARK_KEY, NEXT_SONG_KEY = "q", "x", "m", "n"  # x: arm/disarm if gestures fail; m: mark in the log; n: next song
ARM_GESTURE, DISARM_GESTURE = "Thumb_Up", "Thumb_Down"
QUIT_GESTURE = "Fuck_Up"  # quits, like the q key. It is not in MediaPipe's model:
NEXT_SONG_GESTURE = "Victory"  # next song (--music), like the n key.
HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
KEY_JOINTS = [L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST]

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

class LatestFrame(threading.Thread):
    """Reads the video source and keeps only the latest frame (avoids buffer lag)."""

    def __init__(self, source, size=C.LOCAL_CAMERA_SIZE):
        """size: requested resolution if `source` is a local camera (index)."""
        super().__init__(daemon=True)
        self.source = source
        self.size = size
        self.cap = None
        self.frame, self.stamp = None, 0.0
        self.lock = threading.Lock()
        self.running = True

    def _open(self):
        if isinstance(self.source, int):
            # Local camera: request format, resolution and fps (by default OpenCV opens YUYV 640x480)
            cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*C.LOCAL_CAMERA_FOURCC))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            cap.set(cv2.CAP_PROP_FPS, C.LOCAL_CAMERA_FPS)
            # 2 buffers, not 1: with a single one V4L2 waits for it to be returned before capturing
            # the next frame and the camera drops from 30 to 16 fps. This thread keeps the latest frame anyway.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        else:
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            return None
        if isinstance(self.source, int):
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="ignore")
            print(f"[cámara {self.source}] {fourcc} {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} a {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
        return cap

    def run(self):
        # The robot's video server takes a moment to start and the Wi-Fi may drop: if there is no
        # connection or a read fails, reopen instead of retrying read() forever.
        while self.running:
            if self.cap is None:
                self.cap = self._open()
                if self.cap is None:
                    time.sleep(0.5)
                    continue
            ok, frame = self.cap.read()
            if not ok:
                self.cap.release()
                self.cap = None
                time.sleep(0.5)
                continue
            with self.lock:
                self.frame, self.stamp = frame, time.monotonic()
        if self.cap is not None:
            self.cap.release()

    def get(self):
        with self.lock:
            return self.frame, self.stamp


class ConsoleKeys(threading.Thread):
    """Reads single key presses from the terminal, without Enter, in addition to the OpenCV window.

    cv2.waitKey only receives keys while the video window has focus. cbreak mode (not raw) is used
    so that Ctrl+C keeps working.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self.keys = queue.SimpleQueue()
        self.running = True
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self.old_attrs = None

    def start(self):
        if self.fd is None:
            return  # no terminal (e.g. redirected output): window only
        self.old_attrs = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        super().start()

    def run(self):
        while self.running:
            ready, _, _ = select.select([self.fd], [], [], 0.1)
            if ready:
                self.keys.put(os.read(self.fd, 1).decode(errors="ignore"))

    def get(self):
        try:
            return self.keys.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.old_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_attrs)  # restore the terminal


def read_key(console):
    """Key pressed in the OpenCV window or in the terminal ('' if none)."""
    key = cv2.waitKey(1) & 0xFF
    if key != 0xFF:
        return chr(key).lower()
    ch = console.get()
    return ch.lower() if ch else ""


def make_landmarker(model_path):
    opts = vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(opts)

def make_gesture_model(model_path):
    # VIDEO mode: uses tracking between frames (faster and steadier than IMAGE)
    options = vision.GestureRecognizerOptions(
        base_options=mpt.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return vision.GestureRecognizer.create_from_options(options)


def is_middle_finger(landmarks) -> bool:
    """Middle finger extended, index, ring and pinky folded (the thumb does not matter).

    A finger is extended if its tip is clearly further from the wrist than its middle knuckle (PIP),
    and folded otherwise. It is a ratio of distances: independent of hand size and rotation.
    """
    wrist = np.array([landmarks[0].x, landmarks[0].y])
    def reach(tip, pip):  # >1: finger extended; <1: folded
        t = np.linalg.norm(np.array([landmarks[tip].x, landmarks[tip].y]) - wrist)
        p = np.linalg.norm(np.array([landmarks[pip].x, landmarks[pip].y]) - wrist)
        return t / (p + 1e-9)
    middle = reach(12, 10)
    others = (reach(8, 6), reach(16, 14), reach(20, 18))  # index, ring, pinky
    return middle > 1.3 and all(r < 1.0 for r in others)


def detect_gestures(result) -> dict:
    """{'Left': gesture, 'Right': gesture}, with 'None' for a hand that is not seen or has no gesture.

    Note: MediaPipe labels the hand assuming a selfie (mirrored) image; with a non-mirrored camera
    'Left' is actually the person's right hand. It does not matter for arming/disarming.
    """
    gestures = {"Left": "None", "Right": "None"}
    for handedness, gesture in zip(result.handedness, result.gestures):
        if handedness and gesture and handedness[0].category_name in gestures:
            gestures[handedness[0].category_name] = gesture[0].category_name
    for handedness, landmarks in zip(result.handedness, getattr(result, "hand_landmarks", [])):
        if handedness and handedness[0].category_name in gestures and is_middle_finger(landmarks):
            gestures[handedness[0].category_name] = QUIT_GESTURE
    return gestures


class GestureTrigger:
    """Turns a held gesture into ONE event: fires once the gesture has been held for `hold_s` and does
    not fire again until it is released (otherwise disarming would send home on every frame)."""

    def __init__(self, hold_s: float):
        self.hold_s = hold_s
        self.gesture, self.since, self.fired = None, 0.0, False

    def update(self, gesture, t: float):
        """gesture: gesture seen on this frame (or None). Returns the gesture if it fires now."""
        if gesture != self.gesture:
            self.gesture, self.since, self.fired = gesture, t, False
            return None
        if gesture is None or self.fired or t - self.since < self.hold_s:
            return None
        self.fired = True
        return gesture

    def progress(self, t: float) -> float:
        if self.gesture is None or self.fired:
            return 0.0
        return min(1.0, (t - self.since) / self.hold_s)


def draw_hands(frame, result, gestures):
    """Draws on `frame` (BGR, in place) the 21 points of each hand, their connections and the gesture."""
    h, w = frame.shape[:2]
    for handedness, landmarks in zip(result.handedness, result.hand_landmarks):
        pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 1, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        label = handedness[0].category_name
        x0 = min(x for x, _ in pts)
        y0 = max(15, min(y for _, y in pts) - MARGIN)
        cv2.putText(frame, f"{label}: {gestures.get(label, 'None')}", (x0, y0), cv2.FONT_HERSHEY_DUPLEX,
                    0.5, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default=C.ROBOT_IP)
    ap.add_argument("--dry-run", action="store_true", help="do not move the servos")
    ap.add_argument("--source", help="alternative video source (camera index such as 0, or a URL)")
    ap.add_argument("--res", help="local camera resolution, e.g. 1920x1080 (default %dx%d)"
                    % C.LOCAL_CAMERA_SIZE)
    ap.add_argument("--no-mirror", action="store_true", help="direct imitation instead of mirror")
    ap.add_argument("--vis", action="store_true", help="video and arm angles in the same window")
    ap.add_argument("--log", metavar="SESSION.csv", help="record landmarks, angles and commands for later analysis")
    ap.add_argument("--music", action="store_true", help="play music on the robot while it moves")
    args = ap.parse_args()

    if args.dry_run:
        # With the robot camera, dry-run must still really turn it on
        robot = DryRunRobot(C.ARM_SERVOS, stream=YanAPIRobot(args.ip, args.music) if args.source is None else None)
    else:
        robot = YanAPIRobot(args.ip, args.music)
    home_angles = {n: C.CALIB[n]["home"] for n in C.ARM_SERVOS}
    robot.home(home_angles, C.HOME_RUNTIME_MS)
    if args.source is None:
        robot.open_stream(C.STREAM_RESOLUTION)
        source = C.STREAM_URL.format(ip=args.ip)
    else:
        source = int(args.source) if args.source.isdigit() else args.source

    t_start = time.monotonic()
    log = SessionLog(args.log, C.ARM_SERVOS) if args.log else None
    on_sent = (lambda t, *rest: log.sent(t - t_start, *rest)) if log else None

    def add_mark(origin):
        if log:
            log.mark += 1
            print(f"[log] marca {log.mark} ({origin})")

    cam = LatestFrame(source, tuple(int(v) for v in args.res.lower().split("x")) if args.res else C.LOCAL_CAMERA_SIZE)
    cam.start()
    sender = CommandSender(robot, C.SEND_HZ, on_sent=on_sent)
    sender.start()
    smoother = JointSmoother(C.DEADBAND_DEG, C.MAX_STEP_DEG, C.ONE_EURO_MIN_CUTOFF, C.ONE_EURO_BETA)
    landmarker = make_landmarker(LANDMARKER_MODEL)
    gesture_model = make_gesture_model(GESTURE_MODEL)
    mirror = C.MIRROR and not args.no_mirror
    # Only the person's arms that drive an active servo must hold the home pose
    active_sides = driving_sides(C.ARM_SERVOS, mirror)
    print("Brazo(s) de la persona que controlan el robot: " +
          ", ".join({"left": "IZQUIERDO", "right": "DERECHO"}[s] for s in sorted(active_sides)))

    # States: DESARMADO (disarmed) --arm--> SYNC: robot at home, waiting for the person to hold the
    # same pose --> ARMADO (armed): the robot copies how far the person moves away from their home.
    # Disarm or person lost: DESARMADO and robot to home. q: quit and robot to rest.
    sync = HomeSync(C.HOME_HOLD_S, C.HOME_TOL_DEG)
    reference = default_reference()  # until synced, a theoretical perfect T-pose
    prev_human = {}  # previous frame's angles, for arm_angles' branch hysteresis
    arm_lengths = {}  # full 2D upper arm/forearm length, measured when syncing (rebuild_arms)
    depth = DepthSign()  # robust upper arm/forearm depth sign (median + hysteresis)
    ui = Dashboard(C.ARM_SERVOS, C.CALIB, height=C.UI_HEIGHT) if args.vis else None
    cv2.namedWindow("Yanshee mirror", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # resizable
    state, first_move = "DESARMADO", False
    last_seen = time.monotonic()
    last_stamp = 0.0
    trigger = GestureTrigger(C.GESTURE_HOLD_S)
    frame_count, gesture_res = 0, None

    def disarm():
        nonlocal state
        state = "DESARMADO"
        sender.clear()
        smoother.reset()
        robot.home(home_angles, C.HOME_RUNTIME_MS)

    console = ConsoleKeys()
    console.start()
    print(f"Gestos ({C.GESTURE_HOLD_S:.0f} s): pulgar arriba = armar (SYNC), pulgar abajo = desarmar, "
          f"dedo corazón = marca en el log (con --log).\n"
          f"Teclas (ventana o terminal): {ARM_KEY} = armar/desarmar, {MARK_KEY} = marca en el log, {QUIT_KEY} = salir")
    try:
        # Without a window, waitKey gets no keys: until the first frame arrives there is no way to
        # quit with q, so wait with a time limit and exit with a clear error.
        print(f"Esperando vídeo de {source} ...")
        deadline = time.monotonic() + C.CAMERA_TIMEOUT_S
        while cam.get()[0] is None:
            if time.monotonic() > deadline:
                print(f"[error] sin vídeo tras {C.CAMERA_TIMEOUT_S:.0f} s desde {source}. "
                      "¿Está el robot encendido y en la misma red?")
                return
            time.sleep(0.1)
        timer_music = time.time()
        pending_key = ""
        while True:
            frame, stamp = cam.get()
            if frame is None or stamp == last_stamp:
                # Waiting for a frame: the key is not lost, it stays pending for the next frame
                pending_key = pending_key or read_key(console)
                if pending_key == QUIT_KEY:
                    break
                continue
            last_stamp = stamp
            now = time.monotonic()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                              int((now - t_start) * 1000))
            
            if gesture_res is None or frame_count % C.GESTURE_EVERY_N_FRAMES == 0:
                gesture_res = gesture_model.recognize_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int((now - t_start) * 1000))
            frame_count += 1
            gestures = detect_gestures(gesture_res)
            seen = set(gestures.values())
            # If each hand makes a different gesture, disarming wins (the safe choice)
            # Priority: disarm (safety) > quit > arm > next song
            gesture = next((g for g in (DISARM_GESTURE, QUIT_GESTURE, ARM_GESTURE, NEXT_SONG_GESTURE) if g in seen), None)
            fired = trigger.update(gesture, now)

            if args.music and not args.dry_run:
                if int(now - timer_music) % 5 == 0:
                    status = robot.api.get_media_music_state()['data']['status'] == 'run'
                    if not status:
                        robot.play_song()
                    timer_music = time.time()
    
            status = state
            visible = False
            human = targets = smoothed = cmd = None
            if res.pose_world_landmarks:
                img_lms = res.pose_landmarks[0]
                visible = all((img_lms[i].visibility or 0) >= C.MIN_VISIBILITY for i in KEY_JOINTS)
                if visible:
                    last_seen = now
                    world = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]])
                    img = np.array([[p.x * frame.shape[1], p.y * frame.shape[0]] for p in img_lms])
                    if arm_lengths:
                        pts = rebuild_arms(world, img, arm_lengths, dz=depth.update(world))
                    elif state == "SYNC" and C.DEPTH_FROM_2D:
                        # In a T-pose the arms are in the image plane: measure them in 2D. MediaPipe's z
                        # gives ~30 deg of elbow with a straight arm (sesion1.csv) and syncing failed.
                        pts = rebuild_arms(world, img, segment_lengths(img))
                    else:
                        pts = world
                    human = {side: arm_angles(pts, side, prev_human.get(side)) for side in ("left", "right")}
                    prev_human = human
                    if state == "SYNC":
                        ref = sync.update({s: human[s] for s in active_sides}, now,
                                          frame=(world, img) if C.DEPTH_FROM_2D else None)
                        if ref is not None:
                            reference = {**reference, **ref}
                            arm_lengths.update({s: r["lengths"] for s, r in ref.items() if "lengths" in r})
                            prev_human = {}  # from now on the arms are measured rebuilt
                            state, first_move = "ARMADO", True
                            smoother.reset()
                            print("ARMADO. Referencia home de la persona: " + ", ".join(
                                f"{side} abd={r['abduction']:.0f} codo={r['elbow']:.0f}" for side, r in ref.items()))
                    targets = human_to_robot(human, C.CALIB, mirror, reference, C.ARM_SERVOS)
                    smoothed = smoother.smooth(targets, now)
                    if state == "ARMADO":
                        current = robot.get_angles(C.ARM_SERVOS) if first_move else {}
                        cmd = smoother.next_command(smoothed, current)
                        sender.submit(cmd, C.FIRST_MOVE_RUNTIME_MS if first_move else C.RUNTIME_MS)
                        first_move = False
                    for p in img_lms:
                        cv2.circle(frame, (int(p.x * frame.shape[1]), int(p.y * frame.shape[0])), 3, (0, 255, 0), -1)
                    if state == "SYNC":
                        # err = max(home_error(human[s]) for s in active_sides)
                        err = max(home_error(human[s]) for s in active_sides)
                        status = (f"SYNC: ponte en cruz con los codos rectos. "
                                  f"Desviacion {err:.0f}/{C.HOME_TOL_DEG:.0f}, {sync.progress(now) * 100:.0f}%")
                    
            if log:
                lms = res.pose_landmarks[0] if res.pose_landmarks else None
                wlms = res.pose_world_landmarks[0] if res.pose_world_landmarks else None
                log.frame(now - t_start, state, wlms, lms, human, targets, smoothed, cmd)
            if not visible:
                depth.reset()  # its history no longer belongs to the current pose
                prev_human = {}  # no reliable continuity: the next branch is chosen from home
                if state == "SYNC":
                    sync.reset()  # the home pose must be held continuously and fully visible

            # Watchdog: if the person is lost, disarm and send the robot home
            if state == "ARMADO" and now - last_seen > C.LOST_TIMEOUT_S:
                print("[seguridad] persona perdida: desarmado y robot a home")
                disarm()

            draw_hands(frame, gesture_res, gestures)
            if not visible:
                status = f"{state}: no se ve a la persona entera (hombros, codos y munecas)"
            notes = [f"Gestos: izq {gestures['Left']} / der {gestures['Right']}"]
            if trigger.progress(now) > 0:
                notes.append(f"Manteniendo {trigger.gesture}: {trigger.progress(now) * 100:.0f}%")
            if log:
                notes.append(f"REC  {os.path.basename(args.log)}  (marcas: {log.mark})")
            if ui:  # --vis: video + angle panel
                ui.add(now - t_start, smoothed)
                cv2.imshow("Yanshee mirror", ui.render(frame, state, status, smoothed,
                                                       smoother.sent if state == "ARMADO" else {}, notes))
            else:   # video only, with the state on top
                cv2.imshow("Yanshee mirror", overlay(frame, state, status, notes[1:]))
            key = pending_key or read_key(console)  # each key is read only once and used here
            pending_key = ""
            if key == QUIT_KEY or fired == QUIT_GESTURE:
                break
            if args.music and not args.dry_run and (key == NEXT_SONG_KEY or fired == NEXT_SONG_GESTURE):
                robot.play_song()
                timer_music = time.time()
            if key == MARK_KEY:
                add_mark("tecla")
            # if fired == QUIT_GESTURE:
            #     add_mark("gesto")
            arm_request = fired == ARM_GESTURE or (key == ARM_KEY and state == "DESARMADO")
            disarm_request = fired == DISARM_GESTURE or (key == ARM_KEY and state != "DESARMADO")
            if arm_request and state == "DESARMADO":
                state = "SYNC"
                sync.reset()
                arm_lengths.clear()  # measured again in the new T-pose
                robot.home(home_angles, C.HOME_RUNTIME_MS)
                print("SYNC: robot en home; ponte en cruz con los codos rectos")
            elif disarm_request and state != "DESARMADO":
                disarm()
                print("DESARMADO")

    except KeyboardInterrupt:
        pass  # Ctrl+C: clean exit, the finally block sends the robot to rest
    finally:
        # Each step separately: if the Wi-Fi has dropped, calls to the robot fail and must not
        # prevent restoring the terminal.
        console.stop()
        sender.stop()
        cam.running = False
        cam.join(timeout=2)  # let it leave cap.read() before exiting (otherwise: 'FATAL: exception not rethrown')
        if not args.dry_run:
            robot.api.stop_play_music()

        if log:
            log.close()
            print(f"[log] sesión guardada en {args.log}")

        for step in ([robot.rest] + ([robot.close_stream] if args.source is None else [])):
            try:
                step()
            except Exception as e:
                print(f"[robot] {step.__name__} falló al salir: {e}")
        # Close them here: if they close on their own when Python exits, MediaPipe prints __del__ errors
        landmarker.close()
        gesture_model.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
