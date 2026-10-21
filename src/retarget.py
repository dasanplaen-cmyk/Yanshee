"""Converts MediaPipe landmarks into human arm angles.

Only uses numpy, so it can be tested without a camera or a robot.

Human angle convention (degrees), per arm:
  The shoulder is decomposed as "pitch then abduction" (same as the Yanshee arm):
  pitch     : rotation about the shoulder axis. 0 = arm in the plane of the body,
              90 = straight forward.
  abduction : arm opening out to the side. 0 = down, 90 = T-pose, 180 = up over the side.
  elbow     : SIGNED elbow flexion. 0 = straight arm; positive = the forearm bends upwards (in a
              T-pose), negative = downwards. Measured in the plane in which the robot's elbow bends
              (the frontal plane at pitch 0, rotated with the pitch). If the elbow bends out of that
              plane (towards the camera) the previous sign is kept.

  Every arm direction has TWO solutions (pitch, abduction): (p, a) and (p+180, 180-a).
  The branch of the previous frame is kept while its pitch is reachable (BRANCH_KEEP_DEG), and only
  then does it switch to the other one: a branch switch moves ShoulderRoll and ShoulderFlex almost
  end to end. Without a previous branch, the one with the smallest pitch is chosen.
  Near the T-pose the pitch is noise (1/cos sensitivity) and is attenuated towards 0. It does not
  matter for the robot, because in a T-pose the pitch servo only rotates the arm about its own axis.
  The abduction is computed afterwards, as the one that best points the arm given that pitch: it is
  continuous and matches the exact one when the pitch is exact.

Arm depth (rebuild_arms): MediaPipe's z barely sees movements towards the camera (in sesion1.csv,
bending the elbow forwards gave 7..29 deg with the forearm shortened to 10% in the image). After
syncing in a T-pose the full 2D length of upper arm and forearm is known, and the depth of each
segment comes from how much it is shortened in the image: sqrt(1 - r^2). Only the sign (towards or
away from the camera) is taken from MediaPipe.

Mapping to the robot (see CALIB in config.py), relative to the home pose (T-pose, straight elbows):
  servo = home + sign * (human - human_at_home)
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

# MediaPipe Pose indices (33 landmarks)
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
# Hand points in the pose model (pinky, index, thumb): they surround the wrist and give a more robust z
HAND_POINTS = {"left": (17, 19, 21), "right": (18, 20, 22)}

# How much the pitch is used depending on how far the arm leaves the sagittal plane (90 deg in a
# T-pose): fully up to 45 deg, not at all from 75 deg.
PITCH_FULL_DEG = 45.0
PITCH_ZERO_DEG = 75.0
# Stay on the current branch while its pitch does not exceed this; only then switch to the other one.
# Switching branch moves ShoulderRoll and ShoulderFlex almost end to end (e.g. Roll 170 -> 10 when
# lowering straight arms in front from overhead), so it is only done once the current one is no longer
# reachable: ShoulderRoll range in CALIB (10..170 = +-80 deg of pitch) plus a 30 deg margin.
BRANCH_KEEP_DEG = 110.0
# Upper arm and forearm aligned in the image (2D angle between them): below FULL they are treated as
# a straight arm (same depth); above ZERO, each one keeps its own (see rebuild_arms)
STRAIGHT_FULL_DEG = 10.0
STRAIGHT_ZERO_DEG = 25.0
# Depth sign (towards or away from the camera, see DepthSign): median of the last DEPTH_WINDOW
# frames; it only changes if |dz| exceeds DEPTH_SIGN_HYST_M (otherwise the previous one is kept)
DEPTH_WINDOW = 5
DEPTH_SIGN_HYST_M = 0.03
# Below this (cosine between the person's bending plane and the robot's) the elbow sign is not
# reliable and the previous frame's sign is kept
ELBOW_SIGN_MIN = 0.3

# Human angles in the home pose: arms in a T-pose and straight elbows. The pitch cannot be measured
# in a T-pose; 0 means "arm in the plane of the body", which is where the robot is at home.
HOME_HUMAN = {"abduction": 90.0, "pitch": 0.0, "elbow": 0.0}


@dataclass
class ArmAngles:
    abduction: float
    pitch: float
    elbow: float
    branch_pitch: Optional[float] = None  # exact pitch on the chosen branch (for the hysteresis)


HOME_ARM = ArmAngles(abduction=HOME_HUMAN["abduction"], pitch=HOME_HUMAN["pitch"], elbow=HOME_HUMAN["elbow"])


def _wrap(a: float) -> float:
    """Angle in (-180, 180]."""
    return -((-a + 180.0) % 360.0 - 180.0)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("zero vector")
    return v / n


def body_frame(pts: np.ndarray):
    """Return the person's torso unit vectors (left, up, forward).

    pts: (33, 3) array in a right-handed frame (MediaPipe world landmarks are:
    x to the right of the image, y down, z away from the camera).
    """
    mid_sh = (pts[L_SHOULDER] + pts[R_SHOULDER]) / 2
    mid_hip = (pts[L_HIP] + pts[R_HIP]) / 2
    up = _unit(mid_sh - mid_hip)
    left = pts[L_SHOULDER] - pts[R_SHOULDER]
    left = _unit(left - np.dot(left, up) * up)  # orthogonalise with respect to up
    forward = np.cross(left, up)  # out of the person's chest
    return left, up, forward


def arm_angles(pts: np.ndarray, side: str, prev: Optional[ArmAngles] = None) -> ArmAngles:
    """side: 'left' or 'right' (the person's anatomical side).

    prev: angles of the same arm on the previous frame, for the branch hysteresis.
    """
    left, up, forward = body_frame(pts)
    if side == "left":
        sh, el, wr, outward = pts[L_SHOULDER], pts[L_ELBOW], pts[L_WRIST], left
    else:
        sh, el, wr, outward = pts[R_SHOULDER], pts[R_ELBOW], pts[R_WRIST], -left

    u = _unit(el - sh)  # upper arm
    f = _unit(wr - el)  # forearm
    u_out, u_up, u_fwd = np.dot(u, outward), np.dot(u, up), np.dot(u, forward)

    # Exact pitch of the two branches: the previous frame's one while it is still reachable; if there
    # is none, or it no longer is, the one with the smallest pitch
    raw = float(np.degrees(np.arctan2(u_fwd, -u_up))) if np.hypot(u_fwd, u_up) > 1e-6 else 0.0
    branches = (raw, _wrap(raw + 180.0))
    exact = min(branches, key=abs)
    if prev is not None and prev.branch_pitch is not None:
        same = min(branches, key=lambda p: abs(_wrap(p - prev.branch_pitch)))
        if abs(same) <= BRANCH_KEEP_DEG:
            exact = same
    # Near the T-pose the exact pitch is not reliable: attenuate it towards 0
    sagittal_out = float(np.degrees(np.arcsin(np.clip(abs(u_out), 0.0, 1.0))))
    w = float(np.clip((PITCH_ZERO_DEG - sagittal_out) / (PITCH_ZERO_DEG - PITCH_FULL_DEG), 0.0, 1.0))
    pitch = w * exact
    # With that pitch, the abduction that best points the robot arm towards the person's arm.
    # Robot arm: out = sin(a), up = -cos(a)cos(p), fwd = cos(a)sin(p).
    t = np.radians(pitch)
    abduction = float(np.degrees(np.arctan2(u_out, -np.cos(t) * u_up + np.sin(t) * u_fwd)))
    if abduction < -90.0:
        # Arm past the vertical above the head: 181 deg, not -179 (otherwise, with the arm up it
        # jumped between -179 and 177 and the servo went end to end; sesion3.csv)
        abduction += 360.0

    # Elbow: magnitude of the upper arm-forearm angle, and sign depending on which side of the robot's
    # bending axis (forward rotated with the pitch) it bends to. For the right arm the axis is reversed.
    bend = float(np.degrees(np.arccos(np.clip(np.dot(u, f), -1.0, 1.0))))
    axis = np.cos(t) * forward + np.sin(t) * up
    normal = np.cross(u, f)
    side_sign = 1.0 if side == "left" else -1.0
    prev_sign = 1.0 if prev is None or prev.elbow >= 0 else -1.0
    n = np.linalg.norm(normal)
    in_plane = side_sign * np.dot(normal, axis) / n if n > 1e-6 else 0.0
    sign = prev_sign if abs(in_plane) < ELBOW_SIGN_MIN else float(np.sign(in_plane))
    elbow = sign * bend
    return ArmAngles(abduction=abduction, pitch=pitch, elbow=elbow, branch_pitch=exact)


ARM_CHAINS = {"left": (L_SHOULDER, L_ELBOW, L_WRIST), "right": (R_SHOULDER, R_ELBOW, R_WRIST)}


class DepthSign:
    """Robust dz of upper arm and forearm on each side, for the depth sign in rebuild_arms.

    The sign is the only thing that tells an arm pointing forwards from one pointing backwards (both
    look equally short in the image), and MediaPipe's z of a single point on a single frame is noisy:
      - wrist z: median of wrist, pinky, index and thumb (one badly detected point does not dominate)
      - dz of each segment: median of the last `window` frames
      - hysteresis: the sign only changes if |dz| > `hyst_m`; otherwise the previous one is kept
    """

    def __init__(self, window: int = DEPTH_WINDOW, hyst_m: float = DEPTH_SIGN_HYST_M):
        self.window, self.hyst_m = window, hyst_m
        self.reset()

    def reset(self):
        self.history = {}
        self.signs = {}

    def update(self, world: np.ndarray) -> dict:
        out = {}
        for side, (sh, el, wr) in ARM_CHAINS.items():
            z_wrist = float(np.median(world[[wr, *HAND_POINTS[side]], 2]))
            raw = (world[el][2] - world[sh][2], z_wrist - world[el][2])
            dz = []
            for k, value in enumerate(raw):
                hist = self.history.setdefault((side, k), [])
                hist.append(float(value))
                del hist[:-self.window]
                med = float(np.median(hist))
                prev = self.signs.get((side, k))
                sign = (1.0 if med >= 0 else -1.0) if prev is None or abs(med) > self.hyst_m else prev
                self.signs[(side, k)] = sign
                dz.append(sign * max(abs(med), 1e-6))
            out[side] = tuple(dz)
        return out


def segment_lengths(img: np.ndarray) -> dict:
    """2D length of upper arm and forearm on each side, in shoulder widths (also 2D).

    img: (33, 2) array of image landmarks in pixels.
    """
    shoulder_px = np.linalg.norm(img[L_SHOULDER] - img[R_SHOULDER])
    return {side: (float(np.linalg.norm(img[e] - img[sh]) / shoulder_px),
                   float(np.linalg.norm(img[w] - img[e]) / shoulder_px))
            for side, (sh, e, w) in ARM_CHAINS.items()}


def rebuild_arms(world: np.ndarray, img: np.ndarray, full_lengths: dict, dz: Optional[dict] = None) -> np.ndarray:
    """3D landmarks with the arms rebuilt: direction in the image plane, and depth from the
    foreshortening relative to `full_lengths` (measured in the T-pose, see HomeSync). The depth sign
    and the torso (shoulders, hips) are still MediaPipe's.

    Straight in 2D => straight in 3D: if upper arm and forearm are aligned in the image, they share
    the depth (the larger of the two) with the upper arm's sign. Without this, with the arm forwards or
    backwards the upper arm did not seem to shorten (MediaPipe moves the shoulder point, and with the arm
    down it looks ~1.2x longer than in the T-pose) while the forearm did: the elbow came out bent +-70
    deg with a straight arm (sesion2.csv).

    world: (33, 3) pose_world_landmarks (x right, y down, z away from the camera).
    img:   (33, 2) pose_landmarks in pixels.
    full_lengths: {'left': (upper_arm, forearm), ...} in shoulder widths; missing sides are untouched.
    dz: robust {'left': (dz_upper_arm, dz_forearm), ...} (see DepthSign). Without it, the frame's z is used.
    """
    pts = world.copy()
    shoulder_px = np.linalg.norm(img[L_SHOULDER] - img[R_SHOULDER])
    m_per_px = np.linalg.norm(world[L_SHOULDER] - world[R_SHOULDER]) / shoulder_px
    for side, lengths in full_lengths.items():
        sh, el, wr = ARM_CHAINS[side]
        segs = []  # (2D vector, full length in px, out-of-plane fraction, MediaPipe dz)
        for (a, b), full in zip(((sh, el), (el, wr)), lengths):
            d = img[b] - img[a]
            full_px = full * shoulder_px
            ratio = min(np.linalg.norm(d) / full_px, 1.0)
            segs.append([d, full_px, float(np.sqrt(1.0 - ratio ** 2)), world[b][2] - world[a][2]])
        if dz is not None and side in dz:
            segs[0][3], segs[1][3] = dz[side]
        (du, _, fu, zu), (df, _, ff, zf) = segs
        cos2d = np.dot(du, df) / (np.linalg.norm(du) * np.linalg.norm(df) + 1e-9)
        angle2d = float(np.degrees(np.arccos(np.clip(cos2d, -1.0, 1.0))))
        k = float(np.clip((STRAIGHT_ZERO_DEG - angle2d) / (STRAIGHT_ZERO_DEG - STRAIGHT_FULL_DEG), 0.0, 1.0))
        if k > 0:
            shared = max(fu, ff)
            segs[0][2] = fu + k * (shared - fu)
            segs[1][2] = ff + k * (shared - ff)
            # Same depth direction for both: the upper arm's. The hand's is biased (in sesion3.csv
            # MediaPipe puts it ~13 cm in front of the elbow with the arm hanging straight) and, when
            # summed, it turned an arm pointing backwards into one pointing forwards.
            segs[0][3] = segs[1][3] = zu or 1.0
        start_pt = pts[sh]
        for (d, full_px, f, seg_dz), b in zip(segs, (el, wr)):
            sign = 1.0 if seg_dz >= 0 else -1.0
            inplane = d / (np.linalg.norm(d) + 1e-9) * full_px * np.sqrt(max(0.0, 1.0 - f * f))
            start_pt = start_pt + np.array([inplane[0], inplane[1], sign * f * full_px]) * m_per_px
            pts[b] = start_pt
    return pts


def home_error(arm: ArmAngles) -> float:
    """Largest deviation (degrees) of an arm from the home pose. The pitch does not count:
    it is undefined in a T-pose (and arm_angles attenuates it to 0)."""
    return max(abs(arm.abduction - HOME_HUMAN["abduction"]), abs(arm.elbow - HOME_HUMAN["elbow"]))


def driving_sides(servos, mirror: bool) -> set:
    """The person's arms ('left'/'right') that drive any of these robot servos."""
    sides = set()
    for name in servos:
        robot_side = "right" if name.startswith("Right") else "left"
        sides.add(({"right": "left", "left": "right"}[robot_side]) if mirror else robot_side)
    return sides


def default_reference() -> dict:
    """Theoretical reference (perfect T-pose) used before syncing with the person."""
    return {side: dict(HOME_HUMAN) for side in ("left", "right")}


def human_to_robot(human: dict, calib: dict, mirror: bool, reference: dict, servos=None) -> dict:
    """human: {'left': ArmAngles, 'right': ArmAngles}. Returns unfiltered {servo: angle}.

    reference: the person's angles at home, {'left': {...}, 'right': {...}} (see HomeSync).
    servos: if given, only those servos are returned (the rest are never sent).
    mirror=True: the robot acts as a mirror (your right arm moves its left arm).
    """
    out = {}
    for person_side, arm in human.items():
        robot_side = ("Left" if person_side == "right" else "Right") if mirror else person_side.capitalize()
        for joint in ("ShoulderRoll", "ShoulderFlex", "ElbowFlex"):
            name = f"{robot_side}{joint}"
            if servos is not None and name not in servos:
                continue
            c = calib[name]
            value = getattr(arm, c["human"])
            angle = c["home"] + c["sign"] * (value - reference[person_side][c["human"]])
            out[name] = float(np.clip(angle, c["min"], c["max"]))
    return out


class HomeSync:
    """Waits for the person to hold the home pose and stores their angles as the reference.

    This way the robot (already at home) and the person start from the same pose: from then on only
    how far the person moves away from THEIR T-pose, which is never perfect, is copied.
    """

    def __init__(self, hold_s: float, tol_deg: float):
        self.hold_s, self.tol_deg = hold_s, tol_deg
        self.reset()

    def reset(self):
        self.t0 = None
        self.samples = []
        self.frames = []

    def progress(self, t: float) -> float:
        """0..1: fraction of the required time already held at home."""
        return 0.0 if self.t0 is None else min(1.0, (t - self.t0) / self.hold_s)

    def update(self, human: dict, t: float, frame=None) -> Optional[dict]:
        """Return the reference once home has been held for `hold_s` in a row; otherwise None.

        frame: (world, img) of this frame. If given, the reference also includes the full 2D length of
        upper arm and forearm ("lengths", for rebuild_arms), and the reference angles are recomputed
        with the rebuilt arms, which is how they will be measured afterwards.
        """
        if max(home_error(arm) for arm in human.values()) > self.tol_deg:
            self.reset()  # it must be held continuously
            return None
        if self.t0 is None:
            self.t0 = t
        self.samples.append(human)
        if frame is not None:
            self.frames.append(frame)
        if t - self.t0 < self.hold_s:
            return None

        samples, lengths = self.samples, None
        if self.frames:
            all_lengths = [segment_lengths(img) for _, img in self.frames]
            lengths = {side: tuple(float(np.median([l[side][k] for l in all_lengths])) for k in (0, 1))
                       for side in human}
            samples, prev = [], {}
            for world, img in self.frames:
                pts = rebuild_arms(world, img, lengths)
                prev = {side: arm_angles(pts, side, prev.get(side)) for side in human}
                samples.append(prev)
        ref = {}
        for side in human:
            ref[side] = {
                "abduction": float(np.mean([h[side].abduction for h in samples])),
                "pitch": HOME_HUMAN["pitch"],
                "elbow": float(np.mean([h[side].elbow for h in samples])),
            }
            if lengths:
                ref[side]["lengths"] = lengths[side]
        return ref
