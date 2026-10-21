"""Tests with synthetic poses (no camera or robot): python -m pytest test_retarget.py"""
import numpy as np
import pytest

import config as C
from filters import JointSmoother
from retarget import (L_ELBOW, L_HIP, L_SHOULDER, L_WRIST, R_ELBOW, R_HIP, R_SHOULDER, R_WRIST,
                      ARM_CHAINS, HomeSync, _unit, _wrap, arm_angles, default_reference, human_to_robot,
                      rebuild_arms, segment_lengths)

# Person facing the camera, in MediaPipe-world-like coordinates (x right of the image, y down,
# z away from the camera). Their anatomical left appears on the right of the image (+x);
# their chest points towards the camera (-z).
LEFT, UP, FWD = np.array([1.0, 0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, -1.0])


def person(l_upper, l_fore, r_upper, r_fore):
    pts = np.zeros((33, 3))
    pts[L_HIP], pts[R_HIP] = 0.1 * LEFT, -0.1 * LEFT
    pts[L_SHOULDER], pts[R_SHOULDER] = 0.5 * UP + 0.18 * LEFT, 0.5 * UP - 0.18 * LEFT
    pts[L_ELBOW] = pts[L_SHOULDER] + 0.28 * l_upper
    pts[L_WRIST] = pts[L_ELBOW] + 0.25 * l_fore
    pts[R_ELBOW] = pts[R_SHOULDER] + 0.28 * r_upper
    pts[R_WRIST] = pts[R_ELBOW] + 0.25 * r_fore
    return pts


DOWN = -UP


def both(pts):
    return {"left": arm_angles(pts, "left"), "right": arm_angles(pts, "right")}


def sideways(deg, outward):
    """Arm direction with `deg` degrees of abduction (0 = down, 90 = T-pose)."""
    r = np.radians(deg)
    return np.cos(r) * DOWN + np.sin(r) * outward


HOME = person(LEFT, LEFT, -LEFT, -LEFT)  # T-pose, straight elbows


def test_arms_down_straight():
    pts = person(DOWN, DOWN, DOWN, DOWN)
    for side in ("left", "right"):
        a = arm_angles(pts, side)
        assert a.abduction == pytest.approx(0, abs=1)
        assert a.pitch == pytest.approx(0, abs=1)
        assert a.elbow == pytest.approx(0, abs=1)


def test_left_arm_sideways_right_arm_forward_bent():
    pts = person(LEFT, LEFT, FWD, UP)
    l, r = arm_angles(pts, "left"), arm_angles(pts, "right")
    assert l.abduction == pytest.approx(90, abs=1)
    assert l.pitch == pytest.approx(0, abs=1)  # in a T-pose the pitch is attenuated to 0
    assert r.pitch == pytest.approx(90, abs=1)
    assert r.abduction == pytest.approx(0, abs=1)
    assert r.elbow == pytest.approx(90, abs=1)


def test_arm_overhead():
    # From home, an arm pointing up is read as raised over the side (what the robot can reach)
    a = arm_angles(person(UP, UP, DOWN, DOWN), "left")
    assert a.abduction == pytest.approx(180, abs=1)
    assert a.pitch == pytest.approx(0, abs=1)
    # Even when raised in front: the robot cannot reach pitch 180, but it can reach abduction 180
    b = arm_angles(person(UP, UP, DOWN, DOWN), "left", prev=arm_angles(person(FWD, FWD, DOWN, DOWN), "left"))
    assert b.abduction == pytest.approx(180, abs=1)
    assert b.pitch == pytest.approx(0, abs=1)


def track(directions, side="left"):
    """Angles for a sequence of (straight) arm directions, chaining prev."""
    out, prev = [], None
    outward = LEFT if side == "left" else -LEFT
    for d in directions:
        arm = d if side == "left" else d * np.array([-1, 1, 1])  # mirror x for the right arm
        pts = person(arm, arm, -outward, -outward) if side == "left" else person(LEFT, LEFT, arm, arm)
        prev = arm_angles(pts, side, prev)
        out.append(prev)
    return out


def max_jump(seq):
    return max(max(abs(b.abduction - a.abduction), abs(b.pitch - a.pitch)) for a, b in zip(seq, seq[1:]))


def test_arm_above_shoulder_with_depth_noise_does_not_flip():
    # What the plot showed: arm above the shoulder, out to the side, with MediaPipe's depth
    # oscillating in front/behind. The pitch used to jump between +180 and -180.
    base = sideways(135, LEFT)
    seq = track([_unit(base + (0.08 if i % 2 else -0.08) * FWD) for i in range(20)])
    assert max_jump(seq) < 20  # +-4.6 deg of arm noise is a real +-6.5 deg of pitch here
    for a in seq:
        assert a.abduction == pytest.approx(135, abs=6)
        assert abs(a.pitch) < 10


def big_jumps(seq, limit=30):
    """Jumps larger than `limit`, flagging whether they coincide with a branch switch."""
    out = []
    for a, b in zip(seq, seq[1:]):
        jump = max(abs(b.abduction - a.abduction), abs(b.pitch - a.pitch))
        if jump > limit:
            switched = abs(_wrap(b.branch_pitch - a.branch_pitch)) > 90
            out.append((jump, switched))
    return out


@pytest.mark.parametrize("side", ["left", "right"])
def test_sweeps(side):
    # Raising the arm over the side, from down to up: continuous
    frontal = [sideways(a, LEFT) for a in range(0, 181, 3)]
    assert max_jump(track(frontal, side)) < 8
    # Raising it in front: continuous until past the horizontal; there a single branch switch (with
    # hysteresis) towards the solution the robot can reach, instead of asking for pitch 170
    angles = list(range(0, 171, 3))
    seq = track([np.cos(np.radians(a)) * DOWN + np.sin(np.radians(a)) * FWD for a in angles], side)
    jumps = [i for i, (a, b) in enumerate(zip(seq, seq[1:]))
             if max(abs(b.abduction - a.abduction), abs(b.pitch - a.pitch)) > 8]
    assert len(jumps) == 1 and 95 <= angles[jumps[0] + 1] <= 115
    assert abs(seq[-1].pitch) < 20 and seq[-1].abduction > 150  # up: over the side, small pitch
    # Any smooth path over the sphere: large jumps can only be branch switches
    rng = np.random.default_rng(0)
    d, path = sideways(90, LEFT), []
    for _ in range(400):
        d = _unit(d + 0.05 * rng.normal(size=3))
        path.append(d)
    jumps = big_jumps(track(path, side))
    assert all(switched for _, switched in jumps)
    assert len(jumps) <= 6


def test_home_pose_maps_to_robot_home():
    out = human_to_robot(both(HOME), C.CALIB, mirror=True, reference=default_reference())
    for side in ("Left", "Right"):
        assert out[f"{side}ShoulderFlex"] == pytest.approx(C.CALIB[f"{side}ShoulderFlex"]["home"], abs=1)
        assert out[f"{side}ElbowFlex"] == pytest.approx(C.CALIB[f"{side}ElbowFlex"]["home"], abs=1)
        assert out[f"{side}ShoulderRoll"] == pytest.approx(C.CALIB[f"{side}ShoulderRoll"]["home"], abs=1)


def test_home_sync_absorbs_imperfect_home():
    # The person's T-pose is not perfect: arms at 75 degrees and elbows bent 10
    pts = person(sideways(75, LEFT), sideways(85, LEFT), sideways(75, -LEFT), sideways(85, -LEFT))
    sync = HomeSync(hold_s=1.0, tol_deg=20)
    ref = None
    for i in range(12):
        ref = sync.update(both(pts), i * 0.1)
    assert ref is not None
    assert ref["left"]["abduction"] == pytest.approx(75, abs=1)
    assert ref["left"]["elbow"] == pytest.approx(10, abs=1)
    # With that reference, THEIR T-pose is the robot's home
    out = human_to_robot(both(pts), C.CALIB, mirror=True, reference=ref)
    for name in ("LeftShoulderFlex", "LeftElbowFlex", "RightShoulderFlex", "RightElbowFlex"):
        assert out[name] == pytest.approx(C.CALIB[name]["home"], abs=1)


def test_home_sync_needs_continuous_hold():
    sync = HomeSync(hold_s=1.0, tol_deg=20)
    assert sync.update(both(HOME), 0.0) is None
    assert sync.update(both(HOME), 0.5) is None
    assert sync.update(both(person(DOWN, DOWN, DOWN, DOWN)), 0.6) is None  # breaks the pose
    assert sync.progress(0.6) == 0
    assert sync.update(both(HOME), 0.7) is None
    assert sync.update(both(HOME), 1.5) is None  # only 0.8 s in a row
    assert sync.update(both(HOME), 1.7) is not None


def test_mirror_swaps_sides():
    # The person lowers their LEFT arm and keeps the right one in a T-pose
    human = both(person(DOWN, DOWN, -LEFT, -LEFT))
    ref = default_reference()
    mirrored = human_to_robot(human, C.CALIB, mirror=True, reference=ref)
    direct = human_to_robot(human, C.CALIB, mirror=False, reference=ref)
    for out, moved, still in ((mirrored, "RightShoulderFlex", "LeftShoulderFlex"),
                              (direct, "LeftShoulderFlex", "RightShoulderFlex")):
        c = C.CALIB[moved]
        assert out[moved] == pytest.approx(np.clip(c["home"] - c["sign"] * 90, c["min"], c["max"]), abs=1)
        assert out[still] == pytest.approx(C.CALIB[still]["home"], abs=1)


def test_only_requested_servos():
    out = human_to_robot(both(HOME), C.CALIB, mirror=True, reference=default_reference(),
                         servos=["RightElbowFlex"])
    assert set(out) == {"RightElbowFlex"}


def test_output_is_clamped():
    human = {"left": arm_angles(person(UP, UP, UP, UP), "left")}
    out = human_to_robot(human, C.CALIB, mirror=False, reference=default_reference())
    for name, v in out.items():
        assert C.CALIB[name]["min"] <= v <= C.CALIB[name]["max"]


def test_smoother_deadband_and_step_limit():
    s = JointSmoother(deadband_deg=3, max_step_deg=25, min_cutoff=1.0, beta=0.0)
    assert s.next_command({"X": 150}, {"X": 90}) == {"X": 115}  # limited to 25
    assert s.next_command({"X": 116}, {}) == {}                  # within the deadband


def test_driving_sides():
    from retarget import driving_sides
    right_arm = ["RightShoulderRoll", "RightShoulderFlex", "RightElbowFlex"]
    assert driving_sides(right_arm, mirror=True) == {"left"}   # mirror: driven by your left arm
    assert driving_sides(right_arm, mirror=False) == {"right"}
    assert driving_sides(C.CALIB, mirror=True) == {"left", "right"}


def test_home_sync_one_arm_ignores_the_other():
    # Only the person's left arm is synced; the right one may be down
    pts = person(LEFT, LEFT, DOWN, DOWN)
    sync = HomeSync(hold_s=1.0, tol_deg=20)
    ref = None
    for i in range(12):
        ref = sync.update({"left": arm_angles(pts, "left")}, i * 0.1)
    assert ref is not None and set(ref) == {"left"}


def test_servo_starting_inside_deadband_still_moves_later():
    # Real bug from sesion1.csv: when arming, robot and person are at home (difference < deadband);
    # that servo was not sent and then took its own target as reference -> it never moved.
    s = JointSmoother(deadband_deg=3, max_step_deg=25, min_cutoff=1.0, beta=0.0)
    assert s.next_command({"Elbow": 91}, {"Elbow": 90}) == {}   # first frame: within the deadband
    assert s.next_command({"Elbow": 120}, {}) == {"Elbow": 115}  # afterwards it must move
    assert s.next_command({"Elbow": 120}, {}) == {"Elbow": 120}


def test_servo_without_known_position_is_not_commanded():
    s = JointSmoother(deadband_deg=3, max_step_deg=25, min_cutoff=1.0, beta=0.0)
    assert s.next_command({"Elbow": 150}, {}) == {}


def camera_views(pts, px_per_m=400.0):
    """What MediaPipe returns for a pose: the image (projection, in pixels) is correct, but in world,
    when a forearm points at the camera, it leaves it almost aligned with the upper arm ("straight"
    elbow) keeping its length and the sign of its depth, as in sesion1.csv."""
    img = pts[:, :2] * px_per_m
    world = pts.copy()
    for sh, el, wr in (ARM_CHAINS["left"], ARM_CHAINS["right"]):
        upper, fore = pts[el] - pts[sh], pts[wr] - pts[el]
        depth = abs(fore[2]) / np.linalg.norm(fore)
        seen = _unit(0.25 * _unit(fore) + depth * _unit(upper))
        world[wr] = world[el] + np.linalg.norm(fore) * seen
    return world, img


def test_rebuild_recovers_elbow_bent_towards_camera():
    # Left arm in a T-pose with the forearm pointing at the camera (-z): elbow at 90
    lengths = segment_lengths(camera_views(HOME)[1])
    bent = person(LEFT, FWD, -LEFT, -LEFT)
    world, img = camera_views(bent)
    assert arm_angles(world, "left").elbow < 30                    # MediaPipe alone: barely visible
    rebuilt = rebuild_arms(world, img, lengths)
    assert arm_angles(rebuilt, "left").elbow == pytest.approx(90, abs=3)


def test_rebuild_keeps_poses_in_the_image_plane():
    lengths = segment_lengths(camera_views(HOME)[1])
    for pose in (HOME, person(DOWN, DOWN, DOWN, DOWN), person(LEFT, UP, -LEFT, UP)):
        world, img = camera_views(pose)
        for side in ("left", "right"):
            a, b = arm_angles(pose, side), arm_angles(rebuild_arms(world, img, lengths), side)
            assert (b.abduction, b.elbow) == pytest.approx((a.abduction, a.elbow), abs=2)


def test_home_sync_measures_segment_lengths():
    sync = HomeSync(hold_s=1.0, tol_deg=20)
    world, img = camera_views(HOME)
    ref = None
    for i in range(12):
        ref = sync.update(both(HOME), i * 0.1, frame=(world, img))
    assert ref["left"]["lengths"] == pytest.approx(segment_lengths(img)["left"])
    assert ref["left"]["elbow"] == pytest.approx(0, abs=1)


@pytest.mark.parametrize("side", ["left", "right"])
def test_elbow_sign_up_and_down(side):
    out = LEFT if side == "left" else -LEFT
    up_pose = person(LEFT, UP, -LEFT, UP)       # T-pose, forearms up
    down_pose = person(LEFT, DOWN, -LEFT, DOWN)  # T-pose, forearms down
    assert arm_angles(up_pose, side).elbow == pytest.approx(90, abs=1)
    assert arm_angles(down_pose, side).elbow == pytest.approx(-90, abs=1)


def test_elbow_sign_held_when_bending_towards_camera():
    prev = arm_angles(person(LEFT, sideways(135, LEFT), -LEFT, -LEFT), "left")  # bent upwards
    assert prev.elbow > 0
    a = arm_angles(person(LEFT, FWD, -LEFT, -LEFT), "left", prev=prev)
    assert a.elbow == pytest.approx(90, abs=1)  # out of plane: magnitude with the previous sign


def test_sender_keeps_servos_from_earlier_frames():
    # Bug from sesion2.csv: the pending command was replaced and servos not yet sent were lost
    from robot import CommandSender
    s = CommandSender(robot=None, hz=5)
    s.submit({"Flex": 100}, 200)
    s.submit({"Roll": 80}, 200)
    s.submit({"Roll": 85}, 200)
    assert s._pending == ({"Flex": 100, "Roll": 85}, 200)


def test_straight_arm_backwards_keeps_elbow_straight():
    # sesion2.csv: straight arm backwards; in the image the upper arm barely shortens (its T-pose
    # reference is short) while the forearm does. Without the "aligned in 2D" rule the elbow came out ~70 deg.
    lengths = segment_lengths(camera_views(HOME)[1])
    lengths = {s: (0.8 * u, f) for s, (u, f) in lengths.items()}  # upper arm reference too short
    back = _unit(DOWN - 0.6 * FWD)
    world, img = camera_views(person(LEFT, LEFT, back, back))
    a = arm_angles(rebuild_arms(world, img, lengths), "right")
    assert abs(a.elbow) < 10


def test_depth_sign_ignores_outliers_and_needs_evidence_to_flip():
    from retarget import DepthSign
    pts = person(DOWN, DOWN, _unit(DOWN + 0.5 * FWD), _unit(DOWN + 0.5 * FWD))  # right arm forwards
    world = pts.copy()
    world[[18, 20, 22]] = world[16]  # hand at the wrist
    d = DepthSign(window=5, hyst_m=0.03)
    for _ in range(5):
        assert d.update(world)["right"][0] < 0          # towards the camera (-z)
    bad = world.copy()
    bad[14, 2] += 0.5                                   # one frame with a badly detected elbow
    assert d.update(bad)["right"][0] < 0                # the median ignores it
    tiny = world.copy()
    tiny[14, 2] = tiny[12, 2] + 0.01                     # small evidence (1 cm) in the opposite direction
    for _ in range(5):
        assert d.update(tiny)["right"][0] < 0           # below the hysteresis: no change
    wrist_outlier = world.copy()
    wrist_outlier[16, 2] += 0.5                          # wrong wrist, but pinky/index/thumb fine
    assert DepthSign(window=1).update(wrist_outlier)["right"][1] == pytest.approx(
        DepthSign(window=1).update(world)["right"][1])


def test_straight_arm_uses_upper_arm_depth_sign():
    # sesion3.csv: straight arm backwards, but MediaPipe puts the hand in front of the elbow.
    # The shared sign must be the upper arm's (backwards), not the sum with the hand.
    lengths = segment_lengths(camera_views(HOME)[1])
    back = _unit(DOWN - 0.6 * FWD)
    world, img = camera_views(person(LEFT, LEFT, back, back))
    dz = {"right": (+0.05, -0.13)}  # upper arm moving away from the camera, (biased) hand towards it
    a = arm_angles(rebuild_arms(world, img, lengths, dz=dz), "right")
    assert a.pitch < -20


def test_abduction_does_not_wrap_overhead():
    # Left arm vertical, slightly towards the other side and slightly towards its own side
    over = arm_angles(person(_unit(UP - 0.03 * LEFT), _unit(UP - 0.03 * LEFT), DOWN, DOWN), "left")
    under = arm_angles(person(_unit(UP + 0.03 * LEFT), _unit(UP + 0.03 * LEFT), DOWN, DOWN), "left")
    assert over.abduction == pytest.approx(182, abs=1)
    assert under.abduction == pytest.approx(178, abs=1)


def test_lowering_forward_from_overhead_keeps_branch():
    # Arms up (over the side) and lowered straight in front to slightly below the horizontal:
    # it used to switch branch at the horizontal (ShoulderRoll 170 -> 10).
    seq = [arm_angles(person(UP, UP, DOWN, DOWN), "left")]
    for a in range(180, 69, -3):  # angle from down in the sagittal plane: 180 = up, 90 = forwards
        d = np.cos(np.radians(a)) * DOWN + np.sin(np.radians(a)) * FWD
        seq.append(arm_angles(person(d, d, DOWN, DOWN), "left", prev=seq[-1]))
    assert max_jump(seq) < 10
    assert seq[-1].abduction > 150 and seq[-1].pitch < -90  # stays on the "from above" branch
