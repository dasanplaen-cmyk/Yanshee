"""Tests of the gesture logic (no camera): python -m pytest test_gestures.py"""
from types import SimpleNamespace as NS

from mirror import ARM_GESTURE, DISARM_GESTURE, GestureTrigger, detect_gestures


def result(*hands):
    """hands: (hand label, gesture or None)."""
    return NS(handedness=[[NS(category_name=h)] for h, _ in hands],
              gestures=[[NS(category_name=g)] if g else [] for _, g in hands])


def test_detect_gestures():
    assert detect_gestures(result()) == {"Left": "None", "Right": "None"}
    assert detect_gestures(result(("Right", "Thumb_Up"))) == {"Left": "None", "Right": "Thumb_Up"}
    assert detect_gestures(result(("Left", None), ("Right", "Thumb_Down"))) == {"Left": "None", "Right": "Thumb_Down"}


def test_trigger_fires_once_after_hold():
    t = GestureTrigger(hold_s=1.0)
    fired = [t.update(DISARM_GESTURE, i * 0.1) for i in range(40)]  # thumb down for 4 s in a row
    assert fired.count(DISARM_GESTURE) == 1                         # a single event, not one per frame
    assert fired.index(DISARM_GESTURE) == 10


def test_trigger_needs_release_and_continuous_hold():
    t = GestureTrigger(hold_s=1.0)
    for i in range(5):
        assert t.update(ARM_GESTURE, i * 0.1) is None
    assert t.update(None, 0.5) is None                 # released before 1 s: does not count
    events = [t.update(ARM_GESTURE, 0.6 + i * 0.1) for i in range(12)]
    assert events.count(ARM_GESTURE) == 1
    t.update(None, 2.0)
    assert [t.update(ARM_GESTURE, 2.1 + i * 0.1) for i in range(12)].count(ARM_GESTURE) == 1  # after releasing, again


from mirror import QUIT_GESTURE, is_middle_finger


def hand(extended):
    """Upright hand (fingers up in the image); `extended`: which fingers (index..pinky) are straight."""
    pts = [NS(x=0.5, y=0.9) for _ in range(21)]
    for k, (mcp, pip, tip) in enumerate(((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))):
        x = 0.44 + 0.04 * k
        pts[mcp], pts[pip] = NS(x=x, y=0.7), NS(x=x, y=0.6)
        pts[tip] = NS(x=x, y=0.4) if extended[k] else NS(x=x, y=0.75)  # folded: tip goes back towards the palm
    return pts


def test_middle_finger_detection():
    assert is_middle_finger(hand((False, True, False, False)))
    assert not is_middle_finger(hand((True, True, True, True)))    # open hand
    assert not is_middle_finger(hand((False, False, False, False)))  # fist
    assert not is_middle_finger(hand((True, True, False, False)))  # victory


def test_middle_finger_overrides_model_label():
    r = result(("Right", "None"))
    r.hand_landmarks = [hand((False, True, False, False))]
    assert detect_gestures(r)["Right"] == QUIT_GESTURE
