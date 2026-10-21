"""Parameters. CALIB signs and limits are robot-specific: verify them with --dry-run --vis."""

ROBOT_IP = "192.168.1.62"          # Yanshee IP address
STREAM_RESOLUTION = "640x480"       # open_vision_stream (the robot ignores it: always 640x480 at ~10 fps)
STREAM_URL = "http://{ip}:8000/stream.mjpg"
CAMERA_TIMEOUT_S = 10.0             # no frame at all within this time at startup -> exit with an error
# Local camera (--source N). By default OpenCV opens it as YUYV, which on this webcam only gives
# 30 fps at 640x480 (10 fps at 720p, 5 at 1080p). MJPG gives 30 fps up to 1920x1080. MediaPipe
# downscales the image to ~256 px, so above 720p detection barely improves and display costs more.
LOCAL_CAMERA_FOURCC = "MJPG"
LOCAL_CAMERA_SIZE = (1280, 720)     # can be changed with --res 1920x1080
LOCAL_CAMERA_FPS = 30
UI_HEIGHT = 720                     # window height (video + angle panel, --vis)

MIRROR = True                       # your right arm moves the robot's left arm

# Sending to the robot. set_servos_angles accepts a runtime of 200 to 4000 ms,
# so there is no point in sending commands faster than ~5 Hz.
SEND_HZ = 5.0
RUNTIME_MS = 200
FIRST_MOVE_RUNTIME_MS = 1500        # first move after arming: slow
DEADBAND_DEG = 3.0                  # do not send smaller changes
MAX_STEP_DEG = 15.0                 # maximum change per command
ONE_EURO_MIN_CUTOFF = 2.5           # One Euro angle smoothing: lower = smoother but laggier
ONE_EURO_BETA = 0.7              

# Safety
MIN_VISIBILITY = 0.5                # minimum visibility of shoulders/elbows/wrists
LOST_TIMEOUT_S = 2.0                # person not seen for this long -> robot to home and disarm
ARM_SERVOS = [
    "RightShoulderRoll", "RightShoulderFlex", "RightElbowFlex",
    "LeftShoulderRoll", "LeftShoulderFlex", "LeftElbowFlex",
]
# Legs and neck are NEVER commanded by the teleoperation.

# Arm depth from 2D foreshortening (measured in the T-pose when syncing) instead of MediaPipe's z,
# which barely sees movements towards the camera. See rebuild_arms in retarget.py.
DEPTH_FROM_2D = True

GESTURE_HOLD_S = 1.0               # a gesture must be held this long to trigger
# Gestures only on 1 out of every N frames (the last result is reused). The recognizer costs 12-20 ms;
# with it every frame took over 33 ms, so with a 30 fps camera only every other frame was processed.
GESTURE_EVERY_N_FRAMES = 3

ROBOT_VOL = 15
MUSIC_CHECK_S = 5.0                 # --music: how often to check whether the robot is still playing

# Home sync (arms in a T-pose, elbows straight): when arming, the robot goes to home and waits for
# the person to hold that pose for HOME_HOLD_S in a row, with their angles within HOME_TOL_DEG.
HOME_HOLD_S = 1.0
HOME_TOL_DEG = 20.0
HOME_RUNTIME_MS = 1500              # slow move to home

# servo = home + sign * (human - human_at_home), clipped to [min, max].
# home : servo angle in the home pose (all 90: arms in a T-pose, elbows straight).
# human: human angle that drives it (see retarget.py): "pitch" (arm forwards/backwards),
#        "abduction" (arm out to the side) or "elbow".
#        ShoulderFlex is the one that opens the arm into the T-pose (rest -> home only changes Flex
#        and Elbow), so ShoulderRoll is left for the pitch.
# sign : direction of each servo relative to the human angle (opposite on left and right).
# min/max: within the manufacturer's 0-180. At rest ShoulderFlex is at 140 (right) and 40 (left):
#        going past those values may push the arm against the body.
CALIB = {
    "RightShoulderRoll": {"human": "pitch",     "home": 90, "sign": -1, "min": 10, "max":170},
    "RightShoulderFlex": {"human": "abduction", "home": 90, "sign": -1, "min": 20, "max": 170},
    "RightElbowFlex":    {"human": "elbow",     "home": 90, "sign": -1, "min": 20, "max": 180},
    "LeftShoulderRoll":  {"human": "pitch",     "home": 90, "sign": +1, "min": 10, "max": 170},
    "LeftShoulderFlex":  {"human": "abduction", "home": 90, "sign": +1, "min": 20, "max": 170},
    "LeftElbowFlex":     {"human": "elbow",     "home": 90, "sign": +1, "min": 20, "max": 180},
}
