# Yanshee mirror teleoperation

Make a UBTECH **Yanshee** humanoid copy your arm movements in real time from a camera.
A PC (or a Jetson) runs **MediaPipe Pose** on the video, turns your arm pose into angles for the
robot's six arm servos and sends them over Wi-Fi with the robot's REST API (`YanAPI`).
By default it works as a mirror: your right arm moves the robot's left arm.

Hand gestures arm/disarm the robot, an optional panel shows the video together with every servo
angle, and every session can be recorded to CSV to analyse detection and filtering offline.

## Features

- Pose detection with MediaPipe Pose Landmarker (Tasks API, VIDEO mode) on the PC, not on the robot.
- Video from the robot's head camera (MJPEG stream) or from any local camera (`--source N`), which
  is much faster: ~30 fps vs. the robot's fixed ~10 fps.
- **Home sync**: the robot and the person both start from a T-pose; from then on the robot copies how
  far *you* move away from *your* T-pose, so it does not need a perfect one.
- Arm depth reconstructed from 2D foreshortening, because MediaPipe's depth barely sees movements
  towards the camera (see [How it works](#how-it-works)).
- Signed elbow (bending up and down) and both shoulder joints, with continuity between frames so the
  servos do not jump end to end.
- One Euro filter, deadband and step limit per servo; a sender thread that always sends the latest
  command and never queues stale ones.
- Safety: starts disarmed, arming requires holding the home pose, watchdog that disarms and sends the
  robot home if you are lost, legs and neck are never commanded, configurable servo limits.
- Gestures: thumb up = arm, thumb down = disarm, middle finger = quit, victory = next song.
- `--vis`: one window with the video plus a panel with the state, a bar per servo (target and last
  sent value) and a 10 s plot.
- `--log`: per-frame CSV with raw landmarks, angles, targets and commands, plus the robot's responses.
- `--dry-run`: everything except moving the servos.

## Architecture

```
 Yanshee (Raspberry Pi)                      PC / Jetson
 ┌───────────────────────┐   Wi-Fi    ┌────────────────────────────────────────────────┐
 │ camera -> MJPEG :8000 │ ─────────▶ │ 1. Frame reader thread (keeps only the latest) │
 │                       │            │    (robot camera, or a local USB camera)       │
 │                       │            │ 2. MediaPipe Pose + Gesture Recognizer         │
 │                       │            │ 3. Arm rebuild (2D depth) -> human angles      │
 │                       │            │ 4. Home sync -> servo targets (CALIB)          │
 │ REST API :9090        │ ◀───────── │ 5. One Euro + deadband + max step              │
 │   set_servos_angles   │   ~3-5 Hz  │ 6. Sender thread: latest command only          │
 └───────────────────────┘            │ 7. State machine, gestures, watchdog, UI, log  │
                                      └────────────────────────────────────────────────┘
```

## Requirements

- A Yanshee robot on the same network as the PC.
- Python 3.10 (tested on Ubuntu 22.04 / x86_64).
- Tested with: `mediapipe 0.10.32`, `opencv-python 5.0` (the GUI build, not `-headless`),
  `numpy 2.2`, `requests`, `nest_asyncio`, `protobuf 3.20.3` (needed by `lib_ukit`), `pytest`.
- `YanAPI.py` and `lib_ukit/` (UBTECH's SDK) are included in this repo, copied from the robot's
  Raspberry Pi.

## Setup

```bash
git clone https://github.com/dasanplaen-cmyk/Yanshee-Teleoperation.git
cd Yanshee-Teleoperation

python3 -m venv venv && source venv/bin/activate
pip install mediapipe opencv-python numpy requests nest_asyncio "protobuf==3.20.3" pytest

# MediaPipe models, into config/
mkdir -p config
wget -P config https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
wget -P config https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
```

Set the robot IP in `src/config.py` (`ROBOT_IP`) or pass `--ip`.

**`YanAPI.py` fix for remote use.** The copy in this repo is already fixed. If you replace it with
the one from your robot, check `yan_api_init`: the original builds the URL as
`"http://" + robot_ip + ":9090/  v1/"` (two spaces). On the robot itself this goes unnoticed because
the default URL is used, but from a PC every call fails. It must be `":9090/v1/"`.

## Usage

```bash
# Real robot, local camera 2, video + angle panel
python src/mirror.py --ip 192.168.1.62 --source 2 --vis

# Robot camera, servos not moved
python src/mirror.py --ip 192.168.1.62 --dry-run

# Local webcam only, no robot at all
python src/mirror.py --dry-run --source 0 --vis

# Record a session for later analysis (the folder is created if needed)
python src/mirror.py --ip 192.168.1.62 --source 2 --log logs/session1.csv
```

| Option | Description |
|---|---|
| `--ip IP` | Robot IP (default `ROBOT_IP` in `config.py`) |
| `--source S` | Video source: local camera index (`0`, `2`, ...) or a URL. Default: the robot's camera |
| `--res WxH` | Local camera resolution, e.g. `1920x1080` (default `1280x720`) |
| `--dry-run` | Do not move the servos (the robot's camera is still used unless `--source` is given) |
| `--vis` | Video and arm angles in the same window. Without it, only the video with a status line |
| `--log FILE.csv` | Record the session (see [Session logs](#session-logs)) |
| `--no-mirror` | Direct imitation (your right arm moves the robot's right arm) |
| `--music` | Play songs on the robot while it moves (see [Music](#music)) |

### Controls

| Action | Gesture (hold `GESTURE_HOLD_S`, 1 s) | Key (video window or terminal) |
|---|---|---|
| Arm (goes to SYNC) / disarm | thumb up / thumb down | `x` toggles |
| Add a mark to the log | – | `m` |
| Next song (`--music`) | victory | `n` |
| Quit (robot back to rest) | middle finger | `q` or Ctrl+C |

### States

1. **DISARMED**: the robot is at home (arms in a T-pose, all arm servos at 90) and does
   not follow you.
2. **SYNC**: after arming, stand in a T-pose with straight elbows and hold it for `HOME_HOLD_S`.
   Your angles at that moment become the reference.
3. **ARMED**: the robot follows you. Disarming, or not seeing your shoulders, elbows and
   wrists for `LOST_TIMEOUT_S`, sends the robot home and disarms. Quitting sends it to its rest pose.

Stand 2-3 m from the camera so your whole upper body including the hips is visible.

Without `--vis` the window shows only the video, with the state and notices in a small band at the
top left. With `--vis` a panel on the right shows the state (grey = DISARMED, yellow = SYNC,
red = ARMED), the fps, the detected gestures, and a bar per servo with its allowed range, home (90),
the target (coloured dot) and the last angle sent to the robot (white line), plus a 10 s plot.

### Music

`--music` plays the songs in the local `music/` folder on the robot, in random order, and every
`MUSIC_CHECK_S` seconds starts the next one if the robot has stopped playing. The robot plays files
from its own storage, so upload them first (it only uploads the ones the robot does not have):

```bash
python src/upload_music.py --ip 192.168.1.62
```

The volume is `ROBOT_VOL` in `config.py`. `music/` is not part of the repo.

## How it works

### Arm angles (`retarget.py`)

The robot's shoulder is "pitch then abduction": `ShoulderRoll` swings the arm forwards/backwards and
`ShoulderFlex` opens it out to the side. Each person's arm is decomposed the same way:

- **pitch**: rotation about the shoulder axis (0 = arm in the plane of the body, 90 = forwards).
- **abduction**: opening out to the side (0 = down, 90 = T-pose, 180 = up over the side).
- **elbow**: signed elbow flexion (positive = forearm bends up in a T-pose, negative = down).

Every arm direction has two solutions, `(p, a)` and `(p + 180, 180 - a)`. The previous frame's
branch is kept while its pitch is reachable (`BRANCH_KEEP_DEG`); switching branch moves both shoulder
servos almost end to end. Near the T-pose the pitch is undefined (gimbal lock), so it is attenuated
towards 0, and the abduction is then computed as the one that best points the robot arm at yours.

### Depth from 2D foreshortening (`rebuild_arms`)

MediaPipe's world `z` compresses depth: with a forearm pointing at the camera it reported 7-29 deg of
elbow while the forearm was shortened to 10% in the image. So, after syncing in the T-pose (arms in
the image plane, full length visible), each segment's depth is computed from how much it shrinks in
the image, `sqrt(1 - r^2)`. Only the *sign* (towards or away from the camera) comes from MediaPipe,
made robust by `DepthSign` (median of the wrist and three hand points, median over 5 frames and a
3 cm hysteresis). If the upper arm and forearm are aligned in the image, the arm is treated as
straight and both segments share the depth and the upper arm's sign.

### Mapping to the robot (`config.py` → `CALIB`)

```
servo = home + sign * (human - human_at_home)      clipped to [min, max]
```

`human` says which human angle drives each servo, `home` is the servo angle in the home pose (90),
`sign` its direction and `min`/`max` the allowed range (inside the manufacturer's 0-180).

### Filtering and sending (`filters.py`, `robot.py`)

Each target goes through a One Euro filter, then a deadband (`DEADBAND_DEG`) and a maximum step per
command (`MAX_STEP_DEG`). `CommandSender` sends at most `SEND_HZ` commands per second and merges
pending servo values instead of replacing them, so a servo that changed is never lost.

## Configuration

Everything is in `src/config.py`. The most relevant parameters:

| Parameter | Meaning |
|---|---|
| `CALIB` | Per-servo mapping: driving human angle, home, sign, min/max |
| `ARM_SERVOS` | Servos that may be commanded (comment out a side to test one arm) |
| `MIRROR` | Mirror mode (your right arm -> robot's left arm) |
| `ONE_EURO_MIN_CUTOFF`, `ONE_EURO_BETA` | Smoothing vs. lag |
| `DEADBAND_DEG`, `MAX_STEP_DEG`, `SEND_HZ`, `RUNTIME_MS` | Command shaping |
| `HOME_HOLD_S`, `HOME_TOL_DEG` | How long and how precisely the T-pose must be held to arm |
| `MIN_VISIBILITY`, `LOST_TIMEOUT_S` | When the person counts as lost |
| `LOCAL_CAMERA_*`, `UI_HEIGHT` | Local camera format/resolution/fps and window size |
| `GESTURE_HOLD_S`, `GESTURE_EVERY_N_FRAMES` | Gesture timing |

Verify the `sign` of every servo with `--dry-run --vis` before enabling the real robot: move one
arm and check that the corresponding bar moves in the expected direction.

## Session logs

`--log logs/session.csv` writes two files:

- `session.csv`: one row per processed frame with the state, the mark number, the raw MediaPipe
  landmarks (world and image coordinates and visibility for shoulders, elbows, wrists, hips and hand
  points), the human angles, and the target, smoothed and commanded value of each servo.
- `session_sent.csv`: one row per command sent to the robot with the response code and the latency.

The `state` column is `DISARMED`, `SYNC` or `ARMED`.

Press `m` between movements to split the recording into blocks. Because the raw landmarks are saved,
any change to the retargeting can be re-run on a recording without repeating the test.

## Tests

```bash
cd src && python -m pytest -p no:anyio
```

`test_retarget.py` covers the angle decomposition, branch continuity, depth rebuild, home sync,
filters and sender with synthetic poses; `test_gestures.py` covers gesture parsing and triggering.
(`-p no:anyio` avoids an incompatible system-wide pytest plugin; it is harmless otherwise.)

## Project structure

| Path | Contents |
|---|---|
| `src/mirror.py` | Main loop: camera, pose and gestures, state machine, sending, watchdog, UI, log |
| `src/retarget.py` | Landmarks -> human angles -> servo targets; depth rebuild; home sync |
| `src/filters.py` | One Euro filter, deadband and step limit |
| `src/robot.py` | YanAPI client, dry-run simulator, sender thread |
| `src/dashboard.py` | `--vis` panel and the status overlay |
| `src/session_log.py` | `--log` CSV recording |
| `src/upload_music.py` | Uploads the songs in `music/` to the robot (for `--music`) |
| `src/config.py` | All parameters and the servo calibration |
| `src/test_*.py` | Tests |
| `config/` | MediaPipe `.task` models (downloaded, see Setup; not in the repo) |
| `YanAPI.py`, `lib_ukit/` | UBTECH SDK, copied from the robot |
| `logs/`, `music/` | Local session recordings and songs (not in the repo) |

## Measured behaviour and known limitations

- **Robot camera**: always 640x480 at ~10 fps, whatever resolution is requested. A local camera in
  MJPG gives ~30 fps at 720p/1080p (YUYV only reaches 30 fps at 640x480).
- **Latency**: each `set_servos_angles` call takes ~290 ms to return (a read takes ~100 ms), so the
  robot receives ~3-4 commands per second and lags ~0.3-0.5 s behind you.
- **Arm pointing at the camera**: both segments are very short in the image and their direction is
  mostly noise; tracking is unreliable in that pose.
- **Forwards vs. backwards**: in the image both look the same; the sign comes from MediaPipe, which
  can be systematically wrong for arms behind the body.
- **Reach**: with `ShoulderRoll` limited to 10-170 (±80 deg of pitch) the robot cannot point its arm
  straight forwards at shoulder height; it stays at its limit instead of jumping between solutions.

## Safety

- The robot starts disarmed and only moves after the home sync.
- Only the six arm servos are ever commanded; legs and neck never are.
- Use `--dry-run --vis` first after any change to `CALIB`.
- Keep the robot on a stable surface with space around its arms; `q`, Ctrl+C or a thumb down stop it.

## Third-party components

- `YanAPI.py` and `lib_ukit/` are part of UBTECH's Yanshee SDK
  ([UBTEDU/Yanshee-SDK](https://github.com/UBTEDU/Yanshee-SDK),
  [YanAPI documentation](https://ubtrobot-new.oss-cn-shenzhen.aliyuncs.com/static/MINI/Yanshee/yanapi/html-en/YanAPI.html)).
- MediaPipe models from Google ([Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker),
  [Gesture Recognizer](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer)).
