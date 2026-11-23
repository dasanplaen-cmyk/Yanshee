"""ROS 2 driver: Xbox controller -> Yanshee.

  /cmd_vel (geometry_msgs/Twist, from teleop_twist_joy) -> walking gait (YanAPI.control_motion_gait)
  /joy     (sensor_msgs/Joy)                            -> built-in motions and emergency stop

The robot's gait API takes integer speeds in [-5, 5] forwards/backwards (speed_v) and sideways
(speed_h); it has no turn rate, so angular.z is mapped to a sidestep and turning is done with the
built-in "turn around" motion on the D-pad.

Every call to the robot is an HTTP request (100-300 ms), so the calls run in their own thread and a
gait command is only sent when the quantised speed changes.
"""
import os
import sys
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Joy

# Xbox controller layout with the Linux xpad driver (checked with `ros2 topic echo /joy`)
BTN_A, BTN_B, BTN_X, BTN_Y, BTN_LB, BTN_RB, BTN_BACK, BTN_START = range(8)
AXIS_DPAD_X = 6  # +1 = left, -1 = right


class DryRunApi:
    """Stands in for YanAPI with --dry-run: logs the calls instead of moving the robot."""

    def __init__(self, logger):
        self.log = logger

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.log.info(f"[dry-run] {name}({', '.join([repr(a) for a in args] + [f'{k}={v!r}' for k, v in kwargs.items()])})")
            return {"code": 0, "data": {}, "msg": "dry-run"}
        return call


def load_yanapi(root):
    """Import UBTECH's YanAPI.py from the project root (it is not a ROS package)."""
    # lib_ukit was generated with an old protoc; with protobuf >= 4 it only loads with the
    # pure-Python implementation (the system Python used by ROS 2 Humble ships protobuf 5)
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    if root not in sys.path:
        sys.path.insert(0, root)
    import YanAPI
    return YanAPI


class YansheeDriver(Node):
    def __init__(self):
        super().__init__("yanshee_driver")
        p = self.declare_parameter
        self.robot_ip = p("robot_ip", "192.168.1.62").value
        self.dry_run = p("dry_run", False).value
        root = p("yanshee_root", os.environ.get("YANSHEE_ROOT", os.path.expanduser("~/Yanshee"))).value
        self.linear_full_scale = p("linear_full_scale", 0.7).value    # linear.x that means full speed (5)
        self.angular_full_scale = p("angular_full_scale", 0.4).value  # angular.z that means full sidestep (5)
        self.max_speed = p("max_speed", 5).value                      # cap on |speed_v|, |speed_h| (API: 5)
        self.invert_v = p("invert_v", False).value
        self.invert_h = p("invert_h", False).value
        self.gait_period = p("gait_period", 1).value                  # control_motion_gait `period` (1-5)
        self.cmd_timeout = p("cmd_timeout", 0.5).value                # no /cmd_vel for this long -> stop
        self.rate_hz = p("rate_hz", 5.0).value                         # max requests per second to the robot
        self.motion_speed = p("motion_speed", "normal").value

        if self.dry_run:
            self.api = DryRunApi(self.get_logger())
        else:
            self.api = load_yanapi(root)
            self.api.yan_api_init(self.robot_ip)
        self.get_logger().info(f"Yanshee at {self.robot_ip}{' (dry-run)' if self.dry_run else ''}")

        self._lock = threading.Lock()
        self._target = (0, 0)       # (speed_v, speed_h) wanted
        self._last_cmd_t = 0.0
        self._actions = []           # pending one-shot actions from /joy
        self._prev_buttons = []
        self._prev_dpad = 0.0
        self._walking = (0, 0)       # what the robot was last told
        self._stop = threading.Event()

        self.create_subscription(Twist, "cmd_vel", self.on_cmd_vel, 10)
        self.create_subscription(Joy, "joy", self.on_joy, 10)
        self._worker = threading.Thread(target=self.run, daemon=True)
        self._worker.start()

    # ---------- ROS callbacks (fast: they only store what is wanted) ----------

    def _quantise(self, value, full_scale, invert):
        speed = round(max(-1.0, min(1.0, value / full_scale)) * self.max_speed)
        return -speed if invert else speed

    def on_cmd_vel(self, msg: Twist):
        v = self._quantise(msg.linear.x, self.linear_full_scale, self.invert_v)
        # The gait API has no turn rate: yaw from the stick (and linear.y, if any) becomes a sidestep
        h = self._quantise(msg.angular.z + msg.linear.y, self.angular_full_scale, self.invert_h)
        with self._lock:
            self._target = (int(v), int(h))
            self._last_cmd_t = time.monotonic()

    def on_joy(self, msg: Joy):
        pressed = [b and not (i < len(self._prev_buttons) and self._prev_buttons[i])
                   for i, b in enumerate(msg.buttons)]
        self._prev_buttons = list(msg.buttons)
        actions = []
        if len(pressed) > BTN_BACK and pressed[BTN_BACK]:
            actions.append(("stop", None))
        if len(pressed) > BTN_START and pressed[BTN_START]:
            actions.append(("motion", ("reset", "")))
        if len(pressed) > BTN_A and pressed[BTN_A]:
            actions.append(("motion", ("wave", "both")))
        if len(pressed) > BTN_B and pressed[BTN_B]:
            actions.append(("motion", ("bow", "")))
        if len(pressed) > BTN_Y and pressed[BTN_Y]:
            actions.append(("motion", ("raise", "both")))
        dpad = msg.axes[AXIS_DPAD_X] if len(msg.axes) > AXIS_DPAD_X else 0.0
        if abs(dpad) > 0.5 and abs(self._prev_dpad) <= 0.5:
            actions.append(("motion", ("turn around", "left" if dpad > 0 else "right")))
        self._prev_dpad = dpad
        if actions:
            with self._lock:
                self._actions.extend(actions)

    # ---------- worker thread: the only place that talks to the robot ----------

    def run(self):
        period = 1.0 / self.rate_hz
        while not self._stop.is_set():
            t0 = time.monotonic()
            with self._lock:
                target = self._target if t0 - self._last_cmd_t < self.cmd_timeout else (0, 0)
                actions, self._actions = self._actions, []
            try:
                for kind, arg in actions:
                    self.do_action(kind, arg)
                self.update_gait(target)
            except Exception as e:  # the Wi-Fi may drop: keep the node alive and retry
                self.get_logger().error(f"robot request failed: {e}", throttle_duration_sec=2.0)
            self._stop.wait(max(0.0, period - (time.monotonic() - t0)))

    def update_gait(self, target):
        if target == self._walking:
            return
        if target == (0, 0):
            self.api.exit_motion_gait()
            self.get_logger().info("gait: stop")
        else:
            self.api.control_motion_gait(speed_v=target[0], speed_h=target[1], steps=0,
                                         period=self.gait_period, wave=True)
            self.get_logger().info(f"gait: speed_v={target[0]} speed_h={target[1]}")
        self._walking = target

    def do_action(self, kind, arg):
        if kind == "stop":
            self.get_logger().warn("emergency stop: gait off, motions stopped, back to rest")
            with self._lock:
                self._target = (0, 0)
                self._last_cmd_t = 0.0
            self.api.exit_motion_gait()
            self.api.stop_play_motion()
            self.api.start_play_motion(name="reset")
            self._walking = (0, 0)
        elif kind == "motion":
            if self._walking != (0, 0):
                self.get_logger().warn(f"ignoring '{arg[0]}': stop walking first")
                return
            name, direction = arg
            self.get_logger().info(f"motion: {name} {direction}".strip())
            self.api.start_play_motion(name=name, direction=direction, speed=self.motion_speed)

    def shutdown(self):
        self._stop.set()
        self._worker.join(timeout=2.0)
        try:
            self.api.exit_motion_gait()
        except Exception as e:
            self.get_logger().error(f"could not stop the gait on exit: {e}")


def main():
    rclpy.init()
    node = YansheeDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
