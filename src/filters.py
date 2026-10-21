"""Smoothing and limiting of the angles before sending them to the robot."""
import math


class OneEuro:
    """One Euro filter: smooths a lot when still and little when moving fast."""

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class JointSmoother:
    """One Euro per servo + deadband + maximum step per command."""

    def __init__(self, deadband_deg, max_step_deg, min_cutoff, beta):
        self.deadband = deadband_deg
        self.max_step = max_step_deg
        self.min_cutoff, self.beta = min_cutoff, beta
        self.filters = {}
        self.sent = {}  # last angle sent, per servo

    def smooth(self, targets: dict, t: float) -> dict:
        out = {}
        for name, value in targets.items():
            f = self.filters.setdefault(name, OneEuro(self.min_cutoff, self.beta))
            out[name] = f(value, t)
        return out

    def next_command(self, smoothed: dict, current: dict) -> dict:
        """Return only the servos worth moving, with a limited step.

        current: actual robot angles (read when arming). They are each servo's starting point until
        something is sent to it. Previously, a servo that did not exceed the deadband on that first
        frame took its own target as reference and never received a command again in the session.
        """
        for name, angle in current.items():
            if isinstance(angle, (int, float)):
                self.sent.setdefault(name, angle)
        cmd = {}
        for name, target in smoothed.items():
            if name not in self.sent:
                continue  # without knowing where the servo is, send nothing
            ref = self.sent[name]
            delta = target - ref
            if abs(delta) < self.deadband:
                continue
            delta = max(-self.max_step, min(self.max_step, delta))
            cmd[name] = int(round(ref + delta))
        self.sent.update(cmd)
        return cmd

    def reset(self):
        self.filters.clear()
        self.sent.clear()
