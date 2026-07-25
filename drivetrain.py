#!/usr/bin/env python3
"""drivetrain.py - Combines wheel-speed sensing and L298N motor control so both
run at the same time on a Raspberry Pi 4, built on top of wheel_speed_sensor.py,
motor_control.py and l298n.py.

Wiring (BCM numbering)
-----------------------
Motor - L298N:
    IN1 -> GPIO23, IN2 -> GPIO24, ENA -> GPIO18

Wheel speed - two KY-003 Hall sensors, offset ~90 degrees around the wheel:
    Sensor A SIG -> GPIO17 (pin 11)
    Sensor B SIG -> GPIO27 (pin 13)

Note: the original wheel_speed_sensor.py used GPIO23/24 for the sensors,
which collides with the motor's IN1/IN2 pins above. Wire the sensors to
GPIO17/27 (as this file's defaults expect) so the two subsystems can run
simultaneously without fighting over the same pins.

Design
------
WheelSpeedMonitor runs as a background daemon thread, polling both sensors
and updating .rpm / .speed_ms / .direction under a lock - no printing, so
callers (main.py, or the keyboard controller below) decide what to do with
the numbers.

KeyboardMotorController owns the same w/s/f/r/space/q terminal control loop
from motor_control.py, and prints the live speed reading after each
keypress if a WheelSpeedMonitor is attached.

Drivetrain wires the two together and exposes start()/run_keyboard_control()/
stop() so main.py only needs a few lines.
"""

import os
import select
import sys
import termios
import threading
import time
import tty

import RPi.GPIO as GPIO

from l298n import L298N

# ---- Wheel speed sensing ---------------------------------------------------


class _SensorChannel:
    """Tracks one sensor pin and reports debounced falling edges via polling."""

    def __init__(self, pin, debounce_s):
        self.pin = pin
        self._debounce_s = debounce_s
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        self.last_state = GPIO.input(pin)
        self.last_edge_time = None

    def poll(self, now):
        state = GPIO.input(self.pin)
        edge = False
        if self.last_state == GPIO.HIGH and state == GPIO.LOW:
            if self.last_edge_time is None or (now - self.last_edge_time) >= self._debounce_s:
                self.last_edge_time = now
                edge = True
        self.last_state = state
        return edge


class WheelSpeedMonitor(threading.Thread):
    """Background thread measuring wheel RPM/speed/direction from two KY-003
    Hall sensors offset ~90 degrees apart. Read .rpm / .speed_ms / .direction
    from any thread; they update every `update_interval_s` seconds."""

    def __init__(
        self,
        sensor_a_pin=17,
        sensor_b_pin=27,
        magnets_per_rev=1,
        wheel_diameter_m=0.072,
        debounce_s=0.002,
        stale_timeout_s=2.0,
        update_interval_s=0.5,
        poll_sleep_s=0.0005,
    ):
        super().__init__(daemon=True)
        self.sensor_a_pin = sensor_a_pin
        self.sensor_b_pin = sensor_b_pin
        self._pulses_per_rev = magnets_per_rev * 2
        self._wheel_diameter_m = wheel_diameter_m
        self._debounce_s = debounce_s
        self._stale_timeout_s = stale_timeout_s
        self._update_interval_s = update_interval_s
        self._poll_sleep_s = poll_sleep_s

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._rpm = 0.0
        self._speed_ms = 0.0
        self._direction = "-"

        GPIO.setmode(GPIO.BCM)
        self._sensor_a = _SensorChannel(sensor_a_pin, debounce_s)
        self._sensor_b = _SensorChannel(sensor_b_pin, debounce_s)

    @property
    def rpm(self):
        with self._lock:
            return self._rpm

    @property
    def speed_ms(self):
        with self._lock:
            return self._speed_ms

    @property
    def direction(self):
        with self._lock:
            return self._direction

    def _linear_speed_ms(self, rpm):
        circumference_m = 3.14159265358979 * self._wheel_diameter_m
        return (rpm / 60.0) * circumference_m

    def run(self):
        channels = ((self.sensor_a_pin, self._sensor_a), (self.sensor_b_pin, self._sensor_b))
        pulse_count = 0
        last_edge_time = None
        last_pulse_dt = None
        last_edge_pin = None
        prev_edge_pin = None
        direction = "-"
        window_start = time.monotonic()

        while not self._stop_event.is_set():
            now = time.monotonic()

            for pin, sensor in channels:
                if not sensor.poll(now):
                    continue
                pulse_count += 1
                if last_edge_time is not None:
                    last_pulse_dt = now - last_edge_time
                last_edge_time = now
                prev_edge_pin, last_edge_pin = last_edge_pin, pin
                # Two sensors offset ~90 degrees: the order a single magnet
                # trips them in reveals rotation direction, like a quadrature
                # encoder.
                if prev_edge_pin == self.sensor_a_pin and last_edge_pin == self.sensor_b_pin:
                    direction = "forward"
                elif prev_edge_pin == self.sensor_b_pin and last_edge_pin == self.sensor_a_pin:
                    direction = "reverse"

            elapsed = now - window_start
            if elapsed >= self._update_interval_s:
                stale = last_edge_time is None or (now - last_edge_time) > self._stale_timeout_s
                if stale:
                    rpm = 0.0
                    direction = "-"
                else:
                    rpm = (pulse_count / self._pulses_per_rev) * (60.0 / elapsed)
                    if rpm == 0.0 and last_pulse_dt:
                        rpm = (1.0 / self._pulses_per_rev) / last_pulse_dt * 60.0

                with self._lock:
                    self._rpm = rpm
                    self._speed_ms = self._linear_speed_ms(rpm)
                    self._direction = direction

                pulse_count = 0
                window_start = now

            time.sleep(self._poll_sleep_s)

    def stop(self):
        self._stop_event.set()


# ---- Keyboard motor control -------------------------------------------------

# Escape-sequence tails for arrow keys, as sent by (almost) every terminal.
ARROW_KEYS = {"A": "up", "B": "down", "C": "right", "D": "left"}


def _read_one(fd, timeout):
    """Read a single character from fd via the raw file descriptor (bypassing any
    buffering), waiting up to `timeout` seconds. Returns None if nothing arrived."""
    if not select.select([fd], [], [], timeout)[0]:
        return None
    return os.read(fd, 1).decode(errors="ignore")


def read_key(fd, timeout=0.1):
    """Wait up to `timeout` seconds for a keypress on fd and return it.

    Returns a single character (e.g. "w", "q", " "), one of the ARROW_KEYS
    values ("up"/"down"/"left"/"right"), or None if nothing was pressed
    within the timeout.
    """
    ch = _read_one(fd, timeout)
    if ch is None or ch != "\x1b":  # not the start of an arrow-key escape sequence
        return ch

    # Arrow keys arrive as three bytes: ESC, '[', then a letter.
    if _read_one(fd, 0.01) == "[":
        return ARROW_KEYS.get(_read_one(fd, 0.01))
    return None


class KeyboardMotorController:
    """Drives an L298N motor from keypresses, exactly like motor_control.py:
    w/up = faster, s/down = slower, f = forward, r = reverse, space = stop,
    q = quit. If a WheelSpeedMonitor is attached, prints its live reading
    after every keypress.

    handle_key() is the shared dispatch used both by run_blocking() (local
    terminal, one thread) and by a remote controller such as
    qr_code_streamer.py's /control endpoint (HTTP requests, each handled on
    its own thread under Flask's threaded dev server) - a lock around the
    actual motor call keeps those two sources of keypresses from racing on
    the motor's direction/speed state.
    """

    SPEED_STEP = 10

    def __init__(self, motor, speed_monitor=None):
        self.motor = motor
        self.speed_monitor = speed_monitor
        self._lock = threading.Lock()
        self._actions = {
            "w": self._increase_speed,
            "up": self._increase_speed,
            "s": self._decrease_speed,
            "down": self._decrease_speed,
            "f": self._set_forward,
            "r": self._set_reverse,
            " ": self._stop,
        }

    def _increase_speed(self):
        self.motor.set_speed(self.motor.get_speed() + self.SPEED_STEP)
        print(f"Speed: {self.motor.get_speed()}%")

    def _decrease_speed(self):
        self.motor.set_speed(self.motor.get_speed() - self.SPEED_STEP)
        print(f"Speed: {self.motor.get_speed()}%")

    def _set_forward(self):
        self.motor.forward()
        print("Direction: forward")

    def _set_reverse(self):
        self.motor.backward()
        print("Direction: reverse")

    def _stop(self):
        self.motor.stop()
        self.motor.set_speed(0)
        print("Motor stopped")

    def _print_speed_reading(self):
        if self.speed_monitor is None:
            return
        print(
            f"  -> RPM: {self.speed_monitor.rpm:6.1f}  "
            f"Speed: {self.speed_monitor.speed_ms:5.2f} m/s  "
            f"Direction: {self.speed_monitor.direction}"
        )

    def handle_key(self, key):
        """Dispatch one key - "w"/"s"/"f"/"r"/" ", or an ARROW_KEYS value
        ("up"/"down") - to the matching motor action. Returns True if the
        key mapped to an action, False otherwise (e.g. "q" or an
        unrecognized key, which this method ignores; run_blocking() handles
        "q" itself since only a local terminal session has a loop to quit)."""
        action = self._actions.get(key)
        if action is None:
            return False
        with self._lock:
            action()
        self._print_speed_reading()
        return True

    def run_blocking(self):
        """Read keys from this terminal until 'q' is pressed. Blocks the calling
        thread and restores terminal settings before returning."""
        self.motor.forward()  # default starting direction, speed starts at 0 so the motor stays still
        print("Motor control ready.")
        print("w/up = faster, s/down = slower, f = forward, r = reverse, space = stop, q = quit")

        fd = sys.stdin.fileno()
        old_terminal_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # read keys one at a time, without waiting for Enter
            while True:
                key = read_key(fd)
                if key is None:
                    continue
                if key == "q":
                    break
                self.handle_key(key)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_terminal_settings)


# ---- Combined drivetrain -----------------------------------------------------


class Drivetrain:
    """Ties the motor and wheel-speed sensor together so they run at the same
    time. Import this into main.py instead of running motor_control.py and
    wheel_speed_sensor.py as two separate processes."""

    def __init__(
        self,
        motor_in1=23,
        motor_in2=24,
        motor_ena=18,
        sensor_a_pin=17,
        sensor_b_pin=27,
        **speed_monitor_kwargs,
    ):
        self.motor = L298N(pin_in1=motor_in1, pin_in2=motor_in2, pin_enable=motor_ena)
        self.speed_monitor = WheelSpeedMonitor(
            sensor_a_pin=sensor_a_pin, sensor_b_pin=sensor_b_pin, **speed_monitor_kwargs
        )
        self.keyboard = KeyboardMotorController(self.motor, self.speed_monitor)

    def start(self):
        """Start background wheel-speed monitoring. Call this before driving the
        motor (via run_keyboard_control() or self.motor directly) so speed
        readings are already live."""
        self.speed_monitor.start()

    def run_keyboard_control(self):
        """Blocking w/s/f/r/space/q keyboard control loop, with live speed
        readings printed after each keypress."""
        self.keyboard.run_blocking()

    def stop(self):
        """Stop the speed-monitor thread, stop the motor, and release GPIO."""
        self.speed_monitor.stop()
        self.speed_monitor.join(timeout=1.0)
        self.motor.cleanup()
        GPIO.cleanup()


def run():
    """Standalone entry point: start wheel-speed monitoring and keyboard motor
    control together, cleaning up GPIO on exit."""
    drivetrain = Drivetrain()
    drivetrain.start()
    try:
        drivetrain.run_keyboard_control()
    finally:
        drivetrain.stop()
        print("GPIO cleaned up, exiting.")


if __name__ == "__main__":
    run()
