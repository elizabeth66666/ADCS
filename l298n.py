"""
l298n.py - Raspberry Pi driver for the L298N dual H-Bridge motor driver module.

Python port of the Arduino "L298N" library by Andrea Lombardo
(http://www.lombardoandrea.com), rewritten to use RPi.GPIO instead of the
Arduino digitalWrite()/analogWrite() calls.

Wiring (BCM GPIO numbering):
    ENA (speed / PWM)       -> any GPIO that supports output (e.g. GPIO18)
    IN1 (direction control) -> any GPIO output pin (e.g. GPIO23)
    IN2 (direction control) -> any GPIO output pin (e.g. GPIO24)
    L298N GND               -> a Raspberry Pi GND pin
    L298N motor supply      -> a separate battery/power supply, NOT the Pi's 5V rail

If your L298N board has the ENA jumper installed (fixed full speed), wire only
IN1/IN2 and omit pin_enable when constructing L298N - speed control will be
unavailable, matching the original library's two-constructor design.
"""

import threading
import time
from enum import IntEnum

import RPi.GPIO as GPIO


class Direction(IntEnum):
    FORWARD = 0
    BACKWARD = 1
    STOP = -1


class L298N:
    def __init__(self, pin_in1, pin_in2, pin_enable=None, pwm_frequency=1000):
        """
        pin_in1, pin_in2: GPIO pins wired to L298N IN1/IN2 (direction control).
        pin_enable: GPIO pin wired to L298N ENA (speed control via PWM).
                    Leave as None if the board's speed jumper is installed,
                    in which case the motor always runs at full speed.
        pwm_frequency: PWM switching frequency in Hz, only used if pin_enable is set.
        """
        self._pin_in1 = pin_in1
        self._pin_in2 = pin_in2
        self._pin_enable = pin_enable
        self._speed = 0
        self._direction = Direction.STOP
        self._timer = None

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self._pin_in1, GPIO.OUT)
        GPIO.setup(self._pin_in2, GPIO.OUT)

        self._pwm = None
        if self._pin_enable is not None:
            GPIO.setup(self._pin_enable, GPIO.OUT)
            self._pwm = GPIO.PWM(self._pin_enable, pwm_frequency)
            self._pwm.start(0)

    def set_speed(self, percent):
        """Set motor speed as a percentage (0-100). Persists across stop()/forward()/
        backward() calls, and takes effect immediately if a direction is active."""
        percent = max(0, min(100, percent))
        self._speed = percent
        if self._pwm is not None and self._direction != Direction.STOP:
            self._pwm.ChangeDutyCycle(self._speed)

    def get_speed(self):
        """Return the currently set target speed (0-100), regardless of direction."""
        return self._speed

    def forward(self):
        GPIO.output(self._pin_in1, GPIO.HIGH)
        GPIO.output(self._pin_in2, GPIO.LOW)
        if self._pwm is not None:
            self._pwm.ChangeDutyCycle(self._speed)
        self._direction = Direction.FORWARD

    def backward(self):
        GPIO.output(self._pin_in1, GPIO.LOW)
        GPIO.output(self._pin_in2, GPIO.HIGH)
        if self._pwm is not None:
            self._pwm.ChangeDutyCycle(self._speed)
        self._direction = Direction.BACKWARD

    def stop(self):
        """Stop the motor. The target speed set via set_speed() is preserved, so a
        following forward()/backward() call resumes at the same speed."""
        self._cancel_timer()
        GPIO.output(self._pin_in1, GPIO.LOW)
        GPIO.output(self._pin_in2, GPIO.LOW)
        if self._pwm is not None:
            self._pwm.ChangeDutyCycle(0)
        self._direction = Direction.STOP

    def run(self, direction):
        if direction == Direction.FORWARD:
            self.forward()
        elif direction == Direction.BACKWARD:
            self.backward()
        else:
            self.stop()

    def run_for(self, duration, direction, callback=None):
        """Run in the given direction for `duration` seconds, then stop.

        Non-blocking: starts the motor now and schedules the stop (plus an
        optional callback) on a background timer, mirroring the original
        Arduino library's runFor()/forwardFor()/backwardFor() behavior.
        """
        self._cancel_timer()
        self.run(direction)

        def _finish():
            self.stop()
            if callback is not None:
                callback()

        self._timer = threading.Timer(duration, _finish)
        self._timer.daemon = True
        self._timer.start()

    def forward_for(self, duration, callback=None):
        self.run_for(duration, Direction.FORWARD, callback)

    def backward_for(self, duration, callback=None):
        self.run_for(duration, Direction.BACKWARD, callback)

    def is_moving(self):
        return self._direction != Direction.STOP and self._speed > 0

    def get_direction(self):
        return self._direction

    def _cancel_timer(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def cleanup(self):
        """Stop the motor and release the PWM channel. Does not call GPIO.cleanup()
        since other code/motors may still be using GPIO."""
        self.stop()
        if self._pwm is not None:
            self._pwm.stop()
            # Drop the reference now so the PWM object is destroyed here, before
            # GPIO.cleanup() runs. Otherwise, on the lgpio-backed RPi.GPIO used by
            # Raspberry Pi OS Bookworm, the object's __del__ can fire after
            # GPIO.cleanup() has already invalidated the GPIO chip handle, raising
            # a harmless but noisy "Exception ignored in: PWM.__del__" at exit.
            self._pwm = None


if __name__ == "__main__":
    # Quick hardware smoke test: forward, backward, then stop.
    motor = L298N(pin_in1=23, pin_in2=24, pin_enable=18)
    try:
        motor.set_speed(50)
        print("Forward for 2s...")
        motor.forward()
        time.sleep(2)

        print("Backward for 2s...")
        motor.backward()
        time.sleep(2)

        motor.stop()
        print("Stopped.")
    finally:
        motor.cleanup()
        GPIO.cleanup()
