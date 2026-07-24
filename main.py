#!/usr/bin/env python3
"""Entry point: runs the drivetrain (L298N motor control + dual-Hall-sensor
wheel-speed monitoring) at the same time. See drivetrain.py for the pieces
(Drivetrain, WheelSpeedMonitor, KeyboardMotorController) if you want finer
control than the default keyboard-driven run() below - e.g. driving the
motor programmatically while still reading drivetrain.speed_monitor.rpm.
"""

from drivetrain import run

if __name__ == "__main__":
    run()
