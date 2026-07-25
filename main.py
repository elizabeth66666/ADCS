#!/usr/bin/env python3
"""main.py - Combined entry point for the rover: wires up electrical health
monitoring (health_monitor.py), wheel-speed sensing from the drivetrain
(drivetrain.py), and the camera/QR streamer (qr_code_streamer.py) so they
all run at once, and serves the result as a single web dashboard that
another device on the network can open in a browser.

Before running this file
-------------------------
Run camera_calibration.py once first to produce camera_calib.npz, which
QRCodeStreamer needs at startup. This file checks for it and prints a
helpful message instead of a raw traceback if it's missing.

What runs, and how
--------------------
- HealthMonitor (CPU temp, thermocouple, battery) and Drivetrain's
  WheelSpeedMonitor run as background daemon threads, continuously updating
  their readings.
- QRCodeStreamer serves the video feed plus a /status endpoint reporting
  that telemetry as JSON; its dashboard page - at
  http://<this device's IP>:5000/ - polls /status once a second and shows
  it next to the live camera stream with QR pose overlaid.
- Motor control is keyboard-driven from whatever device has the dashboard
  page open, not from this process's own terminal: the page listens for
  keydown events (w/up faster, s/down slower, f forward, r reverse, space
  stop) and POSTs them to /control, which calls
  Drivetrain.keyboard.handle_key() - no on-screen buttons to click. Run
  `python3 drivetrain.py` directly instead if you want local terminal
  keyboard control (SSH'd into this device) rather than browser-based.

Ctrl+C stops every subsystem and releases the camera/GPIO/SPI/I2C handles.
"""

import os

from drivetrain import Drivetrain
from health_monitor import HealthMonitor
from qr_code_streamer import QRCodeStreamer

CALIB_FILE = "camera_calib.npz"


def _status_payload(health, drivetrain):
    return {
        "cpu_temp_c": health.cpu_temp.temp_c,
        "thermocouple": {
            "temp_c": health.thermocouple.temp_c,
            "temp_f": health.thermocouple.temp_f,
            "is_estimated": health.thermocouple.temp_is_estimated,
        },
        "battery": {
            "voltage_v": health.battery.voltage,
            "current_a": health.battery.current_a,
            "current_is_estimated": health.battery.current_is_estimated,
            "capacity_percent": health.battery.capacity_percent,
            "low_voltage": health.battery.low_voltage,
        },
        "wheel_speed": {
            "rpm": drivetrain.speed_monitor.rpm,
            "speed_ms": drivetrain.speed_monitor.speed_ms,
            "direction": drivetrain.speed_monitor.direction,
        },
    }


def main():
    if not os.path.exists(CALIB_FILE):
        print(
            f"{CALIB_FILE} not found - run camera_calibration.py first to "
            "calibrate the camera before starting main.py."
        )
        return

    health = HealthMonitor()
    drivetrain = Drivetrain()
    streamer = QRCodeStreamer(
        status_provider=lambda: _status_payload(health, drivetrain),
        control_handler=drivetrain.keyboard.handle_key,
    )

    health.start()
    drivetrain.start()  # background wheel-speed monitoring; motor is driven via /control instead

    print(f"Dashboard starting at http://<this device's IP>:{streamer.port}/")
    try:
        streamer.run()  # blocks, serves the dashboard until Ctrl+C
    finally:
        drivetrain.stop()
        health.stop()
        print("Electrical, ADCS, and camera subsystems shut down.")


if __name__ == "__main__":
    main()
