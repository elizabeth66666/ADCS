#!/usr/bin/env python3
"""Poll a single KY-003 Hall sensor and report magnet presence.

Direct Python port of the Arduino "Detect a Magnet" sketch. GPIO.input()
reads the pin's memory-mapped register directly - the same mechanism as
Arduino's digitalRead() - with no interrupts or edge-detection callbacks
involved, so it works even where those subsystems fail to deliver events.

Standalone diagnostic only - run this by itself, before drivetrain.py, to
confirm a single sensor's wiring works. It is not part of the combined
drivetrain module since it exercises one sensor pin at a time.
"""

import time

import RPi.GPIO as GPIO

SENSOR_PIN = 17  # change to test the other sensor (e.g. 27)


def main():
    GPIO.setmode(GPIO.BCM)
    # KY-003 output is open-collector; PUD_UP gives a clean HIGH at rest.
    GPIO.setup(SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    print(f"Polling GPIO{SENSOR_PIN} (BCM). Press Ctrl+C to stop.")
    try:
        while True:
            state = GPIO.input(SENSOR_PIN)
            print("MAGNET DETECTED" if state == GPIO.LOW else "no magnet")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    main()
