#!/usr/bin/env python3
"""health_monitor.py - Combines CPU temperature, thermocouple temperature, and
battery (INA219 current/voltage) monitoring so all three run at the same time
on a Raspberry Pi, built on top of the standalone CPU-temp / thermocouple /
battery-monitor scripts.

Design
------
Each sensor gets its own background daemon thread that polls at its own
interval and stores the latest reading under a lock - no printing, so
callers (main.py, or print_status() below) decide what to do with the
numbers.

HealthMonitor wires the three together and exposes start()/stop()/
print_status() so main.py only needs a few lines, the same shape as
Drivetrain in drivetrain.py.

Dependencies (not in the standard library): the `ina219` package for the
INA219 current sensor, and a `max31855` module exposing AdafruitMAX31855
for the thermocouple amplifier. `vcgencmd` (CPU temp) ships with Raspberry
Pi OS.
"""

import math
import os
import re
import subprocess
import threading
import time

from ina219 import INA219
from max31855 import AdafruitMAX31855

# ---- CPU temperature --------------------------------------------------------


def read_cpu_temp():
    """Read the Pi's own CPU temperature via `vcgencmd measure_temp`.

    Returns (temp_c, raw_message). temp_c is None if vcgencmd failed or its
    output didn't contain a parseable number.
    """
    temp = None
    err, msg = subprocess.getstatusoutput("vcgencmd measure_temp")
    if not err:
        m = re.search(r"-?\d\.?\d*", msg)
        if m is not None:
            try:
                temp = float(m.group())
            except ValueError:
                pass
    return temp, msg


class CPUTempMonitor(threading.Thread):
    """Background thread polling the Raspberry Pi's own CPU temperature."""

    def __init__(self, update_interval_s=2.0):
        super().__init__(daemon=True)
        self._update_interval_s = update_interval_s
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._temp_c = None
        self._raw_message = None

    @property
    def temp_c(self):
        with self._lock:
            return self._temp_c

    @property
    def raw_message(self):
        with self._lock:
            return self._raw_message

    def run(self):
        while not self._stop_event.is_set():
            temp, msg = read_cpu_temp()
            with self._lock:
                self._temp_c = temp
                self._raw_message = msg
            self._stop_event.wait(self._update_interval_s)

    def stop(self):
        self._stop_event.set()


# ---- Thermocouple (MAX31855) ------------------------------------------------


class ThermocoupleMonitor(threading.Thread):
    """Background thread polling a MAX31855 thermocouple amplifier over SPI.

    If a reading is NaN (open circuit / short to VCC / short to GND), .ok
    becomes False and .fault_code holds the MAX31855 error byte instead of
    raising - callers decide how to react (e.g. treat as a sensor fault
    rather than crashing the whole health monitor).
    """

    def __init__(self, bus=0, device=0, update_interval_s=1.0):
        super().__init__(daemon=True)
        self._update_interval_s = update_interval_s
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._temp_c = None
        self._temp_f = None
        self._internal_temp_c = None
        self._fault_code = None

        self._sensor = AdafruitMAX31855(bus=bus, device=device)
        if not self._sensor.begin():
            raise RuntimeError("Could not initialize thermocouple SPI")

    @property
    def temp_c(self):
        with self._lock:
            return self._temp_c

    @property
    def temp_f(self):
        with self._lock:
            return self._temp_f

    @property
    def internal_temp_c(self):
        with self._lock:
            return self._internal_temp_c

    @property
    def fault_code(self):
        with self._lock:
            return self._fault_code

    @property
    def ok(self):
        with self._lock:
            return self._fault_code is None

    def run(self):
        while not self._stop_event.is_set():
            temp_c = self._sensor.read_celsius()
            internal_temp_c = self._sensor.read_internal()
            fault_code = None
            temp_f = None
            if math.isnan(temp_c):
                fault_code = self._sensor.read_error()
            else:
                temp_f = self._sensor.read_fahrenheit()

            with self._lock:
                self._temp_c = temp_c
                self._temp_f = temp_f
                self._internal_temp_c = internal_temp_c
                self._fault_code = fault_code

            self._stop_event.wait(self._update_interval_s)

    def stop(self):
        self._stop_event.set()

    def close(self):
        """Release the SPI handle. Call after stop()+join(), not before -
        the poll loop needs the sensor open for its whole lifetime."""
        self._sensor.close()


# ---- Battery / INA219 --------------------------------------------------------


class BatteryMonitor(threading.Thread):
    """Background thread polling an INA219 current/voltage sensor to track
    battery state of health: bus voltage plus coulomb-counted remaining
    capacity, persisted to `state_file` so it survives restarts.

    Unlike the original standalone script, a battery already at/below
    BATTERY_MINIMUM_VOLTAGE at startup does not exit the process - it sets
    .low_voltage instead, since one drained battery shouldn't take down
    the whole health monitor. main.py should check .low_voltage itself.
    """

    def __init__(
        self,
        capacity_ah=3.0,
        full_voltage=7.4,
        minimum_voltage=6.4,
        i2c_address=0x40,
        i2c_bus=1,
        state_file="battery_state.txt",
        update_interval_s=1.0,
    ):
        super().__init__(daemon=True)
        self._capacity_ah = capacity_ah
        self._full_voltage = full_voltage
        self._minimum_voltage = minimum_voltage
        self._state_file = state_file
        self._update_interval_s = update_interval_s

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._voltage = None
        self._current_a = None
        self._capacity_percent = None
        self._voltage_percent = None
        self._low_voltage = False

        self._ina219 = INA219(address=i2c_address, bus_number=i2c_bus)
        if not self._ina219.begin():
            raise RuntimeError("INA219 not found")
        self._ina219.set_calibration_32V_2A()

        self._remaining_capacity_ah = self._load_initial_capacity()

        starting_voltage = self._ina219.get_bus_voltage_V()
        if starting_voltage <= self._minimum_voltage:
            self._low_voltage = True
            print(
                "WARNING: battery voltage is already at or below the minimum "
                f"({starting_voltage:.2f} V <= {self._minimum_voltage:.2f} V); "
                "battery should be charged before use."
            )

    def _voltage_to_percent(self, voltage):
        percent = (
            (voltage - self._minimum_voltage) / (self._full_voltage - self._minimum_voltage)
        ) * 100
        return max(0, min(100, percent))

    def _load_initial_capacity(self):
        if os.path.exists(self._state_file):
            with open(self._state_file, "r") as file:
                return float(file.read())
        starting_voltage = self._ina219.get_bus_voltage_V()
        starting_percent = self._voltage_to_percent(starting_voltage)
        return (starting_percent / 100) * self._capacity_ah

    @property
    def voltage(self):
        with self._lock:
            return self._voltage

    @property
    def current_a(self):
        with self._lock:
            return self._current_a

    @property
    def remaining_capacity_ah(self):
        with self._lock:
            return self._remaining_capacity_ah

    @property
    def capacity_percent(self):
        with self._lock:
            return self._capacity_percent

    @property
    def voltage_percent(self):
        with self._lock:
            return self._voltage_percent

    @property
    def low_voltage(self):
        with self._lock:
            return self._low_voltage

    def run(self):
        last_time = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            elapsed_hours = (now - last_time) / 3600
            last_time = now

            current_a = self._ina219.get_current_mA() / 1000
            voltage = self._ina219.get_bus_voltage_V()
            remaining_capacity_ah = max(
                0, self._remaining_capacity_ah - current_a * elapsed_hours
            )

            with self._lock:
                self._remaining_capacity_ah = remaining_capacity_ah
                self._current_a = current_a
                self._voltage = voltage
                self._capacity_percent = max(
                    0, min(100, (remaining_capacity_ah / self._capacity_ah) * 100)
                )
                self._voltage_percent = self._voltage_to_percent(voltage)
                self._low_voltage = voltage <= self._minimum_voltage

            with open(self._state_file, "w") as file:
                file.write(str(remaining_capacity_ah))

            self._stop_event.wait(self._update_interval_s)

    def stop(self):
        self._stop_event.set()


# ---- Combined health monitor -------------------------------------------------


class HealthMonitor:
    """Ties CPU temp, thermocouple, and battery monitoring together so they
    run at the same time. Import this into main.py alongside drivetrain.py."""

    def __init__(self, cpu_temp_kwargs=None, thermocouple_kwargs=None, battery_kwargs=None):
        self.cpu_temp = CPUTempMonitor(**(cpu_temp_kwargs or {}))
        self.thermocouple = ThermocoupleMonitor(**(thermocouple_kwargs or {}))
        self.battery = BatteryMonitor(**(battery_kwargs or {}))
        self._threads = (self.cpu_temp, self.thermocouple, self.battery)

    def start(self):
        for thread in self._threads:
            thread.start()

    def stop(self):
        for thread in self._threads:
            thread.stop()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self.thermocouple.close()

    def print_status(self):
        print("==============================")

        cpu_temp = self.cpu_temp.temp_c
        print(f"CPU Temp: {cpu_temp:.1f}°C" if cpu_temp is not None else "CPU Temp: unavailable")

        if self.thermocouple.ok:
            print(
                f"Thermocouple: {self.thermocouple.temp_c:.2f}°C "
                f"({self.thermocouple.temp_f:.2f}°F)"
            )
        else:
            print(f"Thermocouple fault: 0x{self.thermocouple.fault_code:02X}")
        internal_temp = self.thermocouple.internal_temp_c
        if internal_temp is not None:
            print(f"Thermocouple internal temp: {internal_temp:.2f}°C")

        voltage = self.battery.voltage
        if voltage is not None:
            print(
                f"Battery: {voltage:.2f} V, {self.battery.current_a:.3f} A, "
                f"{self.battery.capacity_percent:.1f}% capacity"
            )
        if self.battery.low_voltage:
            print("CRITICAL WARNING: battery at/below minimum voltage.")

        print("==============================")


def run():
    """Standalone entry point: start all three monitors and print status on
    an interval until Ctrl+C."""
    monitor = HealthMonitor()
    monitor.start()
    try:
        while True:
            monitor.print_status()
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        print("Health monitor stopped, exiting.")


if __name__ == "__main__":
    run()
