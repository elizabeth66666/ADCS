#!/usr/bin/env python3
"""qr_code_streamer.py - MJPEG HTTP stream of the Pi Camera with live QR-code
detection and pose estimation overlaid, built on top of the standalone QR
distance-scanner script. Requires a camera_calib.npz produced by
camera_calibration.py.

Unlike the original script, importing this module does nothing by itself -
the camera only starts, calibration is only loaded, and the Flask server
only listens once run() (or QRCodeStreamer.run()) is called, so main.py can
import it alongside drivetrain.py / health_monitor.py / camera_calibration.py
without opening the camera or binding a port at import time.

Reports XYZ instead of a single distance
-----------------------------------------
cv2.solvePnP already gives the QR code's full 3D position (tvec), not just
its distance - the original script threw away X and Y by reducing tvec to
np.linalg.norm(tvec). This version reports all three: X (right/left of the
camera), Y (up/down), Z (straight-ahead depth), each in metres.

Mounting-tilt correction
-------------------------
If the camera is physically pitched off horizontal (mount_tilt_deg, default
6.4 degrees), the raw camera-frame XYZ is tilted with it: "straight ahead"
in camera coordinates is not the same as "straight ahead, level with the
horizon". _tilt_rotation_matrix() rotates the pose about the camera's X
(left-right) axis to undo that pitch, so the reported X/Y/Z are relative to
level ground/horizon rather than the tilted camera body.

Convention: a positive mount_tilt_deg means the camera is pitched downward
(nose tilted toward the ground) from horizontal. If your camera is tilted
upward instead, pass a negative angle. Verify the sign against your own
mount by pointing the camera at a QR code you know is above/below camera
height and checking that the reported Y has the sign you expect.
"""

import math
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from picamera2 import Picamera2


_STATUS_HEAD = """
<script>
    async function refreshStatus() {
        try {
            const res = await fetch("/status");
            const data = await res.json();
            document.getElementById("status").textContent =
                JSON.stringify(data, null, 2);
        } catch (err) {
            document.getElementById("status").textContent =
                "status unavailable: " + err;
        }
    }
    setInterval(refreshStatus, 1000);
    window.addEventListener("load", refreshStatus);
</script>
"""

_STATUS_BODY = """
<pre id="status">loading status...</pre>
"""

# Maps a browser KeyboardEvent.key to the key strings KeyboardMotorController
# understands (see drivetrain.py: "w"/"s"/"f"/"r"/" "/"up"/"down"). The
# heartbeat interval is filled in per-instance (see _index()) from
# control_timeout_s, so it stays comfortably below the server's watchdog
# timeout.
_CONTROL_HEAD_TEMPLATE = """
<script>
    const CONTROL_KEY_MAP = {
        "w": "w", "ArrowUp": "up",
        "s": "s", "ArrowDown": "down",
        "f": "f", "r": "r",
        " ": " ",
    };

    async function sendControlKey(key) {
        try {
            const res = await fetch("/control", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({key: key}),
            });
            const data = await res.json();
            document.getElementById("control-log").textContent =
                (data.handled ? "sent: " : "ignored: ") + key;
        } catch (err) {
            document.getElementById("control-log").textContent =
                "control error: " + err;
        }
    }

    window.addEventListener("keydown", (event) => {
        const key = CONTROL_KEY_MAP[event.key];
        if (key === undefined) return;
        event.preventDefault();  // stop space/arrows from scrolling the page
        sendControlKey(key);
    });

    // Keeps the server-side motor watchdog alive as long as this page stays
    // open and connected, independent of whether a movement key is
    // currently pressed - holding a steady speed shouldn't trip the
    // watchdog, only closing the tab or losing the connection should.
    setInterval(() => {
        fetch("/heartbeat", {method: "POST"}).catch(() => {});
    }, __HEARTBEAT_INTERVAL_MS__);
</script>
"""

_CONTROL_BODY = """
<p>Keyboard controls (click the page once so it has focus):
w/&uarr; faster, s/&darr; slower, f forward, r reverse, space stop</p>
<p><small>The motor stops automatically if this page disconnects.</small></p>
<div id="control-log">no commands sent yet</div>
"""


def _tilt_rotation_matrix(tilt_deg):
    """Rotation matrix that reprojects a camera-frame vector onto a level
    (horizontal) reference frame, undoing a camera mount pitched
    `tilt_deg` degrees downward from horizontal (see module docstring for
    the sign convention). OpenCV camera frame: +X right, +Y down, +Z
    forward out of the lens; the correction rotates about that shared X
    axis.
    """
    theta = math.radians(tilt_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_t, sin_t],
            [0.0, -sin_t, cos_t],
        ]
    )


class QRCodeStreamer:
    """Serves an MJPEG stream at /video (and a viewer page at /) from the Pi
    Camera, drawing the outline, decoded text, and estimated X/Y/Z position
    (in metres, corrected for camera mounting tilt) of any QR code it sees.

    Pass status_provider (a zero-argument callable returning a JSON-
    serializable dict) to also serve that data at /status and show it on
    the dashboard page next to the video - e.g. main.py wires this up to
    report health_monitor/drivetrain telemetry alongside the camera feed,
    without this module needing to import either of those.

    Pass control_handler (a callable taking one key string - "w"/"s"/"f"/
    "r"/" "/"up"/"down" - and returning whether it did anything) to also
    accept POST /control {"key": "..."} requests and add a keydown listener
    to the dashboard page, so a physical keyboard on whatever device is
    viewing the page drives the motor - no on-screen buttons to click.
    main.py wires this to a KeyboardMotorController.handle_key, again
    without this module needing to import drivetrain.py directly.

    Whenever control_handler is set, a background watchdog also stops the
    motor (by calling control_handler(" ")) if control_timeout_s seconds
    pass without any /control or /heartbeat request - the dashboard page
    sends a heartbeat on an interval independent of actual keypresses, so
    the motor keeps running at a steady commanded speed while the page is
    just sitting there, but stops automatically if the tab closes or the
    connection drops.
    """

    def __init__(
        self,
        calib_file="camera_calib.npz",
        qr_size_m=0.05,
        capture_size=(1280, 720),
        warmup_s=2.0,
        host="0.0.0.0",
        port=5000,
        mount_tilt_deg=6.4,
        status_provider=None,
        control_handler=None,
        control_timeout_s=2.0,
    ):
        self.qr_size_m = qr_size_m
        self.capture_size = capture_size
        self.warmup_s = warmup_s
        self.host = host
        self.port = port
        self.mount_tilt_deg = mount_tilt_deg
        self._status_provider = status_provider
        self._control_handler = control_handler
        self._control_timeout_s = control_timeout_s
        self._control_lock = threading.Lock()
        self._last_control_seen = None

        data = np.load(calib_file)
        self.camera_matrix = data["camera_matrix"]
        self.dist_coeffs = data["dist_coeffs"]

        self._object_points = np.array(
            [
                [0, 0, 0],
                [qr_size_m, 0, 0],
                [qr_size_m, qr_size_m, 0],
                [0, qr_size_m, 0],
            ],
            dtype=np.float32,
        )
        self._tilt_rotation = _tilt_rotation_matrix(mount_tilt_deg)

        self._picam2 = None
        self._qr_detector = None
        self._new_camera_matrix = None

        self.app = Flask(__name__)
        self.app.add_url_rule("/", "index", self._index)
        self.app.add_url_rule("/video", "video", self._video)
        if self._status_provider is not None:
            self.app.add_url_rule("/status", "status", self._status)
        if self._control_handler is not None:
            self.app.add_url_rule("/control", "control", self._control, methods=["POST"])
            self.app.add_url_rule("/heartbeat", "heartbeat", self._heartbeat, methods=["POST"])
            self._start_control_watchdog()

    def start_camera(self):
        """Open the Pi Camera and compute the undistortion map. Safe to call
        more than once - a second call is a no-op if already running."""
        if self._picam2 is not None:
            return

        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": self.capture_size})
        picam2.configure(config)
        picam2.start()

        time.sleep(self.warmup_s)  # allow camera to warm up

        first_frame = picam2.capture_array()
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
        h, w = first_frame.shape[:2]

        new_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )

        self._picam2 = picam2
        self._new_camera_matrix = new_camera_matrix
        self._qr_detector = cv2.QRCodeDetector()

    def _start_control_watchdog(self):
        """Background daemon thread: stop the motor if no /control or
        /heartbeat request has arrived in control_timeout_s seconds."""

        def _watch():
            while True:
                time.sleep(self._control_timeout_s / 2)
                with self._control_lock:
                    last_seen = self._last_control_seen
                if last_seen is None:
                    continue  # no client has connected yet
                if time.monotonic() - last_seen > self._control_timeout_s:
                    print(
                        f"WARNING: no /control or /heartbeat received in "
                        f"{self._control_timeout_s:.1f}s; stopping motor for safety"
                    )
                    self._control_handler(" ")
                    with self._control_lock:
                        self._last_control_seen = None  # don't re-stop every cycle while disconnected

        threading.Thread(target=_watch, daemon=True).start()

    def _touch_control(self):
        with self._control_lock:
            self._last_control_seen = time.monotonic()

    def stop_camera(self):
        if self._picam2 is not None:
            self._picam2.stop()
            self._picam2 = None
            self._qr_detector = None
            self._new_camera_matrix = None

    def _estimate_pose_m(self, image_points):
        """Run solvePnP for one QR code's corners and return the tilt-
        corrected (x, y, z) in metres, or None if the pose couldn't be
        solved.

        Uses new_camera_matrix with zero distortion, matching the frame
        these image_points were detected on: it has already been undistorted
        with cv2.undistort(), so re-applying dist_coeffs here would double
        up the distortion correction and skew the estimated pose.
        """
        success, _rvec, tvec = cv2.solvePnP(
            self._object_points,
            image_points,
            self._new_camera_matrix,
            None,
        )
        if not success:
            return None
        x, y, z = (self._tilt_rotation @ tvec.reshape(3)).tolist()
        return x, y, z

    def _generate_frames(self):
        self.start_camera()
        while True:
            frame = self._picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.undistort(
                frame, self.camera_matrix, self.dist_coeffs, None, self._new_camera_matrix
            )

            retval, decoded_info, points, _ = self._qr_detector.detectAndDecodeMulti(frame)

            if retval:
                for qr_data, point in zip(decoded_info, points):
                    if not qr_data:
                        continue

                    image_points = np.array(point, dtype=np.float32)
                    pose = self._estimate_pose_m(image_points)
                    if pose is None:
                        continue
                    x, y, z = pose

                    pts = image_points.astype(int)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

                    center = pts.mean(axis=0).astype(int)
                    text = f"{qr_data} | X:{x:.2f} Y:{y:.2f} Z:{z:.2f} m"
                    cv2.putText(
                        frame,
                        text,
                        (center[0], center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

    def _index(self):
        head = ""
        body = ""

        if self._status_provider is not None:
            head += _STATUS_HEAD
            body += _STATUS_BODY
        if self._control_handler is not None:
            heartbeat_interval_ms = int(max(self._control_timeout_s * 1000 / 3, 100))
            head += _CONTROL_HEAD_TEMPLATE.replace(
                "__HEARTBEAT_INTERVAL_MS__", str(heartbeat_interval_ms)
            )
            body += _CONTROL_BODY

        title = "Rover Dashboard" if head else "QR Distance Scanner"

        return f"""
        <html>
            <head>
                <title>{title}</title>
                {head}
            </head>
            <body>
                <h1>{title}</h1>
                <img src="/video">
                {body}
            </body>
        </html>
        """

    def _status(self):
        return jsonify(self._status_provider())

    def _control(self):
        self._touch_control()
        payload = request.get_json(silent=True) or {}
        key = payload.get("key")
        handled = bool(key) and self._control_handler(key)
        return jsonify({"handled": handled})

    def _heartbeat(self):
        self._touch_control()
        return jsonify({"ok": True})

    def _video(self):
        return Response(
            self._generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def run(self, threaded=True):
        """Start the camera and block serving the Flask app, same as running
        the original script directly."""
        self.start_camera()
        try:
            self.app.run(host=self.host, port=self.port, threaded=threaded)
        finally:
            self.stop_camera()


def run():
    """Standalone entry point: same behavior as the original script."""
    QRCodeStreamer().run()


if __name__ == "__main__":
    run()
