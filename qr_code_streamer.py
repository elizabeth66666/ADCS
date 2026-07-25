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
import time

import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2


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
    (in metres, corrected for camera mounting tilt) of any QR code it sees."""

    def __init__(
        self,
        calib_file="camera_calib.npz",
        qr_size_m=0.05,
        capture_size=(1280, 720),
        warmup_s=2.0,
        host="0.0.0.0",
        port=5000,
        mount_tilt_deg=6.4,
    ):
        self.qr_size_m = qr_size_m
        self.capture_size = capture_size
        self.warmup_s = warmup_s
        self.host = host
        self.port = port
        self.mount_tilt_deg = mount_tilt_deg

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
        return """
        <html>
            <body>
                <h1>QR Distance Scanner</h1>
                <img src="/video">
            </body>
        </html>
        """

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
