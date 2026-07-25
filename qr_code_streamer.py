#!/usr/bin/env python3
"""qr_code_streamer.py - MJPEG HTTP stream of the Pi Camera with live QR-code
detection overlaid, built on top of the standalone Flask streaming script.

Unlike the original script, importing this module does nothing by itself -
the camera only starts and the Flask server only listens once run() (or
QRCodeStreamer.run()) is called, so main.py can import it alongside
drivetrain.py / health_monitor.py / camera_calibration.py without opening
the camera or binding a port at import time.
"""

import time

import cv2
from flask import Flask, Response
from picamera2 import Picamera2


class QRCodeStreamer:
    """Serves an MJPEG stream at /video (and a viewer page at /) from the Pi
    Camera, drawing the outline and decoded text of any QR code it sees."""

    def __init__(self, capture_size=(640, 480), warmup_s=2.0, host="0.0.0.0", port=8000):
        self.capture_size = capture_size
        self.warmup_s = warmup_s
        self.host = host
        self.port = port

        self._picam2 = None
        self._detector = None

        self.app = Flask(__name__)
        self.app.add_url_rule("/", "index", self._index)
        self.app.add_url_rule("/video", "video", self._video)

    def start_camera(self):
        """Open and configure the Pi Camera. Safe to call more than once -
        a second call is a no-op if the camera is already running."""
        if self._picam2 is not None:
            return

        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": self.capture_size, "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        print(picam2.camera_configuration())

        time.sleep(self.warmup_s)  # allow camera to warm up

        self._picam2 = picam2
        self._detector = cv2.QRCodeDetector()

    def stop_camera(self):
        if self._picam2 is not None:
            self._picam2.stop()
            self._picam2 = None
            self._detector = None

    def _generate_frames(self):
        self.start_camera()
        while True:
            frame = self._picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            data, bbox, _ = self._detector.detectAndDecode(frame)

            if bbox is not None and len(bbox) > 0:
                bbox = bbox.astype(int)

                for i in range(len(bbox[0])):
                    pt1 = tuple(bbox[0][i])
                    pt2 = tuple(bbox[0][(i + 1) % len(bbox[0])])
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                if data:
                    cv2.putText(
                        frame,
                        f"QR: {data}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
                    print("Detected QR:", data)

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
            <head>
                <title>QR Camera Stream</title>
            </head>
            <body>
                <h1>Live QR Camera Stream</h1>
                <img src="/video" width="640" height="480">
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
