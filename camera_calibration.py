#!/usr/bin/env python3
"""camera_calibration.py - Interactive chessboard camera calibration for the
Raspberry Pi Camera via Picamera2/OpenCV, built on top of the standalone
calibration script.

Unlike the original script, importing this module does nothing by itself -
opening the camera, showing the capture window, and computing calibration
only happen when CameraCalibrator.calibrate() (or run()) is called. That
makes it safe to import alongside drivetrain.py / health_monitor.py from a
combined main.py without popping up a capture window or blocking on
keypresses at import time.
"""

import time

import cv2
import numpy as np
from picamera2 import Picamera2


class CameraCalibrator:
    """Interactive chessboard calibration: capture frames with SPACE, finish
    with Q, then compute camera_matrix/dist_coeffs and save them to
    `output_file` (an .npz, loadable with CameraCalibrator.load())."""

    def __init__(
        self,
        chessboard_size=(7, 7),
        capture_size=(1280, 720),
        settle_time_s=2.0,
        min_images=5,
        output_file="camera_calib.npz",
        save_captured_images=True,
    ):
        self.chessboard_size = chessboard_size
        self.capture_size = capture_size
        self.settle_time_s = settle_time_s
        self.min_images = min_images
        self.output_file = output_file
        self.save_captured_images = save_captured_images

        self._objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self._objp[:, :2] = np.mgrid[
            0 : chessboard_size[0], 0 : chessboard_size[1]
        ].T.reshape(-1, 2)

        self.camera_matrix = None
        self.dist_coeffs = None
        self.rms_error = None

    def calibrate(self, window_name="Calibration"):
        """Run the interactive capture loop, then compute and save calibration.

        Returns (camera_matrix, dist_coeffs, rms_error), or None if the user
        quit (pressed Q) before capturing at least `min_images` images.
        """
        objpoints = []
        imgpoints = []
        gray_shape = None

        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": self.capture_size})
        picam2.configure(config)
        picam2.start()
        time.sleep(self.settle_time_s)  # let auto-exposure/white-balance settle

        print("Press SPACE to capture calibration image")
        print("Press Q to finish calibration")

        img_count = 0
        try:
            while True:
                frame = picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Picamera2 returns RGB
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                gray_shape = gray.shape[::-1]

                found, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

                display = frame_bgr.copy()
                if found:
                    cv2.drawChessboardCorners(display, self.chessboard_size, corners, found)

                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(" "):
                    if found:
                        objpoints.append(self._objp)
                        imgpoints.append(corners)
                        if self.save_captured_images:
                            cv2.imwrite(f"calibration_{img_count}.jpg", frame_bgr)
                        img_count += 1
                        print(f"Captured calibration image {img_count}")
                    else:
                        print("Chessboard not detected")
                elif key == ord("q"):
                    break
        finally:
            picam2.stop()
            cv2.destroyAllWindows()

        if len(objpoints) < self.min_images:
            print(f"Not enough calibration images ({len(objpoints)} < {self.min_images}).")
            return None

        print("Calculating calibration...")
        rms_error, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray_shape, None, None
        )

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rms_error = rms_error

        np.savez(self.output_file, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        print(f"Calibration saved to {self.output_file}")
        print("\nCamera Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)
        print("RMS:", rms_error)

        return camera_matrix, dist_coeffs, rms_error

    @staticmethod
    def load(calib_file="camera_calib.npz"):
        """Load a previously saved camera_matrix/dist_coeffs pair."""
        data = np.load(calib_file)
        return data["camera_matrix"], data["dist_coeffs"]


def run():
    """Standalone entry point: same interactive flow as the original script."""
    calibrator = CameraCalibrator()
    calibrator.calibrate()


if __name__ == "__main__":
    run()
