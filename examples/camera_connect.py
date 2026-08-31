#!/usr/bin/env python

import argparse
import time
from pathlib import Path

import cv2

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig


def parse_index_or_path(value: str) -> int | Path:
    try:
        return int(value)
    except ValueError:
        return Path(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect and preview an OpenCV camera.")
    parser.add_argument("--index-or-path", default="0", help="Camera index or device path, e.g. 0 or /dev/video0.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="MJPG", help="Optional OpenCV FOURCC, e.g. MJPG or YUYV.")
    parser.add_argument("--display", action="store_true", help="Show frames with cv2.imshow if GUI is available.")
    parser.add_argument(
        "--save-frame",
        type=Path,
        default=Path("runs/inspection/camera_connect_opencv_capture.jpg"),
        help="Path to save the frame.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames. 0 means run forever.")
    args = parser.parse_args()

    config = OpenCVCameraConfig(
        index_or_path=parse_index_or_path(args.index_or_path),
        width=args.width,
        height=args.height,
        fps=args.fps,
        color_mode=ColorMode.BGR,
        fourcc=args.fourcc,
    )
    camera = OpenCVCamera(config)
    camera.connect()

    print("Camera connected. Press Ctrl+C or q to exit.")
    if not args.display:
        print("Display disabled. Use --display to preview frames if your OpenCV build supports GUI windows.")

    if args.save_frame is not None:
        args.save_frame.parent.mkdir(parents=True, exist_ok=True)

    display_enabled = args.display
    frame_count = 0
    start_time = time.perf_counter()
    try:
        while True:
            frame = camera.read()
            frame_count += 1

            if args.save_frame is not None:
                cv2.imwrite(str(args.save_frame), frame)

            if frame_count == 1 or frame_count % max(args.fps, 1) == 0:
                elapsed_s = max(time.perf_counter() - start_time, 1e-6)
                actual_fps = frame_count / elapsed_s
                print(f"Frame {frame_count}: shape={frame.shape}, dtype={frame.dtype}, fps={actual_fps:.2f}")

            if display_enabled:
                try:
                    cv2.imshow("Camera", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error as exc:
                    print(f"OpenCV display is unavailable, continuing without preview: {exc}")
                    display_enabled = False

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        camera.disconnect()
        if display_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
