#!/usr/bin/env python3
"""
yolo_usb_cam.py
USB camera realtime detection with Ultralytics YOLO + OpenCV.
- Works with Logitech/V4L2 cameras
- Auto-uses CUDA if available (for YOLO). OpenCV CUDA not required.
- Adjustable camera index, resolution, confidence, and model path.

Usage examples:
  python yolo_usb_cam.py                       # default /dev/video0, 1280x720, yolov8n/yolov11n if present
  python yolo_usb_cam.py --camera 1 --width 1920 --height 1080 --conf 0.35 --model ~/weights/yolov11n.pt
  python yolo_usb_cam.py --save out.mp4        # save annotated video
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# Ultralytics (supports YOLOv8/YOLOv11 unified API)
from ultralytics import YOLO

try:
    import torch
except ImportError:
    torch = None


def parse_args():
    p = argparse.ArgumentParser(description="USB camera YOLO inference")
    p.add_argument("--camera", type=int, default=0, help="V4L2 device index (e.g., 0 for /dev/video0)")
    p.add_argument("--width", type=int, default=1280, help="Capture width")
    p.add_argument("--height", type=int, default=720, help="Capture height")
    p.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--model", type=str, default="", help="Path to a .pt model (yolov8/11). If empty, tries yolov11n then yolov8n.")
    p.add_argument("--save", type=str, default="", help="Optional output MP4 filename to save annotated stream")
    p.add_argument("--classes", type=int, nargs="*", default=None, help="Restrict classes (e.g., --classes 0 1 for person,bicycle)")
    p.add_argument("--show-fps", action="store_true", help="Overlay FPS and backend info")
    return p.parse_args()


def pick_default_model():
    # Prefer small, common weights if user didn't specify
    candidates = [
        Path.home() / "weights" / "yolov11n.pt",
        Path.home() / "yolov11n.pt",
        Path.home() / "weights" / "yolov8n.pt",
        Path.home() / "yolov8n.pt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Fall back to pulling by name (Ultralytics will download if allowed)
    return "yolov8n.pt"


def open_camera(index: int, w: int, h: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    # Ask the camera for MJPG to get higher FPS at 1080p on Logitech cams
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    # Try to stabilize exposure/gain for less flicker (best-effort; some cams ignore)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode on many Logitech drivers
    # If manual exposure unsupported, this will just be ignored.
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    return cap


def colors_for_classes(n):
    rng = np.random.default_rng(42)
    return (rng.random((n, 3)) * 200 + 55).astype(np.uint8)  # bright-ish colors


def main():
    args = parse_args()

    model_path = args.model if args.model else pick_default_model()

    # Device selection for YOLO
    if torch is not None and torch.cuda.is_available():
        device = 0  # first CUDA device
        device_str = "cuda:0"
    else:
        device = "cpu"
        device_str = "cpu"

    # Load YOLO model
    model = YOLO(model_path)

    # Warmup (helps stabilize first-frame latency on CUDA)
    try:
        model.predict(np.zeros((args.height, args.width, 3), dtype=np.uint8),
                      imgsz=max(args.width, args.height),
                      device=device,
                      conf=args.conf,
                      iou=args.iou,
                      classes=args.classes,
                      verbose=False)
    except Exception:
        # If warmup fails (e.g., empty input path), ignore; runtime will still work.
        pass

    cap = open_camera(args.camera, args.width, args.height)

    # Prepare writer if saving
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 30.0, (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open writer for {args.save}")

    # Class names and colors
    names = model.names
    palette = colors_for_classes(len(names))

    last_t = time.time()
    fps = 0.0

    window = "YOLO USB Camera"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed; retrying...")
            time.sleep(0.01)
            continue

        # Run YOLO inference
        results = model.predict(
            source=frame,                # ndarray input
            device=device,
            conf=args.conf,
            iou=args.iou,
            classes=args.classes,
            verbose=False
        )

        annotated = frame

        # Ultralytics returns a list; we passed a single frame
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy().astype(int)   # (x1,y1,x2,y2)
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), c, k in zip(boxes, confs, clss):
                color = tuple(int(v) for v in palette[k].tolist())
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{names.get(k, k)} {c:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                y_text = max(0, y1 - 8)
                cv2.rectangle(annotated, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 2), color, -1)
                cv2.putText(annotated, label, (x1 + 3, y_text - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # FPS overlay
        now = time.time()
        dt = now - last_t
        last_t = now
        if dt > 0:
            fps = (0.9 * fps) + (0.1 * (1.0 / dt)) if fps > 0 else (1.0 / dt)

        if args.show_fps:
            info = f"{fps:5.1f} FPS | dev={device_str}"
            cv2.putText(annotated, info, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 220, 20), 2, cv2.LINE_AA)

        # Show & optionally save
        cv2.imshow(window, annotated)
        if writer is not None:
            writer.write(annotated)

        # Quit with q or ESC
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
