#!/usr/bin/env python3
# =============================================================================
# Git Commit Instructions (for this project)
# -----------------------------------------------------------------------------
# 1. Stage changes:
#       git add ollama_vision_speaker.py
#
# 2. Commit with a clear message:
#       git commit -m "Update ollama_vision_speaker.py: <short description>"
#
# 3. Push to GitHub (main branch):
#       git push
#
# Notes:
# - Your Git identity is set as:
#       user.name  = jrateliff
#       user.email = jtrdevgit@gmail.com
# - Remote origin is set to SSH:
#       git@github.com:jrateliff/yolo-usb-cam.git
# - VS Code Source Control panel can also do stage/commit/push.
# =============================================================================


"""
ollama_vision_speaker.py
Realtime camera captions using OpenCV + a local Ollama vision model (default: moondream),
with optional spoken output via espeak-ng. Prints captions to the terminal and can show
a preview window with overlays.

Usage examples:
  python ollama_vision_speaker.py
  python ollama_vision_speaker.py --camera 1 --width 1280 --height 720
  python ollama_vision_speaker.py --describe 2.5 --speak --show
  python ollama_vision_speaker.py --ollama-model moondream    # lightweight, good on 8GB
  python ollama_vision_speaker.py --ollama-model llava:7b     # heavier model

Prereqs:
  • Ollama running locally with a multimodal model pulled (e.g., `ollama run moondream`)
  • Python packages: opencv-python, numpy, requests
  • Optional: espeak-ng for text-to-speech
"""

import argparse
import base64
import json
import os
import queue
import subprocess
import textwrap
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import requests


# --------------------------- Utility ---------------------------

def bgr_to_jpeg_b64(img_bgr: np.ndarray, quality: int = 85) -> str:
    """Encode a BGR image to base64 JPEG string without data URI prefix."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 50, 95))]
    ok, buf = cv2.imencode(".jpg", img_bgr, encode_params)
    if not ok:
        raise RuntimeError("JPEG encode failed.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def overlay_caption(frame: np.ndarray, caption: str, max_width_frac: float = 0.9) -> np.ndarray:
    """Draw a semi-transparent caption box with wrapped text at the bottom of the frame."""
    if not caption:
        return frame

    h, w = frame.shape[:2]
    margin = 10
    # Estimate characters per line based on width
    approx_char_w = 12  # heuristic for cv2.putText (FONT_HERSHEY_SIMPLEX, 0.6)
    max_chars = max(20, int((w * max_width_frac - 2 * margin) // approx_char_w))
    lines = textwrap.wrap(caption.strip(), width=max_chars)

    # Text metrics
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    line_h = 18

    box_h = line_h * len(lines) + 2 * margin
    y1 = h - box_h - margin
    y2 = h - margin
    x1 = margin
    x2 = int(w * max_width_frac)

    # Background with alpha
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    alpha = 0.55
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

    # Draw text
    y = y1 + margin + 14
    for ln in lines:
        cv2.putText(frame, ln, (x1 + margin, y), font, scale, (240, 240, 240), thickness, cv2.LINE_AA)
        y += line_h

    return frame


def speak_espeak(text: str, wpm: int = 175, voice: str = "en-us") -> None:
    """Speak using espeak-ng. Non-blocking wrapper should call this in a thread."""
    if not text:
        return
    try:
        subprocess.run(
            ["espeak-ng", "-s", str(int(wpm)), "-v", voice, text],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        # espeak-ng not installed — silently ignore (we still print to console)
        pass


# --------------------------- Ollama client ---------------------------

class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.sess = requests.Session()

    def describe_image(
        self,
        img_b64_jpeg: str,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Try /api/chat first (preferred for multimodal), fall back to /api/generate if needed.
        Returns the text content or raises RuntimeError on failure.
        """
        # 1) Try /api/chat
        try:
            payload = {
                "model": self.model,
                "stream": bool(stream),
                "messages": [
                    *([{"role": "system", "content": system}] if system else []),
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_b64_jpeg],  # raw base64 (no prefix)
                    },
                ],
            }
            r = self.sess.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                # Standard non-stream response
                msg = data.get("message") or {}
                content = msg.get("content") or ""
                if content.strip():
                    return content.strip()
            # If streaming, some Ollama versions return NDJSON. We requested stream=False by default.
        except Exception as e:
            # Fall through to /api/generate
            pass

        # 2) Fallback: /api/generate (older path also supports images for some models)
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [img_b64_jpeg],
                "stream": False,
            }
            r = self.sess.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            text = data.get("response", "")
            if text.strip():
                return text.strip()
            raise RuntimeError("Empty response from /api/generate.")
        except Exception as e:
            raise RuntimeError(f"Ollama request failed: {e}")


# --------------------------- Worker threads ---------------------------

class TTSWorker(threading.Thread):
    """A small worker to speak strings without blocking the main loop."""
    def __init__(self, enabled: bool, wpm: int = 170, voice: str = "en-us"):
        super().__init__(daemon=True)
        self.enabled = enabled
        self.wpm = wpm
        self.voice = voice
        self.q = queue.Queue(maxsize=8)
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                text = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not self.enabled:
                continue
            speak_espeak(text, wpm=self.wpm, voice=self.voice)

    def say(self, text: str):
        if not text:
            return
        # drop oldest if full
        try:
            self.q.put_nowait(text)
        except queue.Full:
            _ = self.q.get_nowait()
            self.q.put_nowait(text)

    def stop(self):
        self._stop.set()


# --------------------------- Main app ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Realtime camera captions using Ollama.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Capture FPS hint.")
    parser.add_argument("--describe", type=float, default=3.0,
                        help="Seconds between captions (float).")
    parser.add_argument("--ollama-url", type=str, default="http://127.0.0.1:11434",
                        help="Base URL for Ollama.")
    parser.add_argument("--ollama-model", type=str, default="moondream",
                        help="Ollama model name, e.g., 'moondream' or 'llava:7b'.")
    parser.add_argument("--prompt", type=str, default="Describe the scene briefly.",
                        help="Prompt sent with each frame.")
    parser.add_argument("--system", type=str, default=None,
                        help="Optional system message to steer the model.")
    parser.add_argument("--jpeg-quality", type=int, default=85,
                        help="JPEG quality for frames [50..95].")
    parser.add_argument("--show", action="store_true", help="Show preview window with overlays.")
    parser.add_argument("--speak", action="store_true", help="Speak captions via espeak-ng.")
    parser.add_argument("--voice", type=str, default="en-us", help="espeak-ng voice.")
    parser.add_argument("--wpm", type=int, default=170, help="espeak-ng words per minute.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds.")
    parser.add_argument("--title", type=str, default="Ollama Vision Speaker", help="Window title.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    cap.set(cv2.CAP_PROP_FPS, float(args.fps))

    if not cap.isOpened():
        raise RuntimeError(f"Camera {args.camera} failed to open.")

    client = OllamaClient(args.ollama_url, args.ollama_model, timeout=args.timeout)
    tts = TTSWorker(enabled=args.speak, wpm=args.wpm, voice=args.voice)
    tts.start()

    last_caption = ""
    last_request_t = 0.0
    describe_period = max(0.1, float(args.describe))

    # Simple FPS meter for overlay
    fps_t0 = time.time()
    fps_counter = 0
    live_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                # Brief backoff if camera hiccups
                time.sleep(0.01)
                continue

            # Update FPS estimate
            fps_counter += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                live_fps = fps_counter / (now - fps_t0)
                fps_t0 = now
                fps_counter = 0

            # Request a new caption if interval elapsed
            if (now - last_request_t) >= describe_period:
                last_request_t = now
                # Prepare the image (downscale for speed if very large)
                send_img = frame
                # Optional quick resize if width > 1024 to reduce bandwidth/latency
                max_w = 1024
                if send_img.shape[1] > max_w:
                    scale = max_w / send_img.shape[1]
                    nh = int(send_img.shape[0] * scale)
                    send_img = cv2.resize(send_img, (max_w, nh), interpolation=cv2.INTER_AREA)

                try:
                    b64 = bgr_to_jpeg_b64(send_img, quality=args.jpeg_quality)
                    text = client.describe_image(
                        img_b64_jpeg=b64,
                        prompt=args.prompt,
                        system=args.system,
                        stream=False,
                    )
                    # Clean up whitespace and overly long runs
                    cleaned = " ".join(text.strip().split())
                    if cleaned:
                        last_caption = cleaned
                        print(f"[{time.strftime('%H:%M:%S')}] {last_caption}")
                        if args.speak:
                            tts.say(last_caption)
                except Exception as e:
                    err = f"Ollama request error: {e}"
                    print(err)
                    # Keep previous caption; try again next cycle

            # Show preview with overlays if requested
            if args.show:
                disp = frame.copy()
                # Small HUD with FPS and model name
                hud = f"{args.ollama_model} | {live_fps:.1f} FPS"
                cv2.putText(disp, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(disp, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

                if last_caption:
                    disp = overlay_caption(disp, last_caption, max_width_frac=0.96)

                cv2.imshow(args.title, disp)
                # 1ms wait keeps UI responsive; Esc or q quits
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
            else:
                # No window — tiny sleep avoids pegging CPU
                time.sleep(0.001)

    finally:
        try:
            cap.release()
        except Exception:
            pass
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        tts.stop()


if __name__ == "__main__":
    main()


    # This is a test commit to ensure the script runs without errors.