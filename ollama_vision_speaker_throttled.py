#!/usr/bin/env python3
# =============================================================================
# Git Commit Process for This File
# -----------------------------------------------------------------------------
# From VS Code:
#   1. Open the Source Control panel (left bar, Git icon).
#   2. Type your commit message in the box at the top.
#   3. Press Ctrl+Enter (or click the ✓ Commit button).
#   4. Push changes: bottom-left → "Push"/"Sync Changes".
#
# From Terminal:
#   cd ~/yolo-usb-cam
#   git add ollama_vision_speaker.py
#   git commit -m "Update ollama_vision_speaker.py: <short description>"
#   git push
#
# Notes:
# - Remote: git@github.com:jrateliff/yolo-usb-cam.git
# - Identity: user.name = jrateliff, user.email = jtrdevgit@gmail.com
# =============================================================================
"""
ollama_vision_speaker.py
Realtime camera captions using OpenCV + a local Ollama vision model (default: moondream),
with optional spoken output via espeak-ng. Prints captions to the terminal and can show
a preview window with overlays.

This version adds:
  • Speak throttling: don't talk too often (--min-speak-interval)
  • Redundancy filter: only speak if caption changed enough (--min-change)
  • Natural pacing: word gap/pitch/rate controls for espeak-ng
  • Lower default bandwidth for less latency/power (smaller resize + jpeg quality)
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
import difflib


# --------------------------- Utility ---------------------------

def bgr_to_jpeg_b64(img_bgr: np.ndarray, quality: int = 75) -> str:
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
    approx_char_w = 12  # heuristic for cv2.putText (FONT_HERSHEY_SIMPLEX, 0.6)
    max_chars = max(20, int((w * max_width_frac - 2 * margin) // approx_char_w))
    lines = textwrap.wrap(caption.strip(), width=max_chars)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    line_h = 18

    box_h = line_h * len(lines) + 2 * margin
    y1 = h - box_h - margin
    y2 = h - margin
    x1 = margin
    x2 = int(w * max_width_frac)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), -1)
    alpha = 0.55
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

    y = y1 + margin + 14
    for ln in lines:
        cv2.putText(frame, ln, (x1 + margin, y), font, scale, (240, 240, 240), thickness, cv2.LINE_AA)
        y += line_h

    return frame


def speak_espeak(text: str, wpm: int = 175, voice: str = "en-us", gap_ms: int = 10, pitch: int = 50) -> None:
    """
    Speak using espeak-ng. gap_ms sets inter-word gap; pitch ~0..99.
    Run this in a worker thread (blocking).
    """
    if not text:
        return
    try:
        subprocess.run(
            ["espeak-ng",
             "-s", str(int(wpm)),
             "-v", voice,
             "-g", str(int(gap_ms)),
             "-p", str(int(pitch)),
             text],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass  # espeak-ng not installed — ignore


def similarity(a: str, b: str) -> float:
    """Return a 0..1 similarity score using difflib."""
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio() if a and b else 0.0


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
                msg = data.get("message") or {}
                content = msg.get("content") or ""
                if content.strip():
                    return content.strip()
        except Exception:
            pass

        # 2) Fallback: /api/generate
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
    """Speak strings without overlapping, with a 1-item queue (latest wins)."""
    def __init__(self, enabled: bool, wpm: int = 170, voice: str = "en-us", gap_ms: int = 10, pitch: int = 50):
        super().__init__(daemon=True)
        self.enabled = enabled
        self.wpm = wpm
        self.voice = voice
        self.gap_ms = gap_ms
        self.pitch = pitch
        self.q = queue.Queue(maxsize=1)  # 1 item: newest replaces older
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                text = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not self.enabled:
                continue
            speak_espeak(text, wpm=self.wpm, voice=self.voice, gap_ms=self.gap_ms, pitch=self.pitch)

    def say_replace(self, text: str):
        """Replace any pending utterance with this one."""
        if not text:
            return
        while True:
            try:
                # If full, drop the pending one
                self.q.put_nowait(text)
                break
            except queue.Full:
                try:
                    _ = self.q.get_nowait()
                except queue.Empty:
                    pass

    def stop(self):
        self._stop.set()


# --------------------------- Main app ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Realtime camera captions using Ollama.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    parser.add_argument("--fps", type=float, default=24.0, help="Capture FPS hint (lower saves power).")
    parser.add_argument("--describe", type=float, default=3.5,
                        help="Seconds between caption requests.")
    parser.add_argument("--ollama-url", type=str, default="http://127.0.0.1:11434",
                        help="Base URL for Ollama.")
    parser.add_argument("--ollama-model", type=str, default="moondream",
                        help="Ollama model name, e.g., 'moondream' or 'llava:7b'.")
    parser.add_argument("--prompt", type=str, default="Describe the scene briefly.",
                        help="Prompt sent with each frame.")
    parser.add_argument("--system", type=str, default=None,
                        help="Optional system message to steer the model.")
    parser.add_argument("--jpeg-quality", type=int, default=72,
                        help="JPEG quality for frames [50..95]. Lower = less bandwidth/latency.")
    parser.add_argument("--max-resize-width", type=int, default=800,
                        help="Downscale width before sending to Ollama (<=1024 recommended on Jetson).")
    parser.add_argument("--show", action="store_true", help="Show preview window with overlays.")
    parser.add_argument("--speak", action="store_true", help="Speak captions via espeak-ng.")
    parser.add_argument("--voice", type=str, default="en-us", help="espeak-ng voice.")
    parser.add_argument("--wpm", type=int, default=165, help="espeak-ng words per minute.")
    parser.add_argument("--gap", type=int, default=25, help="espeak-ng word gap (ms) for natural pacing.")
    parser.add_argument("--pitch", type=int, default=48, help="espeak-ng pitch (0..99).")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds.")
    # NEW: speak throttling and redundancy filtering
    parser.add_argument("--min-speak-interval", type=float, default=4.0,
                        help="Minimum seconds between spoken captions.")
    parser.add_argument("--min-change", type=float, default=0.30,
                        help="Only speak if similarity to last spoken < 1 - min-change. Range 0..1.")
    parser.add_argument("--title", type=str, default="Ollama Vision Speaker", help="Window title.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    cap.set(cv2.CAP_PROP_FPS, float(args.fps))

    if not cap.isOpened():
        raise RuntimeError(f"Camera {args.camera} failed to open.")

    client = OllamaClient(args.ollama_url, args.ollama_model, timeout=args.timeout)
    tts = TTSWorker(enabled=args.speak, wpm=args.wpm, voice=args.voice, gap_ms=args.gap, pitch=args.pitch)
    tts.start()

    last_caption = ""
    last_spoken = ""
    last_request_t = 0.0
    last_spoken_t = 0.0
    describe_period = max(0.3, float(args.describe))

    # Simple FPS meter for overlay
    fps_t0 = time.time()
    fps_counter = 0
    live_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            fps_counter += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                live_fps = fps_counter / (now - fps_t0)
                fps_t0 = now
                fps_counter = 0

            # Request a new caption if interval elapsed
            if (now - last_request_t) >= describe_period:
                last_request_t = now

                send_img = frame
                max_w = int(max(320, args.max_resize_width))
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
                    cleaned = " ".join(text.strip().split())
                    if cleaned:
                        last_caption = cleaned
                        print(f"[{time.strftime('%H:%M:%S')}] {last_caption}")

                        # --- SPEAK DEBOUNCE / THROTTLE / REDUNDANCY FILTER ---
                        if args.speak:
                            # 1) Time gate
                            if (now - last_spoken_t) >= args.min_speak_interval:
                                # 2) Change gate (speak only if changed enough)
                                sim = similarity(last_spoken, last_caption)
                                changed_enough = (1.0 - sim) >= args.min_change
                                if changed_enough:
                                    tts.say_replace(last_caption)
                                    last_spoken = last_caption
                                    last_spoken_t = time.time()
                except Exception as e:
                    print(f"Ollama request error: {e}")

            # Show preview with overlays if requested
            if args.show:
                disp = frame.copy()
                hud = f"{args.ollama_model} | {live_fps:.1f} FPS"
                cv2.putText(disp, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(disp, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

                if last_caption:
                    disp = overlay_caption(disp, last_caption, max_width_frac=0.96)

                cv2.imshow(args.title, disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
            else:
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
