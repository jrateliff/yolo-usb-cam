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
#   git add ollama_vision_speaker_async.py
#   git commit -m "Add async, throttled Ollama captioner (reduced power/latency)"
#   git push
#
# Notes:
# - Remote: git@github.com:jrateliff/yolo-usb-cam.git
# - Identity: user.name = jrateliff, user.email = jtrdevgit@gmail.com
# =============================================================================
"""
ollama_vision_speaker_async.py
Async, low-chatter, low-power captioner:
  • Camera loop never blocks on network or TTS.
  • Background worker pulls only the latest frame (no backlog).
  • Speak throttle + redundancy filter (skip near-duplicates).
  • Lower default FPS, resize, and JPEG Q to cut power spikes.

Prereqs:
  pip install "numpy<2" "opencv-python-headless<4.9" requests
  sudo apt install espeak-ng
"""

import argparse
import base64
import difflib
import queue
import subprocess
import textwrap
import threading
import time
from typing import Optional

import cv2
import numpy as np
import requests


# --------------------------- Utils ---------------------------

def bgr_to_jpeg_b64(img_bgr: np.ndarray, quality: int = 70) -> str:
    """Encode BGR frame to base64 JPEG string (no data URI)."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 50, 95))]
    ok, buf = cv2.imencode(".jpg", img_bgr, encode_params)
    if not ok:
        raise RuntimeError("JPEG encode failed.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def resize_if_needed(img: np.ndarray, max_width: int) -> np.ndarray:
    if img.shape[1] <= max_width:
        return img
    scale = max_width / img.shape[1]
    nh = int(img.shape[0] * scale)
    return cv2.resize(img, (max_width, nh), interpolation=cv2.INTER_AREA)


def overlay_caption(frame: np.ndarray, caption: str, max_width_frac: float = 0.95) -> np.ndarray:
    if not caption:
        return frame
    h, w = frame.shape[:2]
    margin = 10
    approx_char_w = 12
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
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

    y = y1 + margin + 14
    for ln in lines:
        cv2.putText(frame, ln, (x1 + margin, y), font, scale, (240, 240, 240), thickness, cv2.LINE_AA)
        y += line_h
    return frame


def speak_espeak(text: str, wpm: int = 160, voice: str = "en-us", gap_ms: int = 30, pitch: int = 48) -> None:
    if not text:
        return
    try:
        subprocess.run(
            ["espeak-ng", "-s", str(int(wpm)), "-v", voice, "-g", str(int(gap_ms)), "-p", str(int(pitch)), text],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except FileNotFoundError:
        pass


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio() if a and b else 0.0


# --------------------------- Ollama client ---------------------------

class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.sess = requests.Session()

    def describe_image(self, img_b64_jpeg: str, prompt: str, system: Optional[str] = None) -> str:
        # Try /api/chat (preferred)
        try:
            payload = {
                "model": self.model,
                "stream": False,
                "messages": [
                    *([{"role": "system", "content": system}] if system else []),
                    {"role": "user", "content": prompt, "images": [img_b64_jpeg]},
                ],
            }
            r = self.sess.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            msg = data.get("message") or {}
            content = msg.get("content") or ""
            if content.strip():
                return content.strip()
        except Exception:
            pass
        # Fallback /api/generate
        payload = {"model": self.model, "prompt": prompt, "images": [img_b64_jpeg], "stream": False}
        r = self.sess.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        resp = data.get("response", "")
        if resp.strip():
            return resp.strip()
        raise RuntimeError("Empty response from Ollama.")


# --------------------------- Workers ---------------------------

class CaptionWorker(threading.Thread):
    """
    Pulls the latest frame from frame_q (maxsize=1), rate-limited by describe_period.
    Emits cleaned captions on caption_q (maxsize=1), replacing older.
    """
    def __init__(self, client: OllamaClient, prompt: str, system: Optional[str],
                 frame_q: queue.Queue, caption_q: queue.Queue,
                 describe_period: float, max_resize_width: int, jpeg_quality: int):
        super().__init__(daemon=True)
        self.client = client
        self.prompt = prompt
        self.system = system
        self.frame_q = frame_q
        self.caption_q = caption_q
        self.describe_period = max(0.25, float(describe_period))
        self.max_resize_width = int(max(320, max_resize_width))
        self.jpeg_quality = int(np.clip(jpeg_quality, 50, 95))
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        next_time = time.monotonic()
        while not self._stop.is_set():
            # Pace requests
            now = time.monotonic()
            if now < next_time:
                time.sleep(min(0.01, next_time - now))
                continue
            next_time = now + self.describe_period

            # Always use the *latest* frame (drop older)
            frame = None
            try:
                while True:
                    frame = self.frame_q.get(timeout=0.2)
                    # Empty the queue to get the newest
                    while not self.frame_q.empty():
                        frame = self.frame_q.get_nowait()
                    break
            except queue.Empty:
                continue
            if frame is None:
                continue

            try:
                frame_small = resize_if_needed(frame, self.max_resize_width)
                b64 = bgr_to_jpeg_b64(frame_small, self.jpeg_quality)
                text = self.client.describe_image(b64, self.prompt, self.system)
                cleaned = " ".join(text.strip().split())
                if cleaned:
                    # Replace any pending caption with the newest
                    put_ok = False
                    while not put_ok:
                        try:
                            self.caption_q.put_nowait(cleaned)
                            put_ok = True
                        except queue.Full:
                            try:
                                _ = self.caption_q.get_nowait()
                            except queue.Empty:
                                pass
                    print(f"[{time.strftime('%H:%M:%S')}] {cleaned}")
            except Exception as e:
                print(f"Ollama request error: {e}")


class TTSWorker(threading.Thread):
    """Single-slot queue; latest utterance replaces older to avoid overlap/backlog."""
    def __init__(self, enabled: bool, wpm: int, voice: str, gap_ms: int, pitch: int):
        super().__init__(daemon=True)
        self.enabled = enabled
        self.wpm = wpm
        self.voice = voice
        self.gap_ms = gap_ms
        self.pitch = pitch
        self.q = queue.Queue(maxsize=1)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def say_replace(self, text: str):
        if not text:
            return
        while True:
            try:
                self.q.put_nowait(text)
                break
            except queue.Full:
                try:
                    _ = self.q.get_nowait()
                except queue.Empty:
                    pass

    def run(self):
        while not self._stop.is_set():
            try:
                text = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.enabled:
                speak_espeak(text, self.wpm, self.voice, self.gap_ms, self.pitch)


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Async, throttled Ollama vision speaker")
    # Capture / pacing
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=15.0, help="Lower FPS reduces power spikes.")
    # Model request cadence
    ap.add_argument("--describe", type=float, default=4.0, help="Seconds between caption requests.")
    # Ollama
    ap.add_argument("--ollama-url", type=str, default="http://127.0.0.1:11434")
    ap.add_argument("--ollama-model", type=str, default="moondream")
    ap.add_argument("--prompt", type=str, default="Describe the scene briefly.")
    ap.add_argument("--system", type=str, default=None)
    # Compression / resize
    ap.add_argument("--jpeg-quality", type=int, default=68)
    ap.add_argument("--max-resize-width", type=int, default=640)
    # UI / TTS
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--speak", action="store_true")
    ap.add_argument("--voice", type=str, default="en-us")
    ap.add_argument("--wpm", type=int, default=160)
    ap.add_argument("--gap", type=int, default=30)
    ap.add_argument("--pitch", type=int, default=48)
    # Speak gating
    ap.add_argument("--min-speak-interval", type=float, default=6.0,
                    help="Minimum seconds between spoken captions.")
    ap.add_argument("--min-change", type=float, default=0.40,
                    help="Speak only if change ≥ this fraction (1 - similarity).")
    # Misc
    ap.add_argument("--title", type=str, default="Ollama Vision Speaker (Async)")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    # Camera
    cap = cv2.VideoCapture(args.camera, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    cap.set(cv2.CAP_PROP_FPS, float(args.fps))
    if not cap.isOpened():
        raise RuntimeError(f"Camera {args.camera} failed to open.")

    # Queues
    frame_q: queue.Queue = queue.Queue(maxsize=1)
    caption_q: queue.Queue = queue.Queue(maxsize=1)

    # Clients / workers
    client = OllamaClient(args.ollama_url, args.ollama_model, timeout=args.timeout)
    cap_worker = CaptionWorker(client, args.prompt, args.system, frame_q, caption_q,
                               describe_period=args.describe,
                               max_resize_width=args.max_resize_width,
                               jpeg_quality=args.jpeg_quality)
    tts = TTSWorker(args.speak, args.wpm, args.voice, args.gap, args.pitch)
    cap_worker.start()
    tts.start()

    # State
    last_caption = ""
    last_spoken = ""
    last_spoken_t = 0.0

    # HUD FPS
    fps_t0 = time.time()
    fps_n = 0
    live_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            # Update HUD FPS
            fps_n += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                live_fps = fps_n / (now - fps_t0)
                fps_t0 = now
                fps_n = 0

            # Offer latest frame to worker (drop older)
            try:
                # If full, drop the pending older frame to keep only the newest
                if frame_q.full():
                    try:
                        _ = frame_q.get_nowait()
                    except queue.Empty:
                        pass
                frame_q.put_nowait(frame)
            except queue.Full:
                pass  # shouldn't happen due to drop logic

            # Pull newest caption if available
            try:
                while True:
                    last_caption = caption_q.get_nowait()
                    # drain to keep only the newest
                    if caption_q.empty():
                        break
            except queue.Empty:
                pass

            # Speak with throttle + change gate
            if args.speak and last_caption:
                if (now - last_spoken_t) >= args.min_speak_interval:
                    sim = similarity(last_spoken, last_caption)
                    if (1.0 - sim) >= args.min_change:
                        tts.say_replace(last_caption)
                        last_spoken = last_caption
                        last_spoken_t = time.time()

            # UI
            if args.show:
                disp = frame.copy()
                hud = f"{args.ollama_model} | {live_fps:.1f} FPS"
                cv2.putText(disp, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(disp, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                if last_caption:
                    disp = overlay_caption(disp, last_caption, 0.96)
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
        cap_worker.stop()
        tts.stop()


if __name__ == "__main__":
    main()
