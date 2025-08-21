#!/usr/bin/env python3

# use command:  source ~/yolo-venv/bin/activate
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
#   git commit -m "Async captioner: natural SSML speech, pause tuning, overlay-only"
#   git push
#
# Notes:
# - Remote: git@github.com:jrateliff/yolo-usb-cam.git
# - Identity: user.name = jrateliff, user.email = jtrdevgit@gmail.com
# =============================================================================
"""
ollama_vision_speaker_async.py
Async, low-chatter, overlay-only captioner with more natural speech:
  • Camera loop never blocks on network or TTS.
  • Background worker always uses the latest frame (no backlog).
  • Speak throttle + similarity gate (skip near-duplicates).
  • Overlay text onto the preview window; NO terminal prints by default.
  • Program does NOT change Jetson power mode.

Prereqs:
  pip install "numpy<2" "opencv-python-headless<4.9" requests
  sudo apt install espeak-ng
"""

import argparse
import base64
import difflib
import queue
import re
import subprocess
import textwrap
import threading
import time
from typing import Optional, List

import cv2
import numpy as np
import requests


# --------------------------- Utils ---------------------------

def bgr_to_jpeg_b64(img_bgr: np.ndarray, quality: int = 68) -> str:
    """Encode BGR frame to base64 JPEG string (no data URI)."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 50, 95))]
    ok, buf = cv2.imencode(".jpg", img_bgr, encode_params)
    if not ok:
        # No prints: fail silently by returning empty payload
        return ""
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


def _clean_text_for_tts(text: str) -> str:
    """
    Gentle cleanup so TTS sounds less robotic:
      • Remove URLs.
      • Collapse whitespace/punctuation.
      • Ensure sentence-final punctuation exists.
      • Expand a few common symbols.
    """
    t = text.strip()

    # Remove URLs (they sound terrible)
    t = re.sub(r'https?://\S+', '', t)

    # Replace some symbols
    t = t.replace('&', ' and ').replace('/', ' / ')
    t = re.sub(r'\s+', ' ', t)

    # Ensure punctuation at end
    if t and t[-1] not in '.!?':
        t = t + '.'

    # Avoid triple punctuation
    t = re.sub(r'([.!?,]){2,}', r'\1', t)

    return t.strip()


def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter."""
    # Split on ., !, ? followed by space/cap/number
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
    # Fallback if nothing split
    if len(parts) == 1:
        # Try commas
        parts = [p.strip() for p in re.split(r'\s*,\s*', text) if p.strip()]
    else:
        parts = [p.strip() for p in parts if p.strip()]
    return parts


def _to_ssml(text: str, comma_pause_ms: int, sentence_pause_ms: int) -> str:
    """
    Convert plain text into simple SSML for more natural pacing.
    Adds short breaks after commas and longer breaks between sentences.
    """
    t = _clean_text_for_tts(text)
    sents = _split_sentences(t)

    ssml_sents = []
    for s in sents:
        # Insert small pauses at commas
        s = re.sub(r'\s*,\s*', f', <break time="{int(comma_pause_ms)}ms"/> ', s)
        ssml_sents.append(f'<s>{s}</s>')

    # Pause between sentences
    joiner = f'\n<break time="{int(sentence_pause_ms)}ms"/>\n'
    body = joiner.join(ssml_sents)
    return f'<speak version="1.0">{body}</speak>'


def speak_espeak(
    text: str,
    wpm: int = 150,
    voice: str = "en-us",
    gap_ms: int = 30,
    pitch: int = 48,
    amp: int = 160,
    use_ssml: bool = True,
    comma_pause_ms: int = 140,
    sentence_pause_ms: int = 260,
) -> None:
    """
    Speak using espeak-ng. When SSML is enabled, we wrap the text with <speak>
    and inject <break> tags to add natural pauses.
    """
    if not text:
        return
    try:
        if use_ssml:
            ssml = _to_ssml(text, comma_pause_ms, sentence_pause_ms)
            # -m to enable SSML parsing; pass text as a single arg
            subprocess.run(
                [
                    "espeak-ng",
                    "-m",
                    "-s", str(int(wpm)),
                    "-v", voice,
                    "-g", str(int(gap_ms)),
                    "-p", str(int(pitch)),
                    "-a", str(int(np.clip(amp, 0, 200))),
                    ssml,
                ],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            # Plain mode
            cleaned = _clean_text_for_tts(text)
            subprocess.run(
                [
                    "espeak-ng",
                    "-s", str(int(wpm)),
                    "-v", voice,
                    "-g", str(int(gap_ms)),
                    "-p", str(int(pitch)),
                    "-a", str(int(np.clip(amp, 0, 200))),
                    cleaned,
                ],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    except FileNotFoundError:
        # Silent failure by design
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
        if not img_b64_jpeg:
            return ""
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
        try:
            payload = {"model": self.model, "prompt": prompt, "images": [img_b64_jpeg], "stream": False}
            r = self.sess.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            resp = data.get("response", "")
            if resp.strip():
                return resp.strip()
        except Exception:
            pass
        return ""  # No terminal prints


# --------------------------- Workers ---------------------------

class CaptionWorker(threading.Thread):
    """
    Pulls the latest frame from frame_q (maxsize=1), rate-limited by describe_period.
    Emits cleaned captions on caption_q (maxsize=1), replacing older.
    """
    def __init__(self, client: OllamaClient, prompt: str, system: Optional[str],
                 frame_q: queue.Queue, caption_q: queue.Queue,
                 describe_period: float, max_resize_width: int, jpeg_quality: int,
                 enable_log: bool = False):
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
        self.enable_log = enable_log  # if True, will print minimal logs

    def stop(self):
        self._stop.set()

    def _log(self, msg: str):
        if self.enable_log:
            try:
                print(msg)
            except Exception:
                pass

    def run(self):
        next_time = time.monotonic()
        while not self._stop.is_set():
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
                    self._log(f"[{time.strftime('%H:%M:%S')}] {cleaned}")
            except Exception:
                # Silent by default
                pass


class TTSWorker(threading.Thread):
    """Single-slot queue; latest utterance replaces older to avoid overlap/backlog."""
    def __init__(self, enabled: bool, wpm: int, voice: str, gap_ms: int, pitch: int,
                 amp: int, ssml: bool, comma_pause_ms: int, sentence_pause_ms: int):
        super().__init__(daemon=True)
        self.enabled = enabled
        self.wpm = wpm
        self.voice = voice
        self.gap_ms = gap_ms
        self.pitch = pitch
        self.amp = amp
        self.ssml = ssml
        self.comma_pause_ms = comma_pause_ms
        self.sentence_pause_ms = sentence_pause_ms
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
                speak_espeak(
                    text,
                    wpm=self.wpm,
                    voice=self.voice,
                    gap_ms=self.gap_ms,
                    pitch=self.pitch,
                    amp=self.amp,
                    use_ssml=self.ssml,
                    comma_pause_ms=self.comma_pause_ms,
                    sentence_pause_ms=self.sentence_pause_ms,
                )


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Async, throttled Ollama vision speaker (overlay-only)")

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

    # TTS voice shaping (tuned for naturalness)
    ap.add_argument("--voice", type=str, default="en-us")
    ap.add_argument("--wpm", type=int, default=150, help="Speaking rate (words per minute).")
    ap.add_argument("--gap", type=int, default=25, help="Inter-word gap in ms (espeak-ng -g).")
    ap.add_argument("--pitch", type=int, default=50, help="Voice pitch (0-99).")
    ap.add_argument("--amp", type=int, default=160, help="Amplitude/volume (0-200).")
    ap.add_argument("--no-ssml", action="store_true", help="Disable SSML pacing and speak plain text.")
    ap.add_argument("--comma-pause", type=int, default=140, help="Pause at commas (ms) when SSML is on.")
    ap.add_argument("--sentence-pause", type=int, default=260, help="Pause between sentences (ms) with SSML.")

    # Speak gating
    ap.add_argument("--min-speak-interval", type=float, default=6.0,
                    help="Minimum seconds between spoken captions.")
    ap.add_argument("--min-change", type=float, default=0.40,
                    help="Speak only if change ≥ this fraction (1 - similarity).")

    # Misc
    ap.add_argument("--title", type=str, default="Ollama Vision Speaker (Async)")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--log", action="store_true", help="Enable minimal terminal logs (off by default).")
    args = ap.parse_args()

    # Camera
    cap = cv2.VideoCapture(args.camera, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    cap.set(cv2.CAP_PROP_FPS, float(args.fps))
    if not cap.isOpened():
        # No prints: raise to exit quietly; caller sees no logs
        raise RuntimeError(f"Camera {args.camera} failed to open.")

    # Queues
    frame_q: queue.Queue = queue.Queue(maxsize=1)
    caption_q: queue.Queue = queue.Queue(maxsize=1)

    # Clients / workers
    client = OllamaClient(args.ollama_url, args.ollama_model, timeout=args.timeout)
    cap_worker = CaptionWorker(
        client, args.prompt, args.system, frame_q, caption_q,
        describe_period=args.describe,
        max_resize_width=args.max_resize_width,
        jpeg_quality=args.jpeg_quality,
        enable_log=args.log
    )
    tts = TTSWorker(
        enabled=args.speak,
        wpm=args.wpm,
        voice=args.voice,
        gap_ms=args.gap,
        pitch=args.pitch,
        amp=args.amp,
        ssml=(not args.no-ssml),
        comma_pause_ms=args.comma_pause,
        sentence_pause_ms=args.sentence_pause
    )
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
                if frame_q.full():
                    try:
                        _ = frame_q.get_nowait()
                    except queue.Empty:
                        pass
                frame_q.put_nowait(frame)
            except queue.Full:
                pass

            # Pull newest caption if available
            try:
                while True:
                    last_caption = caption_q.get_nowait()
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
