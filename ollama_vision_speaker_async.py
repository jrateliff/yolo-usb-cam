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
#   git commit -m "Speech: strict finish (no mid-talk cuts), overlay live; Logitech MJPG defaults; hard-kill on exit"
#   git push
#
# Notes:
# - Remote: git@github.com:jrateliff/yolo-usb-cam.git
# - Identity: user.name = jrateliff, user.email = jtrdevgit@gmail.com
# =============================================================================
"""
ollama_vision_speaker_async.py

Async overlay captioner with *strict finish* speech:
  • Camera loop never blocks on network or TTS.
  • Overlay updates continuously with the latest caption.
  • Speech NEVER interrupts mid-utterance; new text becomes "next up" and replaces any older pending.
  • Logitech-friendly: FOURCC defaults to MJPG, FPS lowered to reduce USB resets.
  • TTS child process group is killed on exit (no lingering audio).
  • Program does NOT change Jetson power mode.

Prereqs:
  pip install "numpy<2" "opencv-python-headless<4.9" requests
  sudo apt install espeak-ng alsa-utils
"""

import argparse
import atexit
import base64
import difflib
import os
import queue
import re
import shutil
import signal
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
    """Light cleanup to keep TTS smooth."""
    t = text.strip()
    t = re.sub(r'https?://\S+', '', t)   # URLs sound awful
    t = t.replace('&', ' and ').replace('/', ' / ')
    t = re.sub(r'\s+', ' ', t)
    if t and t[-1] not in '.!?':
        t = t + '.'
    t = re.sub(r'([.!?,]){2,}', r'\1', t)
    return t.strip()


def _has_exec(name: str) -> bool:
    return shutil.which(name) is not None


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
        # Preferred: /api/chat
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
        # Fallback: /api/generate
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
        return ""


# --------------------------- Caption worker ---------------------------

class CaptionWorker(threading.Thread):
    """
    Pulls the latest frame from frame_q (maxsize=1), rate-limited by describe_period.
    Emits cleaned captions on caption_q (maxsize=1), replacing older pending.
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
        self.enable_log = enable_log

    def stop(self):
        self._stop.set()

    def _log(self, msg: str):
        if self.enable_log:
            try: print(msg)
            except Exception: pass

    def run(self):
        next_time = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_time:
                time.sleep(min(0.01, next_time - now))
                continue
            next_time = now + self.describe_period

            # Always take the newest frame (drop backlog)
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
                    put_ok = False
                    while not put_ok:
                        try:
                            self.caption_q.put_nowait(cleaned)
                            put_ok = True
                        except queue.Full:
                            try: _ = self.caption_q.get_nowait()
                            except queue.Empty: pass
                    self._log(f"[{time.strftime('%H:%M:%S')}] {cleaned}")
            except Exception:
                pass


# --------------------------- TTS worker (STRICT FINISH) ---------------------------

class TTSWorker(threading.Thread):
    """
    STRICT FINISH policy:
      - Never cut current speech.
      - While speaking, keep only the latest pending text (next-up).
      - Start the next text *only after* the current process exits.
    """
    def __init__(self, enabled: bool, wpm: int, voice: str, gap_ms: int, pitch: int, amp: int,
                 use_ssml: bool, comma_pause_ms: int, sentence_pause_ms: int):
        super().__init__(daemon=True)
        self.enabled = enabled
        self.wpm = wpm
        self.voice = voice
        self.gap_ms = gap_ms
        self.pitch = pitch
        self.amp = amp
        self.use_ssml = use_ssml
        self.comma_pause_ms = comma_pause_ms
        self.sentence_pause_ms = sentence_pause_ms

        self._stop = threading.Event()
        self._current_proc: Optional[subprocess.Popen] = None
        self._next_text: Optional[str] = None
        self._mailbox: queue.Queue = queue.Queue(maxsize=1)  # only used when idle

        # Ensure TTS is reaped on exit/signals
        atexit.register(self._hard_kill)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(sig, self._signal_exit)
            except Exception:
                pass

    # ------------- public API -------------

    def stop(self):
        self._stop.set()
        self._hard_kill()

    def is_speaking(self) -> bool:
        return self._current_proc is not None and self._current_proc.poll() is None

    def offer(self, text: str):
        if not self.enabled or not text:
            return
        t = _clean_text_for_tts(text)
        if self.is_speaking():
            # Do not cut. Replace the pending "next" line.
            self._next_text = t
            return
        # Idle: put into mailbox (replace older if needed)
        try:
            self._mailbox.put_nowait(t)
        except queue.Full:
            try:
                _ = self._mailbox.get_nowait()
            except queue.Empty:
                pass
            try:
                self._mailbox.put_nowait(t)
            except Exception:
                pass

    # ------------- internals -------------

    def _signal_exit(self, signum, frame):
        self._hard_kill()
        os._exit(0)

    def _hard_kill(self):
        p = self._current_proc
        if p is None:
            return
        try:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:
                p.terminate()
            try:
                p.wait(timeout=0.2)
            except Exception:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:
                    p.kill()
        except Exception:
            pass
        finally:
            self._current_proc = None

    def _build_cmd(self, text: str) -> List[str]:
        if not self.use_ssml:
            return [
                "espeak-ng",
                "-s", str(int(self.wpm)),
                "-v", self.voice,
                "-g", str(int(self.gap_ms)),
                "-p", str(int(self.pitch)),
                "-a", str(int(np.clip(self.amp, 0, 200))),
                text,
            ]
        # very light SSML, conservative pauses
        def inject_commas(s: str) -> str:
            return re.sub(r'\s*,\s*', f', <break time="{int(self.comma_pause_ms)}ms"/> ', s)
        sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text) or [text]
        sents = [s.strip() for s in sents if s.strip()]
        body = f'\n<break time="{int(self.sentence_pause_ms)}ms"/>\n'.join(f'<s>{inject_commas(s)}</s>' for s in sents)
        ssml = f'<speak version="1.0">{body}</speak>'
        return [
            "espeak-ng", "-m",
            "-s", str(int(self.wpm)),
            "-v", self.voice,
            "-g", str(int(self.gap_ms)),
            "-p", str(int(self.pitch)),
            "-a", str(int(np.clip(self.amp, 0, 200))),
            ssml,
        ]

    def run(self):
        while not self._stop.is_set():
            # If currently speaking, just wait a tick and re-check
            if self.is_speaking():
                time.sleep(0.03)
                # If speech finished naturally, clear handle
                if self._current_proc and self._current_proc.poll() is not None:
                    self._current_proc = None
                continue

            # If we have a pending "next" line, speak it now
            if self._next_text:
                nxt = self._next_text
                self._next_text = None
            else:
                # Otherwise pull from mailbox (arrives only when idle)
                try:
                    nxt = self._mailbox.get(timeout=0.2)
                except queue.Empty:
                    continue

            if not nxt or not _has_exec("espeak-ng"):
                continue

            cmd = self._build_cmd(nxt)
            # Put espeak in its own process group so we can nuke it on exit
            self._current_proc = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Loop back; we won't speak anything else until this process exits


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Async, throttled Ollama vision speaker (overlay-only)")

    # Capture / pacing (defaults softened for Logitech/USB stability)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=12.0, help="Lower FPS reduces USB/link stress.")
    ap.add_argument("--fourcc", type=str, default="MJPG",
                    help='FOURCC for camera. "MJPG" helps Logitech stability. Use "raw" to skip setting it.')

    # Model cadence (slower = fewer chances to queue too many next-ups)
    ap.add_argument("--describe", type=float, default=6.0, help="Seconds between caption requests.")

    # Ollama
    ap.add_argument("--ollama-url", type=str, default="http://127.0.0.1:11434")
    ap.add_argument("--ollama-model", type=str, default="moondream")
    ap.add_argument("--prompt", type=str, default="Describe the scene briefly.")
    ap.add_argument("--system", type=str, default=None)
    ap.add_argument("--timeout", type=float, default=60.0)

    # Compression / resize
    ap.add_argument("--jpeg-quality", type=int, default=68)
    ap.add_argument("--max-resize-width", type=int, default=640)

    # UI / TTS
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--speak", action="store_true")

    # TTS shaping (smooth defaults)
    ap.add_argument("--voice", type=str, default="en-us")
    ap.add_argument("--wpm", type=int, default=150)
    ap.add_argument("--gap", type=int, default=8, help="Inter-word gap in ms (smaller is smoother).")
    ap.add_argument("--pitch", type=int, default=50)
    ap.add_argument("--amp", type=int, default=170)
    ap.add_argument("--no-ssml", action="store_true", help="Disable SSML pacing; plain espeak is smoothest.")
    ap.add_argument("--comma-pause", type=int, default=70)
    ap.add_argument("--sentence-pause", type=int, default=140)

    # Speak gating (prevents spammy re-reads)
    ap.add_argument("--min-speak-interval", type=float, default=6.0,
                    help="Minimum seconds between starting spoken captions.")
    ap.add_argument("--min-change", type=float, default=0.45,
                    help="Speak only if change ≥ this fraction (1 - similarity).")

    # Misc
    ap.add_argument("--title", type=str, default="Ollama Vision Speaker (Async)")
    ap.add_argument("--log", action="store_true", help="Enable minimal terminal logs (off by default).")
    args = ap.parse_args()

    # Camera
    cap = cv2.VideoCapture(args.camera, cv2.CAP_ANY)
    if args.fourcc and args.fourcc.upper() != "RAW":
        try:
            four = cv2.VideoWriter_fourcc(*args.fourcc.upper())
            cap.set(cv2.CAP_PROP_FOURCC, four)
        except Exception:
            pass
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
        use_ssml=(not args.no_ssml),
        comma_pause_ms=args.comma_pause,
        sentence_pause_ms=args.sentence_pause
    )
    cap_worker.start()
    tts.start()

    # State for gating speech
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

            # HUD FPS
            fps_n += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                live_fps = fps_n / (now - fps_t0)
                fps_t0 = now
                fps_n = 0

            # Offer latest frame to worker (drop older)
            try:
                if frame_q.full():
                    try: _ = frame_q.get_nowait()
                    except queue.Empty: pass
                frame_q.put_nowait(frame)
            except queue.Full:
                pass

            # Pull newest caption for overlay and speech gating
            try:
                while True:
                    last_caption = caption_q.get_nowait()
                    if caption_q.empty():
                        break
            except queue.Empty:
                pass

            # Overlay refresh
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

            # Speech gating: ONLY enqueue when idle, never preempt
            if args.speak and last_caption and not tts.is_speaking():
                if (now - last_spoken_t) >= args.min_speak_interval:
                    sim = difflib.SequenceMatcher(None, last_spoken.strip(), last_caption.strip()).ratio() if last_spoken else 0.0
                    if (1.0 - sim) >= args.min_change:
                        tts.offer(last_caption)  # will speak because idle
                        last_spoken = last_caption
                        last_spoken_t = time.time()

            if not args.show:
                time.sleep(0.001)

    finally:
        try: cap.release()
        except Exception: pass
        if args.show:
            try: cv2.destroyAllWindows()
            except Exception: pass
        cap_worker.stop()
        tts.stop()


if __name__ == "__main__":
    main()
