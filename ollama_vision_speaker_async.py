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
#   git commit -m "Speech policy=finish: never interrupt; queue-next only; hard-kill on exit; MJPG default"
#   git push
#
# Notes:
# - Remote: git@github.com:jrateliff/yolo-usb-cam.git
# - Identity: user.name = jrateliff, user.email = jtrdevgit@gmail.com
# =============================================================================
"""
ollama_vision_speaker_async.py

Async overlay captioner with smooth, non-interrupting speech:
  • Camera loop never blocks on network or TTS.
  • On-screen captions update continuously.
  • Speech policy 'finish' (default): NEVER interrupt an utterance. New text becomes "next up".
  • Optional 'smart' or 'interrupt' modes if you want preemption.
  • TTS processes are tracked and killed instantly on exit/signals.
  • Logitech-friendly: FOURCC defaults to MJPG.
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


# --------------------------- Child process management (for TTS) ---------------------------

class _ProcMgr:
    """Track child PIDs and kill them fast to prevent lingering audio."""
    def __init__(self):
        self._lock = threading.Lock()
        self._procs = set()

    def track(self, p: subprocess.Popen):
        with self._lock:
            self._procs.add(p)

    def untrack(self, p: subprocess.Popen):
        with self._lock:
            self._procs.discard(p)

    def kill_all(self, sig=signal.SIGTERM, wait_s: float = 0.25):
        with self._lock:
            procs = list(self._procs)
            self._procs.clear()
        for p in procs:
            try:
                try:
                    os.killpg(p.pid, sig)  # nuke the whole process group
                except Exception:
                    p.terminate()
                try:
                    p.wait(timeout=wait_s)
                except Exception:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except Exception:
                        p.kill()
            except Exception:
                pass

PROC_MGR = _ProcMgr()

def _install_signal_handlers():
    def _cleanup_and_exit(signum, _frame):
        try: PROC_MGR.kill_all()
        finally:
            try: cv2.destroyAllWindows()
            except Exception: pass
            os._exit(0)  # ensure no lingering audio
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try: signal.signal(sig, _cleanup_and_exit)
        except Exception: pass

atexit.register(lambda: PROC_MGR.kill_all())


# --------------------------- Utils ---------------------------

def bgr_to_jpeg_b64(img_bgr: np.ndarray, quality: int = 68) -> str:
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
    t = text.strip()
    t = re.sub(r'https?://\S+', '', t)  # URLs sound awful
    t = t.replace('&', ' and ').replace('/', ' / ')
    t = re.sub(r'\s+', ' ', t)
    if t and t[-1] not in '.!?':
        t = t + '.'
    t = re.sub(r'([.!?,]){2,}', r'\1', t)
    return t.strip()


def _has_exec(name: str) -> bool:
    return shutil.which(name) is not None


# --------------------------- TTS (speech policies) ---------------------------
# Policies:
#  - finish (default): NEVER interrupt current speech; store only the latest "next" line.
#  - smart: interrupt only if the new line differs a lot and a short window has passed.
#  - interrupt: cut immediately on any new line.

class TTSWorker(threading.Thread):
    def __init__(self,
                 enabled: bool,
                 wpm: int, voice: str, gap_ms: int, pitch: int, amp: int,
                 use_ssml: bool,
                 comma_pause_ms: int, sentence_pause_ms: int,
                 audio_out: str, alsa_device: Optional[str],
                 policy: str = "finish", smart_change: float = 0.75, smart_window: float = 1.0):
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
        self.audio_out = audio_out
        self.alsa_device = alsa_device
        self.policy = policy
        self.smart_change = float(np.clip(smart_change, 0.0, 1.0))
        self.smart_window = max(0.0, float(smart_window))

        self._q = queue.Queue(maxsize=1)    # mailbox: put only when idle
        self._next_text: Optional[str] = None  # next-up storage
        self._current_text: str = ""
        self._current_proc: Optional[subprocess.Popen] = None
        self._speak_start_t: float = 0.0
        self._stop = threading.Event()

    # ---- public API ----
    def stop(self):
        self._stop.set()
        PROC_MGR.kill_all()

    def is_speaking(self) -> bool:
        return self._current_proc is not None and self._current_proc.poll() is None

    def offer(self, text: str):
        """Offer a new line to speak respecting the current policy."""
        if not text or not self.enabled:
            return
        text = _clean_text_for_tts(text)

        # If currently speaking, decide whether to interrupt or queue-next.
        if self.is_speaking():
            if self.policy == "interrupt":
                self._cut_and_replace(text)
                return
            if self.policy == "smart":
                # Only cut if sufficiently different and past the small window.
                sim = difflib.SequenceMatcher(None, self._current_text.strip(), text.strip()).ratio() if self._current_text else 0.0
                change = 1.0 - sim
                if (time.monotonic() - self._speak_start_t) >= self.smart_window and change >= self.smart_change:
                    self._cut_and_replace(text)
                    return
            # 'finish': do not cut; store as the next-up line (replace older).
            self._next_text = text
            return

        # Not speaking: try to enqueue immediately (mailbox semantics).
        try:
            self._q.put_nowait(text)
        except queue.Full:
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(text)
            except Exception:
                pass

    # ---- internals ----
    def _cut_and_replace(self, text: str):
        self._kill_current()
        self._next_text = text  # speak this next

    def _start_proc(self, cmd: List[str]) -> subprocess.Popen:
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        PROC_MGR.track(p)
        self._current_proc = p
        self._speak_start_t = time.monotonic()
        return p

    def _finish_proc(self):
        p = self._current_proc
        self._current_proc = None
        if p is not None:
            try: PROC_MGR.untrack(p)
            except Exception: pass

    def _kill_current(self):
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
            self._finish_proc()

    def _build_cmd(self, text: str) -> List[str]:
        # Plain espeak by default for smoothness; light SSML if requested.
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
        # Light SSML
        ssml = self._to_ssml(text, self.comma_pause_ms, self.sentence_pause_ms)
        return [
            "espeak-ng", "-m",
            "-s", str(int(self.wpm)),
            "-v", self.voice,
            "-g", str(int(self.gap_ms)),
            "-p", str(int(self.pitch)),
            "-a", str(int(np.clip(self.amp, 0, 200))),
            ssml,
        ]

    @staticmethod
    def _to_ssml(text: str, comma_pause_ms: int, sentence_pause_ms: int) -> str:
        # Small breaks only (bigger breaks can sound choppy on some sinks)
        def inject_commas(s: str) -> str:
            return re.sub(r'\s*,\s*', f', <break time="{int(comma_pause_ms)}ms"/> ', s)
        sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text) or [text]
        sents = [s.strip() for s in sents if s.strip()]
        body = f'\n<break time="{int(sentence_pause_ms)}ms"/>\n'.join(f'<s>{inject_commas(s)}</s>' for s in sents)
        return f'<speak version="1.0">{body}</speak>'

    def run(self):
        while not self._stop.is_set():
            # If currently speaking, check completion; if done, speak next if queued.
            if self.is_speaking():
                if self._current_proc.poll() is not None:
                    self._finish_proc()
                    self._current_text = ""
                else:
                    time.sleep(0.03)
                    continue

            # If we have a next-up line, do that first.
            if self._next_text:
                nxt = self._next_text
                self._next_text = None
                self._current_text = nxt
                cmd = self._build_cmd(nxt)
                if _has_exec("espeak-ng"):
                    self._start_proc(cmd)
                continue

            # Otherwise pull from mailbox (arrives only when idle).
            try:
                nxt = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            self._current_text = nxt
            cmd = self._build_cmd(nxt)
            if _has_exec("espeak-ng"):
                self._start_proc(cmd)


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


# --------------------------- Workers ---------------------------

class CaptionWorker(threading.Thread):
    """
    Pulls the latest frame from frame_q (maxsize=1), rate-limited by describe_period.
    Emits cleaned captions on caption_q (maxsize=1), always replacing older.
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

            # Always use the latest frame
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
                            try: _ = self.caption_q.get_nowait()
                            except queue.Empty: pass
                    self._log(f"[{time.strftime('%H:%M:%S')}] {cleaned}")
            except Exception:
                pass


# --------------------------- Main ---------------------------

def main():
    _install_signal_handlers()

    ap = argparse.ArgumentParser(description="Async, throttled Ollama vision speaker (overlay-only)")

    # Capture / pacing
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=15.0, help="Lower FPS reduces USB/link stress.")
    ap.add_argument("--fourcc", type=str, default="MJPG",
                    help='FOURCC for camera. "MJPG" helps Logitech stability. Use "raw" to skip.')

    # Model request cadence
    ap.add_argument("--describe", type=float, default=5.5, help="Seconds between caption requests (slightly slower = fewer interruptions).")

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

    # TTS shaping (defaults chosen for smoothness)
    ap.add_argument("--voice", type=str, default="en-us")
    ap.add_argument("--wpm", type=int, default=150)
    ap.add_argument("--gap", type=int, default=8, help="Inter-word gap in ms (smaller is smoother).")
    ap.add_argument("--pitch", type=int, default=50)
    ap.add_argument("--amp", type=int, default=170)
    ap.add_argument("--no-ssml", action="store_true", help="Disable SSML pacing; plain espeak is usually smoother.")
    ap.add_argument("--comma-pause", type=int, default=70)
    ap.add_argument("--sentence-pause", type=int, default=140)

    # Output backend (kept simple: direct espeak by default)
    ap.add_argument("--audio-out", type=str, choices=["auto", "espeak", "aplay", "paplay"], default="auto")
    ap.add_argument("--alsa-device", type=str, default=None, help="ALSA device for aplay, e.g., hw:0,0")

    # Speech policy
    ap.add_argument("--speak-policy", type=str, choices=["finish", "smart", "interrupt"], default="finish",
                    help="finish=never interrupt; smart=cut only if very different; interrupt=cut immediately.")
    ap.add_argument("--smart-change", type=float, default=0.75, help="1 - similarity needed to cut in smart mode.")
    ap.add_argument("--smart-window", type=float, default=1.0, help="No-cut window (s) at start in smart mode.")

    # Speak gating (still useful to reduce spam)
    ap.add_argument("--min-speak-interval", type=float, default=6.0,
                    help="Minimum seconds between starting spoken captions.")
    ap.add_argument("--min-change", type=float, default=0.45,
                    help="Speak only if change ≥ this fraction (1 - similarity).")

    # Misc
    ap.add_argument("--title", type=str, default="Ollama Vision Speaker (Async)")
    ap.add_argument("--timeout", type=float, default=60.0)
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
        sentence_pause_ms=args.sentence_pause,
        audio_out=args.audio_out,
        alsa_device=args.alsa_device,
        policy=args.speak_policy,
        smart_change=args.smart_change,
        smart_window=args.smart_window
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
                    try: _ = frame_q.get_nowait()
                    except queue.Empty: pass
                frame_q.put_nowait(frame)
            except queue.Full:
                pass

            # Pull newest caption if available (for overlay + potential speech)
            try:
                while True:
                    last_caption = caption_q.get_nowait()
                    if caption_q.empty():
                        break
            except queue.Empty:
                pass

            # Speech gating: start a new utterance only if idle and different enough
            if args.speak and last_caption and not tts.is_speaking():
                if (now - last_spoken_t) >= args.min_speak_interval:
                    sim = difflib.SequenceMatcher(None, last_spoken.strip(), last_caption.strip()).ratio() if last_spoken else 0.0
                    if (1.0 - sim) >= args.min_change:
                        tts.offer(last_caption)       # will speak because idle
                        last_spoken = last_caption
                        last_spoken_t = time.time()
            elif args.speak and last_caption and tts.is_speaking():
                # While speaking, we still feed new text to be "next up", but we DO NOT interrupt in 'finish' mode.
                tts.offer(last_caption)

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
        try: cap.release()
        except Exception: pass
        if args.show:
            try: cv2.destroyAllWindows()
            except Exception: pass
        cap_worker.stop()
        tts.stop()
        PROC_MGR.kill_all()


if __name__ == "__main__":
    main()
