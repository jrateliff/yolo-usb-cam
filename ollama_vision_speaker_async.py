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
#   git commit -m "TTS smoother: smart preempt + no-preempt window; default no-SSML, small word gap; hard-kill on exit"
#   git push
#
# Notes:
# - Remote: git@github.com:jrateliff/yolo-usb-cam.git
# - Identity: user.name = jrateliff, user.email = jtrdevgit@gmail.com
# =============================================================================
"""
ollama_vision_speaker_async.py

Async, overlay-only captioner with *smooth* speech:
  • Camera loop never blocks on network or TTS.
  • Speak throttle + similarity gate (skip near-duplicates).
  • TTS is preemptible but *politely*: smart preemption avoids mid-word chopping.
  • No-SSML by default (cleaner pacing); smaller inter-word gap.
  • TTS processes are tracked and killed instantly on exit/signals.
  • Optional FOURCC lock to MJPG for Logitech webcams.
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
from typing import Optional, List, Tuple

import cv2
import numpy as np
import requests


# --------------------------- Child process management (TTS) ---------------------------

class _ProcMgr:
    """Track child PIDs and kill them all on demand."""
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
                    os.killpg(p.pid, sig)  # whole group
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
        try:
            PROC_MGR.kill_all()
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            os._exit(0)  # ensure no lingering audio
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _cleanup_and_exit)
        except Exception:
            pass

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


# --------------------------- TTS (smooth + smart preemption) ---------------------------

def build_tts_command_espeak(text: str, wpm: int, voice: str, gap_ms: int, pitch: int, amp: int,
                             use_ssml: bool, comma_pause_ms: int, sentence_pause_ms: int) -> List[str]:
    cleaned = _clean_text_for_tts(text)
    if use_ssml:
        # Lightweight SSML: smaller breaks to avoid choppiness
        def to_ssml(s: str) -> str:
            s = re.sub(r'\s*,\s*', f', <break time="{int(comma_pause_ms)}ms"/> ', s)
            return f'<s>{s}</s>'
        sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', cleaned) or [cleaned]
        sents = [s.strip() for s in sents if s.strip()]
        body = f'\n<break time="{int(sentence_pause_ms)}ms"/>\n'.join(to_ssml(s) for s in sents)
        ssml = f'<speak version="1.0">{body}</speak>'
        return ["espeak-ng", "-m",
                "-s", str(int(wpm)), "-v", voice,
                "-g", str(int(gap_ms)), "-p", str(int(pitch)),
                "-a", str(int(np.clip(amp, 0, 200))),
                ssml]
    else:
        return ["espeak-ng",
                "-s", str(int(wpm)), "-v", voice,
                "-g", str(int(gap_ms)), "-p", str(int(pitch)),
                "-a", str(int(np.clip(amp, 0, 200))),
                cleaned]


class TTSWorker(threading.Thread):
    """
    Smooth, preemptible TTS.
    - 'preempt=smart': do NOT interrupt for small changes; allow a brief no-preempt window at start.
    - 'preempt=immediate': cut current audio as soon as a new caption arrives.
    - 'preempt=never': never cut; only play latest after current finishes.
    """
    def __init__(self, enabled: bool, wpm: int, voice: str, gap_ms: int, pitch: int,
                 amp: int, use_ssml: bool, comma_pause_ms: int, sentence_pause_ms: int,
                 audio_out: str, alsa_device: Optional[str],
                 preempt_mode: str, no_preempt_window: float, preempt_change: float):
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
        self.preempt_mode = preempt_mode
        self.no_preempt_window = max(0.0, float(no_preempt_window))
        self.preempt_change = float(np.clip(preempt_change, 0.0, 1.0))  # fraction change needed to cut
        self.q = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._current_proc: Optional[subprocess.Popen] = None
        self._speak_started_t: float = 0.0
        self._current_text: str = ""

    def stop(self):
        self._stop.set()
        PROC_MGR.kill_all()

    def say_replace(self, text: str):
        if not text:
            return
        # overwrite mailbox with latest text; do NOT kill here (preemption decided in run-loop)
        while True:
            try:
                self.q.put_nowait(text)
                break
            except queue.Full:
                try:
                    _ = self.q.get_nowait()
                except queue.Empty:
                    pass

    def _start_proc(self, cmd: List[str]) -> subprocess.Popen:
        p = subprocess.Popen(cmd, preexec_fn=os.setsid, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        PROC_MGR.track(p)
        self._current_proc = p
        self._speak_started_t = time.monotonic()
        return p

    def _finish_proc(self):
        p = self._current_proc
        self._current_proc = None
        if p is not None:
            try:
                PROC_MGR.untrack(p)
            except Exception:
                pass

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
                p.wait(timeout=0.15)
            except Exception:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:
                    p.kill()
        except Exception:
            pass
        finally:
            self._finish_proc()

    def _should_preempt(self, new_text: str) -> bool:
        if self.preempt_mode == "never":
            return False
        if self.preempt_mode == "immediate":
            # allow tiny safety window to avoid frantic cut-start loops
            return (time.monotonic() - self._speak_started_t) >= max(0.05, self.no_preempt_window)

        # smart mode
        if (time.monotonic() - self._speak_started_t) < self.no_preempt_window:
            return False
        sim = difflib.SequenceMatcher(None, (self._current_text or "").strip(), new_text.strip()).ratio()
        change = 1.0 - sim
        return change >= self.preempt_change  # only cut if substantially different

    def run(self):
        while not self._stop.is_set():
            # If nothing speaking, block a bit for next text
            if self._current_proc is None:
                try:
                    self._current_text = self.q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if not self.enabled or not _has_exec("espeak-ng"):
                    # drain until disabled/enabled flips
                    self._current_text = ""
                    continue
                cmd = build_tts_command_espeak(
                    self._current_text, self.wpm, self.voice, self.gap_ms, self.pitch, self.amp,
                    self.use_ssml, self.comma_pause_ms, self.sentence_pause_ms
                )
                self._start_proc(cmd)

            # While speaking, check for updates or process completion
            if self._current_proc is not None:
                # Collect the latest pending text if any (overwrite mailbox)
                pending_latest: Optional[str] = None
                try:
                    while True:
                        pending_latest = self.q.get_nowait()
                except queue.Empty:
                    pass

                if self._current_proc.poll() is not None:
                    # finished naturally
                    self._finish_proc()
                    self._current_text = ""
                    # If we had pending text, loop will pick it up and speak next
                    continue

                if pending_latest:
                    # Decide whether to preempt
                    if self._should_preempt(pending_latest):
                        self._kill_current()
                        self._current_text = pending_latest
                        if self.enabled and _has_exec("espeak-ng"):
                            cmd = build_tts_command_espeak(
                                self._current_text, self.wpm, self.voice, self.gap_ms, self.pitch, self.amp,
                                self.use_ssml, self.comma_pause_ms, self.sentence_pause_ms
                            )
                            self._start_proc(cmd)
                    else:
                        # Don't cut; keep latest to say after finish
                        try:
                            self.q.queue.clear()
                            self.q.put_nowait(pending_latest)
                        except Exception:
                            pass

                # brief sleep to avoid busy-waiting
                time.sleep(0.03)


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
        return ""


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
        self.enable_log = enable_log

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
                            try:
                                _ = self.caption_q.get_nowait()
                            except queue.Empty:
                                pass
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
    ap.add_argument("--fps", type=float, default=15.0, help="Lower FPS reduces power spikes.")
    ap.add_argument("--fourcc", type=str, default="MJPG",
                    help='FOURCC for camera. "MJPG" is good for Logitech. Use "raw" to skip.')

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

    # TTS shaping (defaults chosen for smoothness)
    ap.add_argument("--voice", type=str, default="en-us")
    ap.add_argument("--wpm", type=int, default=150)
    ap.add_argument("--gap", type=int, default=8, help="Inter-word gap in ms (smaller = smoother).")
    ap.add_argument("--pitch", type=int, default=50)
    ap.add_argument("--amp", type=int, default=170)
    ap.add_argument("--no-ssml", action="store_true", help="Disable SSML pacing; speak plain text.")
    ap.add_argument("--comma-pause", type=int, default=80)
    ap.add_argument("--sentence-pause", type=int, default=160)

    # Output backend controls (keep default = direct espeak for smoothness)
    ap.add_argument("--audio-out", type=str, choices=["auto", "espeak", "aplay", "paplay"], default="auto",
                    help="Force a specific audio path or let it auto-select.")
    ap.add_argument("--alsa-device", type=str, default=None,
                    help="ALSA device for aplay, e.g., hw:0,0")

    # Preemption policy
    ap.add_argument("--preempt", type=str, choices=["smart", "immediate", "never"], default="smart",
                    help="How aggressively to cut current speech for new text.")
    ap.add_argument("--no-preempt-window", type=float, default=1.0,
                    help="Seconds after speech start where cuts are disallowed in smart/immediate modes.")
    ap.add_argument("--preempt-change", type=float, default=0.70,
                    help="Fractional change (1-similarity) needed to cut speech in smart mode.")

    # Speak gating
    ap.add_argument("--min-speak-interval", type=float, default=6.0,
                    help="Minimum seconds between spoken captions.")
    ap.add_argument("--min-change", type=float, default=0.45,
                    help="Speak only if change ≥ this fraction (1 - similarity).")

    # Misc
    ap.add_argument("--title", type=str, default="Ollama Vision Speaker (Async)")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--log", action="store_true", help="Enable minimal terminal logs (off by default).")
    args = ap.parse_args()

    # Camera
    cap = cv2.VideoCapture(args.camera, cv2.CAP_ANY)

    # Optional FOURCC lock (helps Logitech webcams behave)
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
        preempt_mode=args.preempt,
        no_preempt_window=args.no_preempt_window,
        preempt_change=args.preempt_change
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
                    sim = difflib.SequenceMatcher(None, last_spoken.strip(), last_caption.strip()).ratio() if last_spoken else 0.0
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
        PROC_MGR.kill_all()


if __name__ == "__main__":
    main()
