"""
FingerLock - main.py

Runs on the Arduino UNO Q's Linux side (inside App Lab), or on your laptop for testing.

  Laptop:  python main.py --camera 0        then open http://localhost:7000
  Board:   press Run in App Lab             then open http://<board-name>.local:7000

What it does:
  * reads the USB webcam
  * runs the finger-counting network (ONNX Runtime, no PyTorch needed)
  * runs the lock: Start -> hold up 3 numbers -> correct code opens the safe
  * serves the web page with the live neural-net visualization
  * tells the STM32 sketch (over the Bridge) to move the motor and set the LEDs.
    On a laptop there is no Bridge, so motor commands are only printed.
"""
import argparse
import base64
import hmac
import json
import os
import struct
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import onnxruntime as ort

from preprocess import crop_16x9, preprocess

HERE = Path(__file__).resolve().parent

# =============================== SETTINGS ===================================
PASSCODE = "253"          # your code: three digits, each 1-5
CONF_THRESHOLD = 0.80     # the net must be at least this sure about what it sees
HOLD_SEC = 1.0            # hold a number steady this long to enter it
RELEASE_SEC = 0.4         # then show no hand (or a fist) this long before the next number
READ_TIMEOUT_SEC = 25     # give up if nothing is entered for this long
DENIED_SEC = 2.5          # how long "wrong code" is shown
MAX_FAILURES = 3          # wrong codes in a row before a lockout
LOCKOUT_SEC = 30
SMOOTH_FRAMES = 4         # average predictions over this many frames
VIZ_FPS = 5               # visualization frames sent to the browser per second
PORT = 7000
CAMERA = os.environ.get("FINGERLOCK_CAMERA", "auto")   # "auto", "arduino", an index like "0", or "/dev/video2"
CAMERA_IS_RGB = False     # set True if the camera image on the web page shows red and blue swapped
# ============================================================================

try:                                  # only exists on the UNO Q, inside App Lab
    from arduino.app_utils import App, Bridge
    ON_BOARD = True
except ImportError:
    App = Bridge = None
    ON_BOARD = False


# ----------------------------------------------------------------------------
class Hardware:
    """Talks to sketch.ino on the STM32. On a laptop, just prints."""
    STATUS = {"locked": 0, "reading": 1, "opening": 2, "open": 2, "closing": 1, "denied": 3, "lockout": 3}

    def __init__(self):
        print("Hardware:", "UNO Q Bridge" if ON_BOARD else "MOCK - motor/LED commands are only printed")

    def _call(self, name):
        """Returns True if the sketch confirmed the command."""
        if not ON_BOARD:
            print(f"[mock MCU] {name}()")
            return True
        try:
            Bridge.call(name, timeout=5)
            return True
        except Exception as e:                 # TimeoutError, ConnectionError, ValueError...
            print(f"Bridge call {name} failed: {e}")
            return False

    def open_safe(self):
        return self._call("open_safe")

    def close_safe(self):
        return self._call("close_safe")

    def set_status(self, state):
        code = self.STATUS.get(state, 0)
        if ON_BOARD:
            try:
                Bridge.notify("set_status", code)
            except Exception as e:
                print("Bridge notify failed:", e)
        else:
            print(f"[mock MCU] set_status({code})  # {state}")

    def heartbeat(self):
        if ON_BOARD:
            try:
                Bridge.notify("heartbeat")
            except Exception:
                pass


# ----------------------------------------------------------------------------
class CameraReader(threading.Thread):
    """Keeps grabbing frames in the background so we always use the newest one."""

    def __init__(self, source, is_rgb):
        super().__init__(daemon=True)
        self.source, self.is_rgb = str(source), is_rgb
        self.cond = threading.Condition()
        self.frame, self.frame_id, self.error = None, 0, None

    def _open(self):
        if ON_BOARD and self.source in ("auto", "arduino"):
            try:
                from arduino.app_peripherals.camera import Camera
                cam = Camera("usb:0", resolution=(1280, 720), fps=15)
                cam.start()
                print("Camera: Arduino Camera peripheral (usb:0)")
                return lambda: cam.capture()
            except Exception as e:
                if self.source == "arduino":
                    raise
                print("Arduino Camera peripheral failed, trying OpenCV:", e)
        if sys.platform.startswith("linux") and self.source in ("auto", "arduino"):
            candidates = self._list_linux_cameras()      # e.g. ["/dev/video2", ...]
        else:
            src = self.source.replace("/dev/video", "")      # "/dev/video2" -> "2"
            candidates = [int(src) if src.isdigit() else self.source]
        for src in candidates:
            cap = self._try_open(src)
            if cap is not None:
                print(f"Camera: OpenCV {src!r} at {int(cap.get(3))}x{int(cap.get(4))}")
                return lambda: (lambda ok_f: ok_f[1] if ok_f[0] else None)(cap.read())
        raise RuntimeError(f"no working camera among {candidates or 'no /dev/video* devices'}")

    @staticmethod
    def _list_linux_cameras():
        import glob
        devices = sorted(glob.glob("/dev/video*"), key=lambda p: int("".join(c for c in p if c.isdigit()) or 0))
        cameras = []
        for d in devices:
            try:
                name = open(f"/sys/class/video4linux/{os.path.basename(d)}/name").read().strip()
            except OSError:
                name = "?"
            skip = "venus" in name.lower()       # the board's own video encoder/decoder, not cameras
            print(f"  found {d}: {name}{'  (skipped: not a camera)' if skip else ''}")
            if not skip:
                cameras.append(int("".join(c for c in os.path.basename(d) if c.isdigit())))
        if not cameras:
            print("  no USB camera visible. Is the webcam on a powered USB-C hub? Check with: lsusb")
        return cameras

    @staticmethod
    def _try_open(src):
        """Open a camera and make sure it really delivers a frame."""
        if sys.platform.startswith("win"):
            backend = cv2.CAP_DSHOW
        elif sys.platform.startswith("linux"):
            backend = cv2.CAP_V4L2        # skip GStreamer, which can't handle these devices well
        else:
            backend = cv2.CAP_ANY
        cap = cv2.VideoCapture(src, backend)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for _ in range(5):
            ok, f = cap.read()
            if ok and f is not None:
                return cap
            time.sleep(0.1)
        cap.release()
        return None

    def run(self):
        while True:
            try:
                grab = self._open()
                self.error = None
                fails = 0
                while fails < 30:
                    f = grab()
                    if f is None:
                        fails += 1
                        time.sleep(0.05)
                        continue
                    fails = 0
                    if self.is_rgb:
                        f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                    with self.cond:
                        self.frame, self.frame_id = f, self.frame_id + 1
                        self.cond.notify_all()
                raise RuntimeError("camera stopped sending frames")
            except Exception as e:
                self.error = f"Camera problem: {e}"
                print(self.error, "- retrying in 3 s")
                time.sleep(3)

    def wait_new(self, last_id, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.frame_id > last_id, timeout=timeout)
            return self.frame, self.frame_id


# ----------------------------------------------------------------------------
class Engine:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        onnx_path, meta_path = model_dir / "fingernet_viz.onnx", model_dir / "fingernet_viz.json"
        if not onnx_path.exists():
            sys.exit(f"Model not found: {onnx_path}\nRun export_for_board.py on your laptop first.")
        self.meta = json.loads(meta_path.read_text())
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(3, (os.cpu_count() or 2) - 1))
        self.sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.layer_names = [l["name"] for l in self.meta["layers"]]
        self.out_w = np.array(self.meta["out_weights"], dtype=np.float32)
        print(f"Model: {self.meta['source_checkpoint']} (width {self.meta['width']}, depth {self.meta['depth']})")

    def run(self, x):
        outs = dict(zip(self.output_names, self.sess.run(None, {self.input_name: x})))
        logits = outs["logits"][0]
        e = np.exp(logits - logits.max())
        return e / e.sum(), outs["fc"][0], [outs[n][0] for n in self.layer_names]


# ----------------------------------------------------------------------------
class Lock:
    """The safe's logic. Every change of state goes through here."""

    def __init__(self, hw, code_len, no_hand):
        self.hw, self.code_len, self.no_hand = hw, code_len, no_hand
        self.mutex = threading.Lock()
        self.digits, self.shown, self.failures = [], 0, 0
        self.until = self.last_activity = self.hold_start = 0.0
        self.hold_label, self.waiting_release = None, False
        self.progress, self.progress_kind = 0.0, "digit"
        self.state, self.msg = "locked", ""
        self._set("locked", "Press Start, then hold up your code one number at a time.")

    def _set(self, state, msg):
        changed = state != self.state
        self.state, self.msg = state, msg
        if changed or state == "locked":
            self.hw.set_status(state)

    def start(self):
        with self.mutex:
            if self.state == "lockout":
                return False, "Too many wrong codes. Wait for the lockout to end."
            if self.state in ("open", "opening", "closing"):
                return False, "The safe is open. Close it first."
            self.digits, self.shown = [], 0
            self.hold_label, self.waiting_release, self.progress = None, False, 0.0
            self.last_activity = time.time()
            self._set("reading", f"Show number 1 of {self.code_len}.")
            return True, "Reading started."

    def close(self):
        with self.mutex:
            if self.state != "open":
                return False, "The safe isn't open."
            self._set("closing", "Closing...")
        ok = self.hw.close_safe()
        with self.mutex:
            if ok:
                self.shown = 0
                self._set("locked", "Locked. Press Start to open again.")
                return True, "Safe closed."
            self._set("open", "The board didn't answer, so the safe may still be open.")
            return False, "Close failed - check the board."

    def update(self, label, conf, now):
        action = None
        with self.mutex:
            if self.state == "denied" and now >= self.until:
                self.shown = 0
                self._set("locked", "Press Start to try again.")
            if self.state == "lockout" and now >= self.until:
                self.failures, self.shown = 0, 0
                self._set("locked", "Lockout over. Press Start to try again.")
            if self.state != "reading":
                self.progress = 0.0
                return
            if now - self.last_activity > READ_TIMEOUT_SEC:
                self.digits, self.shown = [], 0
                self._set("locked", "Timed out. Press Start to try again.")
                return

            sure = label is not None and conf >= CONF_THRESHOLD
            if self.waiting_release:          # between numbers: need an empty view or a fist
                kind, need, ok = "release", RELEASE_SEC, sure and label == self.no_hand
            else:
                kind, need, ok = "digit", HOLD_SEC, sure and label != self.no_hand
            if not ok:
                self.hold_label, self.progress = None, 0.0
                return
            if label != self.hold_label:
                self.hold_label, self.hold_start = label, now
            self.progress_kind = kind
            self.progress = min(1.0, (now - self.hold_start) / need)
            if self.progress < 1.0:
                return

            self.hold_label, self.progress, self.last_activity = None, 0.0, now
            if kind == "release":
                self.waiting_release = False
                self.msg = f"Show number {len(self.digits) + 1} of {self.code_len}."
                return
            self.digits.append(label)
            self.shown = len(self.digits)
            if len(self.digits) < self.code_len:
                self.waiting_release = True
                self.msg = "Got it. Lower your hand before the next number."
                return

            entered, self.digits = "".join(self.digits), []
            if hmac.compare_digest(entered, PASSCODE):
                self._set("opening", "Correct code. Opening...")
                action = "open"
            else:
                self.failures += 1
                if self.failures >= MAX_FAILURES:
                    self.until = now + LOCKOUT_SEC
                    self._set("lockout", f"Wrong code {MAX_FAILURES} times. Locked out for {LOCKOUT_SEC} s.")
                else:
                    self.until = now + DENIED_SEC
                    self._set("denied", f"Wrong code ({self.failures} of {MAX_FAILURES} tries).")

        if action == "open":                      # motor call happens outside the mutex
            ok = self.hw.open_safe()
            with self.mutex:
                if ok:
                    self.failures = 0
                    self._set("open", "Open. Press Close safe to lock it again.")
                else:
                    self.shown = 0
                    self._set("locked", "The board didn't answer, so the safe stayed locked.")

    def snapshot(self, now):
        with self.mutex:
            return {
                "state": self.state, "msg": self.msg, "shown": self.shown, "code_len": self.code_len,
                "progress": round(self.progress, 3),
                "progress_kind": "release" if self.waiting_release else "digit",
                "failures": self.failures,
                "lockout_left": max(0, round(self.until - now)) if self.state == "lockout" else 0,
            }


# ----------------------------------------------------------------------------
class LayerScale:
    """Brightness relative to how strongly a layer usually fires (running 99th percentile)."""

    def __init__(self, decay=0.95):
        self.ref, self.decay = None, decay

    def __call__(self, a):
        p = max(float(np.percentile(a, 99)), 1e-6)
        self.ref = p if self.ref is None else self.decay * self.ref + (1 - self.decay) * p
        return np.clip(a / self.ref, 0, 1) ** 0.8


def downsample_max(a, target_h=24):
    """Shrink big feature maps (max-pool) so the browser gets ~170 KB per frame, not 500 KB."""
    c, h, w = a.shape
    f = 1
    while h // (f * 2) >= target_h and h % (f * 2) == 0 and w % (f * 2) == 0:
        f *= 2
    return a if f == 1 else a.reshape(c, h // f, f, w // f, f).max(axis=(2, 4))


class Publisher:
    """Holds the newest visualization packet; browsers long-poll for it."""

    def __init__(self):
        self.cond = threading.Condition()
        self.seq, self.packet = 0, None

    def publish(self, header, payload=b""):
        with self.cond:
            self.seq += 1
            header["seq"] = self.seq
            hb = json.dumps(header, separators=(",", ":")).encode()
            self.packet = struct.pack("<I", len(hb)) + hb + payload
            self.cond.notify_all()

    def wait_newer(self, after, timeout=2.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq > after and self.packet is not None, timeout=timeout)
            return self.packet if self.seq > after else None


# ----------------------------------------------------------------------------
class FingerLockApp:
    def __init__(self, model_dir, camera):
        self.hw = Hardware()
        self.engine = Engine(model_dir)
        meta = self.engine.meta
        self.lock = Lock(self.hw, len(PASSCODE), meta["no_hand_class"])
        self.camera = CameraReader(camera, CAMERA_IS_RGB)
        self.pub = Publisher()
        self.scales = {n: LayerScale() for n in self.engine.layer_names + ["fc"]}
        bad = [d for d in PASSCODE if d not in meta["classes"] or d == meta["no_hand_class"]]
        if bad:
            sys.exit(f"PASSCODE {PASSCODE!r} can only use the digits 1-5.")

    def start(self):
        self.camera.start()
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        classes = self.engine.meta["classes"]
        history = deque(maxlen=SMOOTH_FRAMES)
        frame_id, last_viz, last_beat, fps = 0, 0.0, 0.0, 0.0
        while True:
            frame, new_id = self.camera.wait_new(frame_id, timeout=1.0)
            now = time.time()
            if now - last_beat > 0.5:
                self.hw.heartbeat()
                last_beat = now
            if new_id == frame_id or frame is None:      # no new frame: keep the lock's timers running
                self.lock.update(None, 0.0, now)
                if now - last_viz > 1.0:
                    hdr = self.lock.snapshot(now)
                    hdr.update(error=self.camera.error or "Waiting for the camera...", layers=[])
                    self.pub.publish(hdr)
                    last_viz = now
                continue
            t0 = time.time()
            frame_id = new_id
            x, seen = preprocess(frame, self.engine.meta)
            probs, fc, acts = self.engine.run(x)
            history.append(probs)
            avg = np.mean(history, axis=0)
            k = int(avg.argmax())
            self.lock.update(classes[k], float(avg[k]), now)
            fps = 0.9 * fps + 0.1 / max(time.time() - t0, 1e-3)

            if now - last_viz >= 1.0 / VIZ_FPS:
                last_viz = now
                self.publish_viz(frame, seen, avg, k, fc, acts, now, fps)

    def publish_viz(self, frame, seen, probs, k, fc, acts, now, fps):
        chunks, layers, offset = [seen.tobytes()], [], seen.size
        for name, a in zip(self.engine.layer_names, acts):
            u8 = (self.scales[name](downsample_max(a)) * 255).astype(np.uint8)
            c, h, w = u8.shape
            layers.append({"name": name, "c": c, "h": h, "w": w, "offset": offset, "full": list(a.shape[1:])})
            chunks.append(u8.tobytes())
            offset += u8.size
        fc_u8 = (self.scales["fc"](fc) * 255).astype(int).tolist()
        contrib = fc * self.engine.out_w[k]          # which FC neurons pushed toward the answer
        lines = []
        if contrib.max() > 0:
            for j in np.argsort(contrib)[::-1][:15]:
                v = float(contrib[j] / contrib.max())
                if v < 0.1:
                    break
                lines.append([int(j), round(v, 3)])
        thumb = cv2.resize(crop_16x9(frame), (256, 144), interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 60])
        hdr = self.lock.snapshot(now)
        hdr.update(
            error=None,
            classes=self.engine.meta["classes"],
            pred=self.engine.meta["classes"][k], conf=round(float(probs[k]), 3),
            probs=[round(float(p), 3) for p in probs],
            fc=fc_u8, lines=lines, layers=layers,
            seen={"h": int(seen.shape[0]), "w": int(seen.shape[1]), "offset": 0},
            cam=base64.b64encode(jpg.tobytes()).decode() if ok else None,
            fps=round(fps, 1),
        )
        self.pub.publish(hdr, b"".join(chunks))


# ----------------------------------------------------------------------------
APP = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):          # keep the console quiet
        pass

    def _send(self, code, body, ctype):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass                            # the browser tab closed mid-request; nothing to do

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            self._send(200, (HERE / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/frame":
            after = int(parse_qs(url.query).get("after", ["0"])[0])
            pkt = APP.pub.wait_newer(after)
            if pkt is None:
                self.send_response(204)
                self.end_headers()
            else:
                self._send(200, pkt, "application/octet-stream")
        elif url.path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n:
            self.rfile.read(n)
        path = urlparse(self.path).path
        if path == "/api/start":
            ok, msg = APP.lock.start()
        elif path == "/api/close":
            ok, msg = APP.lock.close()
        else:
            return self._json({"ok": False, "msg": "Unknown command"}, 404)
        self._json({"ok": ok, "msg": msg})


def main():
    global APP
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default=CAMERA)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--model-dir", default=str(HERE / "model"))
    args = p.parse_args()

    APP = FingerLockApp(args.model_dir, args.camera)
    APP.start()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.daemon_threads = True
    print(f"FingerLock web page on port {args.port}"
          + ("" if ON_BOARD else f"  ->  http://localhost:{args.port}"))
    if ON_BOARD:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        App.run()                          # keeps the App Lab app alive
    else:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopped.")


if __name__ == "__main__":
    main()