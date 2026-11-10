"""
collect_data.py - record labeled hand images from the webcam.

Controls (click the video window first so it receives your keypresses):
  0-5    start recording that class (press the same key again to stop)
  SPACE  stop recording
  q      quit

Images go to data/<class>/<session>_<n>.jpg
Example: python collect_data.py --cam 1
         python collect_data.py --cam 1 --out val_data    (separate validation session)
"""
import argparse
import os
import time

import cv2

from common import CLASSES, SAVE_H, SAVE_W, open_camera


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=1, help="webcam index (built-in is usually 0)")
    p.add_argument("--out", default="data")
    p.add_argument("--fps", type=float, default=8, help="images saved per second while recording")
    args = p.parse_args()

    for c in CLASSES:
        os.makedirs(os.path.join(args.out, c), exist_ok=True)
    counts = {c: len(os.listdir(os.path.join(args.out, c))) for c in CLASSES}

    cap = open_camera(args.cam)
    session = time.strftime("%Y%m%d-%H%M%S")
    recording, last_save = None, 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed - is the camera still plugged in?")
            break

        now = time.time()
        if recording is not None and now - last_save >= 1.0 / args.fps:
            small = cv2.resize(frame, (SAVE_W, SAVE_H), interpolation=cv2.INTER_AREA)
            name = f"{session}_{counts[recording]:05d}.jpg"
            cv2.imwrite(os.path.join(args.out, recording, name), small)
            counts[recording] += 1
            last_save = now

        # ---- display ----
        disp = frame.copy()
        color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.rectangle(disp, (0, 0), (disp.shape[1] - 1, disp.shape[0] - 1), color, 8)  # red border = recording
        status = (f"REC class {recording}  (SPACE to stop)" if recording
                  else "Press 0-5 to record, SPACE stop, q quit")
        cv2.putText(disp, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        tally = "  ".join(f"{c}:{counts[c]}" for c in CLASSES)
        cv2.putText(disp, tally, (20, disp.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        if disp.shape[1] > 1280:
            disp = cv2.resize(disp, (1280, int(disp.shape[0] * 1280 / disp.shape[1])))
        cv2.imshow("collect - whole frame is used", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            recording = None
        elif chr(key) in CLASSES:
            recording = None if recording == chr(key) else chr(key)

    cap.release()
    cv2.destroyAllWindows()
    print("Final counts:", counts)


if __name__ == "__main__":
    main()