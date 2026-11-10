"""
live_test.py - run the trained model on the live webcam.
  python live_test.py --cam 1
The top-left corner shows exactly what the network sees (160x96 grayscale).
"""
import argparse
from collections import deque

import cv2
import torch
import torch.nn.functional as F

from common import bgr_to_pil, eval_tf, open_camera
from model import FingerNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=1)
    p.add_argument("--model", default="fingernet.pt")
    p.add_argument("--smooth", type=int, default=5, help="average predictions over N frames")
    p.add_argument("--thresh", type=float, default=0.7, help="minimum confidence to show a number")
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location="cpu")
    classes = ckpt["classes"]
    model = FingerNet(len(classes), ckpt["width"], ckpt["depth"], ckpt["img_h"], ckpt["img_w"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    cap = open_camera(args.cam)
    history = deque(maxlen=args.smooth)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        x = eval_tf(bgr_to_pil(frame)).unsqueeze(0)      # shape 1x1x96x160
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[0]
        history.append(probs)
        avg = torch.stack(list(history)).mean(0)
        conf, idx = avg.max(0)

        if conf >= args.thresh:
            text = "no hand" if classes[idx] == "0" else f"{classes[idx]} fingers"
            color = (0, 255, 0)
        else:
            text, color = "unsure", (0, 165, 255)

        disp = frame.copy()
        cv2.putText(disp, f"{text} ({conf:.0%})", (360, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

        # What the net sees (un-normalized back to 0-255 so you can view it)
        seen = ((x[0, 0] * 0.5 + 0.5) * 255).byte().numpy()
        seen = cv2.cvtColor(cv2.resize(seen, (320, 192), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        disp[10:202, 10:330] = seen

        # Per-class probability bars
        for i, c in enumerate(classes):
            y = 240 + i * 30
            cv2.putText(disp, c, (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.rectangle(disp, (35, y), (35 + int(avg[i] * 200), y + 20), (255, 200, 0), -1)

        if disp.shape[1] > 1280:
            disp = cv2.resize(disp, (1280, int(disp.shape[0] * 1280 / disp.shape[1])))
        cv2.imshow("FingerNet live (q to quit)", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()