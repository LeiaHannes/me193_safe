"""
preprocess.py - turn a camera frame into the network's input, WITHOUT torch.

The board doesn't need PyTorch, so this file rebuilds the training
preprocessing with OpenCV + Pillow + numpy, step for step:

  collect_data.py : full frame --cv2.INTER_AREA--> 320x180 (saved to disk)
  common.eval_tf  : Grayscale (PIL convert "L") -> Resize to 160x96 (PIL bilinear)
                    -> ToTensor (/255) -> Normalize(0.5, 0.5)

export_for_board.py checks that this produces the same numbers as eval_tf.
"""
import cv2
import numpy as np
from PIL import Image


def crop_16x9(frame):
    """Center-crop to 16:9 so a 4:3 camera mode can't squash the hand.
    (All training images were 16:9.)"""
    h, w = frame.shape[:2]
    target_h = int(round(w * 9 / 16))
    if target_h <= h:
        y0 = (h - target_h) // 2
        return frame[y0:y0 + target_h]
    target_w = int(round(h * 16 / 9))
    x0 = (w - target_w) // 2
    return frame[:, x0:x0 + target_w]


def preprocess(frame_bgr, meta):
    """Returns (x, seen):
       x    : float32 array 1x1xHxW, exactly what the network expects
       seen : uint8 HxW grayscale image (for the "what the net sees" view)"""
    frame = crop_16x9(frame_bgr)
    small = cv2.resize(frame, (meta["save_w"], meta["save_h"]), interpolation=cv2.INTER_AREA)
    pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).convert("L")
    pil = pil.resize((meta["img_w"], meta["img_h"]), Image.BILINEAR)
    seen = np.asarray(pil, dtype=np.uint8)
    x = (seen.astype(np.float32) / 255.0 - meta["norm_mean"]) / meta["norm_std"]
    return x[None, None], seen


def preprocess_saved_image(path, meta):
    """Same pipeline for an already-saved 320x180 training image (used for checks)."""
    pil = Image.open(path).convert("RGB").convert("L")
    pil = pil.resize((meta["img_w"], meta["img_h"]), Image.BILINEAR)
    g = np.asarray(pil, dtype=np.float32) / 255.0
    return ((g - meta["norm_mean"]) / meta["norm_std"])[None, None]
