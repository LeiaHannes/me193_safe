"""
common.py - settings and helpers shared by collect_data.py, train.py and live_test.py.

Why a shared file? Collection, training and live prediction must prepare
images in the same way. If they don't (different crop, color vs. gray,
different size), a model that scores 98% in training can fail on the
live camera. Keeping all of it in one place prevents that.
"""
import sys

import cv2
from PIL import Image
from torchvision import transforms

# The network sees the WHOLE camera frame, shrunk to IMG_W x IMG_H grayscale.
# That's roughly the camera's 16:9 shape, so fingers aren't squashed sideways.
# Both numbers must be divisible by 2**depth (16 with the default depth of 4).
IMG_W, IMG_H = 160, 96
SAVE_W, SAVE_H = 320, 180   # frames are saved larger, so you can change IMG_W/IMG_H later without re-collecting
CLASSES = ["0", "1", "2", "3", "4", "5"]   # "0" = no hand / fist


def open_camera(index, width=1280, height=720):
    """Open the USB webcam. Your laptop's built-in camera is usually index 0,
    so the external webcam is often 1 (try 2 if that's wrong).

    We ask for 1280x720 instead of the full 1920x1080. The image is
    shrunk to 64x64 anyway, so the extra pixels only slow things down.
    MJPG mode lets most USB webcams run at full frame rate."""
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}. Try --cam 0, 1 or 2.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {index} opened at {w}x{h}")
    return cap


def bgr_to_pil(img_bgr):
    """OpenCV gives BGR numpy arrays, but torchvision transforms expect PIL images."""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# Pixels 0..1 become -1..1. Centered inputs make gradient descent behave better.
NORMALIZE = transforms.Normalize(mean=[0.5], std=[0.5])

# Training transforms include DATA AUGMENTATION: random changes to each image
# every epoch, so the net learns "count fingers" instead of memorizing
# exact pictures.
train_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.RandomHorizontalFlip(),          # left hand vs right hand: same count
    # Small shift/zoom/rotate. Kept gentle on purpose: an aggressive random crop
    # can cut a finger out of the frame, turning a "5" into what looks like a "4"
    # while the label still says 5.
    transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.92, 1.05)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor(),
    NORMALIZE,
])

# Evaluation and live transforms: no randomness, just the same basic prep.
eval_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    NORMALIZE,
])