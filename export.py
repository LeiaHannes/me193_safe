"""
export_for_board.py - package a trained model for the lock app.

  python export_for_board.py --model fingernet.pt --out ..\\fingerlock_app\\python\\model

Run it from your training folder (it needs model.py and common.py).
Writes into --out:
  fingernet_viz.onnx  : the network, with EXTRA outputs for the visualization
                        (every conv block's activations and the FC layer).
                        Normal ONNX can't use PyTorch hooks, so we bake them in.
  fingernet_viz.json  : class names, image sizes, layer shapes, output weights.

Then it checks two things, so problems show up here and not at the safe:
  1. ONNX Runtime gives the same answers as PyTorch.
  2. The app's torch-free preprocessing matches eval_tf exactly.
"""
import argparse
import copy
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from common import IMG_H, IMG_W, SAVE_H, SAVE_W, eval_tf
from model import FingerNet


class VizNet(nn.Module):
    """Same network, but forward() also returns what each layer produced."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        acts = []
        for layer in self.net.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                acts.append(x)            # a conv block "firing"
        h, fc = x, None
        for layer in self.net.classifier:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                fc = h                    # the 64 fully connected neurons
        return (h, fc, *acts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="fingernet.pt")
    p.add_argument("--out", default=os.path.join("..", "fingerlock_app", "python", "model"))
    p.add_argument("--check-dir", default="test_data", help="folder of images for the checks")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ckpt = torch.load(args.model, map_location="cpu")
    classes = ckpt["classes"]
    net = FingerNet(len(classes), ckpt["width"], ckpt["depth"], ckpt["img_h"], ckpt["img_w"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    if (ckpt["img_h"], ckpt["img_w"]) != (IMG_H, IMG_W):
        sys.exit(f"Checkpoint was trained at {ckpt['img_h']}x{ckpt['img_w']} but common.py says "
                 f"{IMG_H}x{IMG_W}. Fix common.py to match before exporting.")

    viz = VizNet(copy.deepcopy(net)).eval()
    for m in viz.modules():                 # in-place ReLUs would overwrite the outputs we keep
        if isinstance(m, nn.ReLU):
            m.inplace = False

    dummy = torch.randn(1, 1, IMG_H, IMG_W)
    with torch.no_grad():
        outs = viz(dummy)
    n_blocks = len(outs) - 2
    names = ["logits", "fc"] + [f"block{i + 1}" for i in range(n_blocks)]

    onnx_path = os.path.join(args.out, "fingernet_viz.onnx")
    torch.onnx.export(viz, dummy, onnx_path, input_names=["image"], output_names=names, opset_version=18)
    print("Wrote", onnx_path)

    meta = {
        "classes": classes,
        "no_hand_class": "0",
        "img_h": IMG_H, "img_w": IMG_W,
        "save_w": SAVE_W, "save_h": SAVE_H,
        "norm_mean": 0.5, "norm_std": 0.5,
        "width": ckpt["width"], "depth": ckpt["depth"],
        "layers": [{"name": f"block{i + 1}", "c": int(a.shape[1]), "h": int(a.shape[2]), "w": int(a.shape[3])}
                   for i, a in enumerate(outs[2:])],
        "fc_size": int(outs[1].shape[1]),
        # Output layer weights (classes x 64): used to draw which FC neurons pushed the answer
        "out_weights": [[round(float(v), 5) for v in row] for row in net.classifier[-1].weight.detach()],
        "source_checkpoint": os.path.basename(args.model),
    }
    meta_path = os.path.join(args.out, "fingernet_viz.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print("Wrote", meta_path)

    # ---------------- checks ----------------
    sys.path.insert(0, os.path.abspath(os.path.join(args.out, "..")))   # the app's python/ folder
    try:
        from preprocess import preprocess_saved_image
    except ImportError:
        print("! Could not import the app's preprocess.py - put it next to the model folder to run checks.")
        return

    files = sorted(glob.glob(os.path.join(args.check_dir, "*", "*.jpg")))[::25][:40]
    if not files:
        print(f"! No images found in {args.check_dir}/ - skipping checks.")
        return

    worst_pre = 0.0
    for fpath in files:
        torch_x = eval_tf(Image.open(fpath).convert("RGB")).numpy()[None]
        app_x = preprocess_saved_image(fpath, meta)
        worst_pre = max(worst_pre, float(np.abs(torch_x - app_x).max()))
    print(f"Preprocessing check: max difference {worst_pre:.6f}  "
          f"({'OK' if worst_pre < 1e-4 else 'MISMATCH - the app would see different images!'})")

    try:
        import onnxruntime as ort
    except ImportError:
        print("! onnxruntime not installed (pip install onnxruntime) - skipping ONNX check.")
        return
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    agree, worst = 0, 0.0
    for fpath in files:
        x = preprocess_saved_image(fpath, meta).astype(np.float32)
        with torch.no_grad():
            t_logits = net(torch.from_numpy(x)).numpy()
        o_logits = sess.run(["logits"], {"image": x})[0]
        worst = max(worst, float(np.abs(t_logits - o_logits).max()))
        agree += int(t_logits.argmax() == o_logits.argmax())
    print(f"ONNX check: {agree}/{len(files)} predictions identical, max logit difference {worst:.6f}  "
          f"({'OK' if agree == len(files) and worst < 1e-3 else 'CHECK THIS'})")


if __name__ == "__main__":
    main()