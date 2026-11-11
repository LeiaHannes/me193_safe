"""
find_mistakes.py - copy every image the model gets wrong into folders you can browse.

  python find_mistakes.py --data val_data

Creates  mistakes/true4_pred5/0.87_<original name>.jpg  and so on.
The number at the front is the model's confidence in its WRONG answer,
so the images it was most sure about sort to the bottom of each folder.
"""
import argparse
import os
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from common import eval_tf
from model import FingerNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="val_data")
    p.add_argument("--model", default="fingernet.pt")
    p.add_argument("--out", default="mistakes")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    classes = ckpt["classes"]
    model = FingerNet(len(classes), ckpt["width"], ckpt["depth"], ckpt["img_h"], ckpt["img_w"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = datasets.ImageFolder(args.data, transform=eval_tf)
    assert ds.classes == classes, "folder classes don't match the model's classes"
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    # Clear only the previous run's true*_pred* folders, nothing else.
    if os.path.isdir(args.out):
        for d in os.listdir(args.out):
            if d.startswith("true") and "_pred" in d:
                shutil.rmtree(os.path.join(args.out, d))

    counts, idx = {}, 0
    with torch.no_grad():
        for x, y in loader:
            probs = F.softmax(model(x.to(device)), dim=1).cpu()
            conf, pred = probs.max(1)
            for i in range(len(y)):
                path = ds.samples[idx + i][0]
                t, pr = classes[y[i]], classes[pred[i]]
                if t != pr:
                    folder = f"true{t}_pred{pr}"
                    os.makedirs(os.path.join(args.out, folder), exist_ok=True)
                    name = f"{conf[i].item():.2f}_{os.path.basename(path)}"
                    shutil.copy(path, os.path.join(args.out, folder, name))
                    counts[folder] = counts.get(folder, 0) + 1
            idx += len(y)

    total = sum(counts.values())
    print(f"{total} of {len(ds)} images wrong ({1 - total / len(ds):.1%} accuracy)")
    for folder, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {folder:<14} {n}")
    print(f"Browse them in: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()