"""
train.py - train FingerNet on the images from collect_data.py.

  python train.py                          # random 15% of data/ used for validation
  python train.py --val-dir val_data       # better: a separately recorded session
  python train.py --width 8 --depth 3      # try different widths and depths and compare!

Outputs: fingernet.pt (PyTorch checkpoint) and fingernet.onnx (portable, for the UNO Q).

HOW THE NETWORK LEARNS (one training step):
  1. FORWARD PASS: push a batch of images through the net to get 6 scores each.
  2. LOSS: cross-entropy measures how wrong those scores are. It is small
     when the net gives high probability to the correct finger count.
  3. BACKPROPAGATION (loss.backward()): use the chain rule to work backwards
     from the loss through every layer. This computes dLoss/dWeight for
     EVERY weight, i.e. "if I nudge this weight up, does the loss go up or
     down, and by how much?"
  4. GRADIENT DESCENT (optimizer.step()): move each weight a small step
     downhill, against its gradient:  w = w - learning_rate * gradient
     "Stochastic" means each step uses a small random batch instead of the
     whole dataset, which is faster and the noise even helps.
  Repeat for thousands of batches and the loss keeps going down.
"""
import argparse
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from common import IMG_H, IMG_W, eval_tf, train_tf
from model import FingerNet, count_params


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()                                   # dropout off, batchnorm uses running stats
    conf = torch.zeros(n_classes, n_classes, dtype=torch.long)
    loss_sum, total = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        total += y.size(0)
        for t, p in zip(y.cpu(), logits.argmax(1).cpu()):
            conf[t, p] += 1
    return loss_sum / total, conf.diag().sum().item() / total, conf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--val-dir", default=None)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate = gradient descent step size")
    p.add_argument("--optim", choices=["adam", "sgd"], default="adam")
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="fingernet")
    args = p.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    device = pick_device()
    print("Device:", device)

    # ---------- data ----------
    train_full = datasets.ImageFolder(args.data, transform=train_tf)
    classes = train_full.classes
    if args.val_dir:
        train_ds = train_full
        val_ds = datasets.ImageFolder(args.val_dir, transform=eval_tf)
        assert val_ds.classes == classes, "val folder classes don't match training folder classes"
    else:
        # Two views of the same folder: augmented for training, clean for validation.
        eval_full = datasets.ImageFolder(args.data, transform=eval_tf)
        idx = list(range(len(train_full)))
        random.shuffle(idx)
        n_val = int(len(idx) * args.val_frac)
        val_ds, train_ds = Subset(eval_full, idx[:n_val]), Subset(train_full, idx[n_val:])
        print("NOTE: frames recorded back-to-back look nearly identical, so a random split "
              "gives an optimistic accuracy. Use --val-dir with a separate session for an honest number.")

    counts = [0] * len(classes)
    for _, label in train_full.samples:
        counts[label] += 1
    print("Images per class:", dict(zip(classes, counts)))
    print(f"Train {len(train_ds)} | Val {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    # ---------- model, loss, optimizer ----------
    model = FingerNet(len(classes), args.width, args.depth, IMG_H, IMG_W).to(device)
    print(f"FingerNet width={args.width} depth={args.depth}: {count_params(model):,} parameters")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    if args.optim == "sgd":
        # Classic gradient descent plus momentum (a "rolling ball" that smooths noisy steps).
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr * 10, momentum=0.9, weight_decay=1e-4)
    else:
        # Adam is gradient descent that adapts the step size for each weight.
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Gradually lower the learning rate: big steps early, fine steps at the end.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---------- training loop ----------
    best_acc = -1.0
    print(f"\n{'epoch':>5} {'train_loss':>10} {'train_acc':>9} {'val_loss':>9} {'val_acc':>8}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)                 # 1. forward pass
            loss = criterion(logits, y)       # 2. how wrong are we?
            optimizer.zero_grad()             #    clear the previous step's gradients
            loss.backward()                   # 3. BACKPROPAGATION: chain rule fills every weight's .grad
            optimizer.step()                  # 4. GRADIENT DESCENT: step each weight downhill

            run_loss += loss.item() * y.size(0)
            run_correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
        scheduler.step()

        val_loss, val_acc, _ = evaluate(model, val_loader, device, len(classes))
        flag = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes,
                        "width": args.width, "depth": args.depth, "img_h": IMG_H, "img_w": IMG_W},
                       f"{args.out}.pt")
            flag = "  <- saved"
        print(f"{epoch:>5} {run_loss/seen:>10.4f} {run_correct/seen:>9.3f} {val_loss:>9.4f} {val_acc:>8.3f}{flag}")

    # ---------- final report ----------
    ckpt = torch.load(f"{args.out}.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    _, acc, conf = evaluate(model, val_loader, device, len(classes))
    print(f"\nBest validation accuracy: {acc:.3f}")
    print("Confusion matrix (rows = true count, cols = predicted):")
    print("     " + " ".join(f"{c:>5}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c:>4} " + " ".join(f"{v:>5}" for v in conf[i].tolist()))

    # ---------- export for the UNO Q ----------
    try:
        model.cpu().eval()
        dummy = torch.randn(1, 1, IMG_H, IMG_W)
        torch.onnx.export(model, dummy, f"{args.out}.onnx", input_names=["image"],
                          output_names=["logits"], opset_version=18)
        print(f"Exported {args.out}.onnx")
    except Exception as e:
        print(f"ONNX export skipped ({e}). Try: pip install onnx onnxscript")


if __name__ == "__main__":
    main()