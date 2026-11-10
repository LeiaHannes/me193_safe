"""
model.py - FingerNet, a small convolutional neural network built from scratch.

WIDTH and DEPTH
  * DEPTH = how many layers are stacked. Here it's the number of conv blocks.
    Each block can build more abstract features from the block before it:
      block 1: edges and blobs of light and dark
      block 2: curves, finger outlines
      block 3: whole fingers, the gaps between them
      block 4: "this looks like three raised fingers"
  * WIDTH = how many neurons (here, filters/channels) are in each layer.
    More filters means each layer can detect more different patterns at once.
    We start at `width` filters and double them in every block.
  Bigger isn't automatically better. More width and depth means more
  parameters, which needs more data, risks memorizing the training set
  (overfitting), and runs slower on the UNO Q.

CONVOLUTION
  A 3x3 filter slides across the image. At each position it multiplies its
  9 weights by the 9 pixels under it and adds them up. The SAME weights are
  reused at every position (weight sharing), so a "fingertip detector"
  works wherever the fingertip appears. It also needs far fewer parameters
  than a fully connected layer would.

POOLING
  MaxPool2d(2) keeps the largest value in each 2x2 patch, which halves the
  height and width. That (a) cuts computation, (b) makes the net tolerant
  of small shifts, and (c) lets later 3x3 filters "see" a bigger piece of
  the original image (a larger receptive field).

Shape trace with the defaults (width=16, depth=4, 96x160 input, written CxHxW):
  1x96x160 -> 16x48x80 -> 32x24x40 -> 64x12x20 -> 128x6x10
  -> global average pool 128 -> 64 -> 6 scores (one per class).
  About 106k parameters, roughly 0.4 MB.
"""
import torch
import torch.nn as nn


class FingerNet(nn.Module):
    def __init__(self, num_classes=6, width=16, depth=4, img_h=96, img_w=160, dropout=0.3):
        super().__init__()
        assert img_h % (2 ** depth) == 0 and img_w % (2 ** depth) == 0, \
            "image height and width must be divisible by 2**depth"

        layers = []
        in_ch, out_ch = 1, width            # 1 input channel = grayscale
        for _ in range(depth):
            layers += [
                # Convolution: learns out_ch different 3x3 filters.
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                # BatchNorm keeps activations in a sensible range, so training is faster and more stable.
                nn.BatchNorm2d(out_ch),
                # ReLU nonlinearity. Without it, stacked layers would collapse into one linear function.
                nn.ReLU(inplace=True),
                # Pooling: halve the spatial size.
                nn.MaxPool2d(2),
            ]
            in_ch, out_ch = out_ch, out_ch * 2
        # GLOBAL AVERAGE POOLING: average each final channel over the whole image,
        # turning 128 x 6 x 10 into just 128 numbers. If a channel has learned
        # "fingertip here", its average rises with the number of fingertips, which
        # suits counting. It also replaces a huge 7680->64 layer (about 490k weights)
        # that could get stuck with all its ReLUs "dead" (outputting 0, passing no gradient).
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Dropout randomly zeroes neurons during training, which fights overfitting.
            nn.Dropout(dropout),
            nn.Linear(in_ch, 64),   # fully connected layer (128 -> 64)
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),   # raw scores ("logits"); softmax turns them into probabilities
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Run `python model.py` to see the shapes flow through the network.
    m = FingerNet().eval()
    x = torch.randn(1, 1, 96, 160)
    print(f"input           {tuple(x.shape)}")
    block = 0
    for layer in m.features:
        x = layer(x)
        if isinstance(layer, nn.MaxPool2d):
            block += 1
            print(f"after block {block}   {tuple(x.shape)}")
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            print(f"global avg pool {tuple(x.shape)}")
    print(f"output          {tuple(m.classifier(x).shape)}")
    print(f"trainable params: {count_params(m):,}")