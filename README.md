# FingerLock

A safe that opens when you show the right three-number code with your fingers.
A convolutional neural network, built from scratch in PyTorch, counts the fingers in each webcam frame.
The lock runs on an Arduino UNO Q, and its web page shows the network's neurons firing live.

**How it works:** press Start on the web page, then hold up three numbers (1–5) one at a time, lowering your hand between them.
If the code is right, the STM32 runs a DC motor to open the latch.

## Repo layout

```
me193_safe/            training (runs on a laptop)
  collect_data.py        record labeled webcam images (classes 0-5)
  model.py               FingerNet CNN
  train.py               training with validation, confusion matrix, ONNX export
  find_mistakes.py       copy misclassified images into folders
  live_test.py           live webcam prediction
  viz_live.py            live activation visualizer (OpenCV)
  export.py              package a model for the board
  common.py              shared image settings and preprocessing
fingerlock_app/        Arduino App Lab app (runs on the UNO Q)
  python/main.py         camera, inference, lock logic, web server
  python/web/index.html  web page and network visualization
  python/model/          exported model (fingernet_viz.onnx and .json)
  sketch/sketch.ino      motor, status LEDs, heartbeat failsafe
```

## The model

The network sees a 160×96 grayscale frame and passes it through 5 conv blocks (conv, BatchNorm, ReLU, max pool), with 16 → 256 filters.
Global average pooling and a 64-neuron dense layer then produce 6 outputs: no hand, or 1–5 fingers.
The model has about 410k parameters.
Width and depth were compared experimentally, and depth 5 handled the hard 3-vs-4 and 4-vs-5 cases best.

## Train (laptop)

```
pip install torch torchvision opencv-python onnx onnxscript onnxruntime
python collect_data.py --cam 0                  # also record val_data/ and test_data/ in separate sessions
python train.py --val-dir val_data --epochs 80 --depth 5 --out fingernet_v2
python find_mistakes.py --data test_data --model fingernet_v2.pt
python export.py --model fingernet_v2.pt --out ..\fingerlock_app\python\model
```

## Deploy (UNO Q)

Hardware: an Arduino UNO Q, the Arduino USB-C Hub (8-in-1) powered through its PD port, a USB webcam on the hub, and a DC motor on a DRV8833-style driver.
The driver's inputs go to D5/D6, and the motor needs its own supply with a shared ground.

```
scp -r fingerlock_app/python fingerlock_app/sketch arduino@<board-ip>:~/ArduinoApps/fingerlock/
ssh arduino@<board-ip>
arduino-app-cli app start /home/arduino/ArduinoApps/fingerlock
arduino-app-cli app logs /home/arduino/ArduinoApps/fingerlock --all
```

Then open `http://<board-ip>:7000`.
The passcode and timing settings are at the top of `python/main.py`, and the motor pins and run time are at the top of `sketch/sketch.ino`.