# FingerLock app

Press Start, hold up three numbers to the camera, and the safe opens if the code is right.
The web page shows the neural network firing live.

```
fingerlock_app/
  app.yaml               App Lab descriptor (exposes port 7000)
  python/main.py         camera + network + lock logic + web server
  python/preprocess.py   torch-free copy of the training preprocessing
  python/web/index.html  the web page
  python/model/          put fingernet_viz.onnx + fingernet_viz.json here
  python/requirements.txt
  sketch/sketch.ino      STM32: motor, status LEDs, failsafe
```

## 1. Export your model (laptop, training folder)

    pip install onnxruntime
    python export_for_board.py --model fingernet.pt --out ..\fingerlock_app\python\model

Both checks at the end should say OK.

## 2. Test on the laptop (no board needed)

    cd fingerlock_app\python
    python main.py --camera 0

Open http://localhost:7000. Motor commands are printed in the terminal instead of moving anything.

## 3. Move it to the UNO Q

1. In App Lab, create a new app called FingerLock.
2. Copy in `python/` (with `model/` and `web/`) and replace `sketch/sketch.ino`.
3. Add the `ports: [7000]` line from this app.yaml to the generated app.yaml.
4. Plug the webcam into a powered USB-C hub, then press Run.
5. Open http://<board-name>.local:7000 from a device on the same network.

## Settings

- Passcode and timing: top of `python/main.py` (`PASSCODE = "253"`).
- Motor pins, speed and run time: top of `sketch/sketch.ino`.
