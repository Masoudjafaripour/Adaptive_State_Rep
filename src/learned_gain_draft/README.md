# Learned-gain draft (Acrobot)

Early prototype of the core idea: an `ActorNetwork` that outputs both an action and a sigmoid "Kalman gain" from compact state + raw pixels.

- `main.py`: MLP embedding of flattened 64x64 RGB frames.
- `main_CNN.py`: same, with a CNN image encoder.

**Status: not trained.** Both scripts set up an optimizer but never compute a loss or call `optimizer.step()`, and the predicted gain is not used to fuse anything. No results yet.
