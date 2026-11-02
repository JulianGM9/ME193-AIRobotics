# Ankle IMU Step Counter

An Arduino UNO Q and an MPU-6050 are worn on the ankle. A PyTorch neural network on the laptop counts steps and reports each count with a calibrated confidence interval. The design and its status are in [`PLAN.md`](PLAN.md).

## Setup

From this folder:
```
../../.venv/bin/pip install -r requirements.txt
../../.venv/bin/python -m pytest
```
