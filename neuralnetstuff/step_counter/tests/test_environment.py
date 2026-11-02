"""M0 smoke tests: the venv has everything the project needs, and autograd works."""

import sys

import torch


def test_python_version():
    assert sys.version_info >= (3, 11)


def test_project_imports():
    import fastapi  # noqa: F401
    import numpy  # noqa: F401
    import paho.mqtt.client  # noqa: F401
    import scipy.signal  # noqa: F401
    import uvicorn  # noqa: F401
    import yaml  # noqa: F401

    import stepnet

    assert stepnet.__version__


def test_conv1d_forward_and_backward():
    # Same shape as our real input: batch of 2 windows, 6 IMU channels, 256 samples.
    x = torch.randn(2, 6, 256)
    conv = torch.nn.Conv1d(6, 8, kernel_size=7, padding=3)
    y = conv(x)
    assert y.shape == (2, 8, 256)

    y.sum().backward()
    assert conv.weight.grad is not None
    assert conv.weight.grad.abs().sum() > 0
