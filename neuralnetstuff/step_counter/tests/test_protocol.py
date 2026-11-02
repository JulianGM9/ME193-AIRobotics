"""Wire format, and that the board relay and the laptop agree on it."""

import re
from pathlib import Path

import numpy as np
import pytest

from stepnet.io import protocol as p

BOARD_MAIN = Path(__file__).resolve().parents[1] / "board" / "python" / "main.py"


def test_parse_batch():
    rows = p.parse_batch("350,3947,-2069,-106,305,-245,-204,364;351,3957,-2056,-190,328,-231,-219,379")
    assert rows.shape == (2, 8)
    assert rows[0].tolist() == [350, 3947, -2069, -106, 305, -245, -204, 364]


def test_format_batch_round_trip():
    samples = np.array([[0, 0, -2048, 1, 2, -3, 4, 5], [1, 10, -2047, 0, 0, 0, 0, 0]])
    assert np.array_equal(p.parse_batch(p.format_batch(samples)), samples)


@pytest.mark.parametrize("bad", ["", "1,2,3", "1,2,3,4,5,6,7,x", "1,2,3,4,5,6,7,8,9"])
def test_parse_batch_rejects_malformed(bad):
    with pytest.raises(ValueError):
        p.parse_batch(bad)


def test_board_uses_same_broker_and_topics():
    # The board app can't import stepnet (it is pushed to the board on its own),
    # so it repeats these constants. This test catches them drifting apart.
    src = BOARD_MAIN.read_text()
    assert re.search(r'^BROKER = "([^"]+)"', src, re.M).group(1) == p.BROKER
    assert re.search(r'^BASE_TOPIC = "([^"]+)"', src, re.M).group(1) == "ME193/Julian/StepCounter"
    assert '"ME193/Julian/StepCounter")' in (Path(p.__file__)).read_text()
    for suffix in ("/imu", "/status", "/cmd"):
        assert f'BASE_TOPIC + "{suffix}"' in src
