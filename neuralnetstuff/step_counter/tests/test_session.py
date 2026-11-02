"""Session files, gap filling, motion detection and the protocols."""

import json

import numpy as np
import pytest

from stepnet.io.session import fill_gaps, missing_seqs, next_session_id, read_session, write_session
from stepnet.motion import is_moving, is_still
from stepnet.protocols import PROTOCOLS, steps_for


def make_samples(seqs, gyro_x=0):
    seqs = np.asarray(seqs)
    out = np.zeros((len(seqs), 8), dtype=np.int64)
    out[:, 0] = seqs
    out[:, 1] = seqs * 10
    out[:, 2] = -2048
    out[:, 5] = gyro_x
    return out


def test_write_read_round_trip(tmp_path):
    samples = make_samples(range(100, 110))
    write_session(tmp_path / "S001", samples, {"session": "S001", "segments": []})
    got, meta = read_session(tmp_path / "S001")
    assert np.array_equal(got, samples)
    assert meta["session"] == "S001"


def test_never_overwrites_a_session(tmp_path):
    write_session(tmp_path / "S001", make_samples([1]), {})
    with pytest.raises(FileExistsError):
        write_session(tmp_path / "S001", make_samples([1]), {})


def test_next_session_id(tmp_path):
    assert next_session_id("S", tmp_path) == "S001"
    (tmp_path / "S001").mkdir()
    (tmp_path / "S007").mkdir()
    (tmp_path / "T002").mkdir()
    assert next_session_id("S", tmp_path) == "S008"
    assert next_session_id("T", tmp_path) == "T003"


def test_fill_gaps_from_board_log(tmp_path):
    session = tmp_path / "S001"
    received = make_samples([0, 1, 2, 6, 7, 8])  # 3, 4, 5 lost over MQTT
    write_session(session, received, {"missing": 3})
    board_csv = tmp_path / "stream_x.csv"
    np.savetxt(board_csv, make_samples(range(0, 20)), fmt="%d", delimiter=",", header="h", comments="")

    assert missing_seqs(received).tolist() == [3, 4, 5]
    assert fill_gaps(session, board_csv) == 3
    samples, meta = read_session(session)
    assert samples[:, 0].tolist() == list(range(0, 9))  # filled, but nothing added past the session
    assert meta["missing"] == 0 and meta["filled_from_board"] == 3


def test_motion_detection():
    lsb = 16.4  # raw counts per deg/s
    bias = [-7.9, 1.3, 0.0]
    still = make_samples(range(150), gyro_x=int(-7.9 * lsb))  # only the bias
    walking = make_samples(range(150), gyro_x=int(250 * lsb))
    assert is_still(still, bias) and not is_moving(still, bias)
    assert is_moving(walking, bias) and not is_still(walking, bias)
    assert not is_still(still[:0], bias)  # no data is not "still"


def test_protocols_are_well_formed():
    for name, steps in PROTOCOLS.items():
        assert steps[0].kind == "calibrate", name
        for s in steps:
            assert s.kind in {"calibrate", "bout", "hold", "transition"}
            if s.kind in {"hold", "transition", "calibrate"}:
                assert s.seconds > 0, s.label
            if s.kind == "bout" and s.optional is None:
                assert s.target_steps, s.label
    assert any(s.optional == "stairs" for s in PROTOCOLS["A"])
    assert all(s.optional is None for s in steps_for("A"))
    assert len(steps_for("A", {"stairs"})) == len(PROTOCOLS["A"])
