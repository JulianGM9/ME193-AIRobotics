"""Candidate detection, the count check, label versions, and the backend API."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from stepnet import labels as lab
from stepnet.io.session import write_session

GYRO_LSB = 16.4


def synthetic_session(tmp_path, n_strides=5):
    """20 s still with one 'walking' bout of n_strides swings on the L axis (sensor +z)."""
    n = 2000
    samples = np.zeros((n, 8), dtype=np.int64)
    samples[:, 0] = np.arange(1000, 1000 + n)
    samples[:, 1] = samples[:, 0] * 10
    samples[:, 2] = -2048  # standing: S = -x = +1 g
    t = np.arange(n) / 100
    walking = (t >= 5) & (t < 5 + n_strides)
    samples[:, 7] = np.round(walking * -250 * np.sin(np.pi * (t - 5)) ** 8 * GYRO_LSB)  # one dip per second
    meta = {
        "session": "S001", "protocol": "test", "recorded_at": "2026-11-02 12:00:00", "samples": n,
        "notes": "", "gyro_bias_dps": [0.0, 0.0, 0.0], "accel_calibration": None,
        "segments": [
            {"i": 2, "kind": "transition", "label": "to_start", "start_seq": 1000, "end_seq": 1400, "steps": None, "ok": True},
            {"i": 3, "kind": "bout", "label": "walk_normal", "start_seq": 1400, "end_seq": 1500 + 100 * n_strides + 100,
             "target_steps": 10, "steps": 10, "ok": True},
            {"i": 4, "kind": "hold", "label": "stand", "start_seq": 2400, "end_seq": 2900, "steps": 0, "ok": True},
        ],
    }
    write_session(tmp_path / "raw" / "S001", samples, meta)
    return samples, meta


def test_candidates_find_every_stride(tmp_path):
    samples, meta = synthetic_session(tmp_path)
    cands = lab.find_candidates(samples, meta)
    assert len(cands) == 5
    # each one sits at a dip (the middle of a 1 s swing): 5.5 s, 6.5 s, ...
    assert [round((c - 1000) / 100, 1) for c in cands] == [5.5, 6.5, 7.5, 8.5, 9.5]


@pytest.mark.parametrize("n_strides, status", [(5, "pass"), (4, "fail"), (6, "fail")])
def test_count_check(tmp_path, n_strides, status):
    _, meta = synthetic_session(tmp_path)
    strides = [1550 + 100 * k for k in range(n_strides)]
    bout = next(c for c in lab.check_segments(meta, strides) if c["i"] == 3)
    assert bout["status"] == status  # 10 steps: 5 strides pass (|10 - 10| = 0); 4 or 6 are off by 2


def test_odd_step_counts_allow_either_neighbour(tmp_path):
    _, meta = synthetic_session(tmp_path)
    meta["segments"][1]["steps"] = 11  # started and finished on the same foot
    for n in (5, 6):
        assert lab.check_segments(meta, [1550 + 100 * k for k in range(n)])[0]["status"] == "pass"


def test_a_stride_in_a_no_step_block_fails_until_excluded(tmp_path):
    _, meta = synthetic_session(tmp_path)
    strides = [1550 + 100 * k for k in range(5)] + [2500]
    assert not lab.all_pass(lab.check_segments(meta, strides))
    assert lab.all_pass(lab.check_segments(meta, strides, segment_ok={"4": False}))


def test_versions_and_refusal(tmp_path):
    _, meta = synthetic_session(tmp_path)
    labels_dir = tmp_path / "labels"
    good = [1550 + 100 * k for k in range(5)]
    with pytest.raises(ValueError, match="#3 walk_normal"):
        lab.save_labels("S001", meta, good[:3], labels_dir=labels_dir)
    assert lab.save_labels("S001", meta, good, labels_dir=labels_dir)["version"] == 1
    assert lab.save_labels("S001", meta, good, labels_dir=labels_dir)["version"] == 2
    assert lab.load_latest_labels("S001", labels_dir)["strides"] == good


def test_api(tmp_path):
    synthetic_session(tmp_path)
    client = TestClient(create_app(raw_dir=tmp_path / "raw", labels_dir=tmp_path / "labels"))

    listing = client.get("/api/sessions").json()
    assert listing == [{"session": "S001", "protocol": "test", "recorded_at": "2026-11-02 12:00:00",
                        "samples": 2000, "label_version": None, "strides": None}]

    data = client.get("/api/sessions/S001").json()
    assert len(data["seq"]) == 2000 and len(data["gyr"]) == 3 and len(data["candidates"]) == 5
    assert data["labels"] is None

    assert client.post("/api/sessions/S001/labels", json={"strides": data["candidates"][:2]}).status_code == 422
    saved = client.post("/api/sessions/S001/labels", json={"strides": data["candidates"]})
    assert saved.status_code == 200 and saved.json()["version"] == 1
    assert client.get("/api/sessions").json()[0]["label_version"] == 1

    assert client.get("/api/sessions/..%2Fx").status_code == 404
    assert client.get("/api/sessions/S999").status_code == 404
    assert "Step Counter" in client.get("/").text


def test_strides_before_go_are_not_counted(tmp_path):
    # Turning into position while the instruction plays (seq 1420) is not part of the bout.
    _, meta = synthetic_session(tmp_path)
    strides = [1420] + [1550 + 100 * k for k in range(5)]
    assert lab.check_segments(meta, strides)[0]["status"] == "fail"  # no Go recorded: 6 strides vs 10 steps
    meta["segments"][1]["go_seq"] = 1500
    bout = lab.check_segments(meta, strides)[0]
    assert (bout["strides"], bout["status"]) == (5, "pass")
