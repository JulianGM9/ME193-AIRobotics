"""Gap and restart counting for the live stream."""

from stepnet.io.stream import SeqTracker


def test_contiguous_stream_has_no_gaps():
    t = SeqTracker()
    assert t.update("r1", list(range(0, 10))) == 0
    assert t.update("r1", list(range(10, 20))) == 0
    assert (t.received, t.missing, t.restarts) == (20, 0, 0)


def test_lost_batch_counts_missing_samples():
    t = SeqTracker()
    t.update("r1", list(range(0, 10)))
    assert t.update("r1", list(range(20, 30))) == 10  # batch 10..19 lost in MQTT
    assert t.missing == 10


def test_gap_inside_a_batch():
    t = SeqTracker()
    assert t.update("r1", [0, 1, 2, 5, 6]) == 2  # MCU failed to read seq 3 and 4
    assert t.missing == 2


def test_restart_is_not_a_gap():
    t = SeqTracker()
    t.update("r1", list(range(500, 510)))
    assert t.update("r2", list(range(0, 10))) == 0  # new run id
    assert t.update("r2", list(range(0, 10))) == 0  # seq went backwards (sketch reset)
    assert (t.missing, t.restarts) == (0, 2)
