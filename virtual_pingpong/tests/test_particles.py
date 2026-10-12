import math

from render.particles import ParticleBurst


class FixedRng:
    """A fake rng that always returns a chosen value, for deterministic
    particle positions in tests (same injection pattern as game/opponent.py)."""

    def __init__(self, value):
        self.value = value

    def uniform(self, lo, hi):
        return self.value


class SequenceRng:
    """A fake rng returning successive values from a fixed list, for tests
    that need to control angle and speed separately (trigger() calls
    uniform() for angle first, then speed, per particle)."""

    def __init__(self, values):
        self.values = list(values)

    def uniform(self, lo, hi):
        return self.values.pop(0)


def test_trigger_creates_the_requested_number_of_particles():
    burst = ParticleBurst(num_particles=10)
    burst.trigger(0, 0, now=0.0, rng=FixedRng(0.0))
    assert len(burst.particles) == 10


def test_not_active_before_any_trigger():
    burst = ParticleBurst()
    assert burst.is_active(now=0.0) is False
    assert burst.positions(now=0.0) == []


def test_active_right_after_trigger():
    burst = ParticleBurst(lifetime_seconds=0.6)
    burst.trigger(0, 0, now=5.0, rng=FixedRng(0.0))
    assert burst.is_active(now=5.0) is True


def test_inactive_after_lifetime_elapses():
    burst = ParticleBurst(lifetime_seconds=0.6)
    burst.trigger(0, 0, now=0.0, rng=FixedRng(0.0))
    assert burst.is_active(now=0.6) is False
    assert burst.positions(now=0.6) == []


def test_fade_is_one_right_when_triggered():
    burst = ParticleBurst(lifetime_seconds=1.0)
    burst.trigger(0, 0, now=0.0, rng=FixedRng(0.0))
    assert burst.fade(now=0.0) == 1.0


def test_fade_decays_linearly_to_zero():
    burst = ParticleBurst(lifetime_seconds=1.0)
    burst.trigger(0, 0, now=0.0, rng=FixedRng(0.0))
    assert burst.fade(now=0.5) == 0.5
    assert burst.fade(now=1.0) == 0.0


def test_fade_zero_before_any_trigger():
    burst = ParticleBurst()
    assert burst.fade(now=0.0) == 0.0


def test_particle_moves_outward_at_its_assigned_velocity():
    # First uniform() call is the angle (0 -> straight along +x), second
    # is the speed (100).
    burst = ParticleBurst(num_particles=1)
    burst.trigger(10.0, 20.0, now=0.0, rng=SequenceRng([0.0, 100.0]))
    x, y = burst.positions(now=0.5)[0]
    assert math.isclose(x, 10.0 + 100 * 0.5)  # moved 50 units along +x
    assert math.isclose(y, 20.0, abs_tol=1e-9)  # no y movement at angle=0


def test_retriggering_replaces_the_previous_burst():
    burst = ParticleBurst(num_particles=5)
    burst.trigger(0, 0, now=0.0, rng=FixedRng(0.0))
    burst.trigger(1, 1, now=1.0, rng=FixedRng(0.0))
    assert burst.triggered_at == 1.0
    x, y = burst.positions(now=1.0)[0]
    assert (x, y) == (1, 1)
