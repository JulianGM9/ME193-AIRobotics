"""Recording protocols (PLAN.md D7, D8).

Step kinds:
  calibrate   stand still at the laptop: gyro bias + orientation check
  bout        take target_steps steps (both feet), then stop and stand still. Starts at the
              spoken cue and ends automatically once the leg has been still for a moment.
              Ground truth = the step count you confirm after the session.
  hold        a timed block with no steps (standing, fidgeting, leg swings...): truth = 0
  transition  timed time to get into position; never used for training

A step with `optional` set (e.g. "stairs") is only recorded if enabled for the session.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    kind: str
    label: str
    say: str
    seconds: float = 0.0
    target_steps: int | None = None  # None on a bout = count the steps yourself
    optional: str | None = None


def _walk(label, steps, pace):
    return Step("bout", label, f"Walk {steps} steps at a {pace} pace, then stop and stand still. "
                               "Turn around at the end of the hallway if you need to.", target_steps=steps)


CALIBRATE = Step("calibrate", "calibrate", "Stand up straight and completely still.", seconds=5)
TO_START = Step("transition", "to_start", "Walk to the start of the hallway and wait there.", seconds=15)

PROTOCOLS = {
    # Movement-focused (~4-5 min)
    "A": [
        CALIBRATE,
        TO_START,
        Step("hold", "stand", "Stand still.", seconds=10),
        _walk("walk_slow", 30, "slow"),
        _walk("walk_normal", 40, "normal"),
        _walk("walk_fast", 40, "fast"),
        Step("bout", "run", "Jog 40 steps, then stop and stand still.", target_steps=40),
        Step("bout", "shuffle", "Shuffle 30 small steps without lifting your feet much, then stop.",
             target_steps=30),
        Step("transition", "to_stairs", "Go to the bottom of the stairs and wait there for the next instruction.",
             seconds=30, optional="stairs"),
        Step("bout", "stairs_up", "Walk up the stairs, then stop at the top and stand still. Count your steps.",
             optional="stairs"),
        Step("bout", "stairs_down", "Walk down the stairs, then stop and stand still. Count your steps.",
             optional="stairs"),
        _walk("walk_normal_2", 40, "normal"),
        Step("hold", "stand", "Stand still.", seconds=10),
    ],
    # Hard non-steps (~4 min): things that move the ankle without being a step
    "B": [
        CALIBRATE,
        TO_START,
        Step("hold", "stand", "Stand still.", seconds=15),
        Step("hold", "fidget_standing", "Fidget while standing: shift your weight, rock on your heels, "
                                        "but keep your feet in place.", seconds=30),
        _walk("walk_normal", 40, "normal"),
        Step("transition", "to_chair", "Sit down on a chair.", seconds=10),
        Step("hold", "leg_swing_seated", "Swing your lower leg back and forth.", seconds=30),
        Step("hold", "foot_tap_seated", "Tap your foot.", seconds=30),
        Step("hold", "sit_still", "Sit still.", seconds=10),
        Step("transition", "stand_up", "Stand up.", seconds=8),
        _walk("walk_slow", 30, "slow"),
        Step("hold", "stand", "Stand still.", seconds=10),
    ],
    # ~1.5 minute dry run (saved as T###, never trained on)
    "test": [
        CALIBRATE,
        TO_START,
        _walk("walk_normal", 10, "normal"),
        Step("hold", "fidget_standing", "Fidget while standing, feet in place.", seconds=10),
        Step("hold", "stand", "Stand still.", seconds=5),
    ],
}

BOUT_TIMEOUT_S = 90.0  # a bout ends after this even if the leg never goes still


def steps_for(protocol: str, enabled=()) -> list[Step]:
    """The protocol's steps, minus optional ones not enabled for this session."""
    return [s for s in PROTOCOLS[protocol] if s.optional is None or s.optional in enabled]
