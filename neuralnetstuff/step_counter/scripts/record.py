"""Record one labeled session (milestone M2, PLAN.md D7-D9).

Run from neuralnetstuff/step_counter/ with the board app running:
  ../../.venv/bin/python scripts/record.py --protocol test          # ~1.5 min dry run -> T###
  ../../.venv/bin/python scripts/record.py --protocol A --stairs    # movement session -> S###
  ../../.venv/bin/python scripts/record.py --protocol B             # hard non-steps   -> S###

  1. At the laptop: a 5 s standing calibration measures the gyro bias and checks the mount.
  2. Away from the laptop: spoken cues guide every block. A walking bout ends by itself
     once you stop and stand still, so you never have to touch the laptop.
  3. Back at the laptop: confirm how many steps (both feet) you took in each bout.
  4. Saves data/raw/<id>/imu.csv + meta.json.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stepnet import calibration as cal
from stepnet.io.live import LiveStream
from stepnet.io.session import RAW_DIR, missing_seqs, next_session_id, read_session, write_session
from stepnet.motion import MOVING_DPS, gyro_magnitude_dps, is_still
from stepnet.preprocess import AXIS_MAP, to_body, to_physical
from stepnet.protocols import BOUT_TIMEOUT_S, steps_for

# Samples reach the laptop ~0.3 s after they happen (MQTT round trip / 2 + the relay's
# 0.25 s grouping), so a cue given "now" lines up with a sample ~30 seqs newer than the
# latest one received. Bout/hold boundaries only need to be this accurate: individual
# steps are labeled later from the signal itself.
LAG_SAMPLES = 30
STILL_WINDOW = 150  # 1.5 s of stillness ends a bout
END_MARGIN = 50  # keep 0.5 s of the stillness inside the bout
MIN_ACTIVE = 100  # a bout lasts at least 1 s of movement before stillness can end it
NUDGE_AFTER_S = 10.0

HALLWAY_M = 4.88  # 16 ft, measured by Julian 2026-11-02 (for distance estimation, M11)

MAX_ORIENTATION_DEG = 20.0  # gravity this far from +S means the board is mounted wrongly
MAX_ORIENTATION_CHANGE_DEG = 10.0  # ...or this far from last session's mount


class Voice:
    """Spoken cues with macOS `say`, plus a printed banner (works with --no-voice too)."""

    def __init__(self, enabled):
        self.enabled = enabled

    def say(self, text, wait=True):
        print(f"\n>>> {text.upper()}", flush=True)
        if self.enabled:
            proc = subprocess.Popen(["say", "-r", "185", text])
            if wait:
                proc.wait()

    def beep(self, sound="Ping"):
        if self.enabled:
            subprocess.Popen(["afplay", f"/System/Library/Sounds/{sound}.aiff"])


def latest(stream):
    return stream.latest_seq() + LAG_SAMPLES


def calibrate(stream, voice, accel_cal):
    """Stand still for 5 s at the laptop.

    Returns (gyro bias deg/s in sensor xyz, gravity g in S/F/L, tilt from +S in deg,
    change from the previous session's mount in deg or None).
    """
    while True:
        input("\nStand up straight next to the laptop, then press Enter and stay completely still...")
        voice.say("Calibrating. Stay still.", wait=False)
        raw = stream.collect(5.0)
        acc, gyr = to_physical(raw, accel_cal)
        bias = gyr.mean(axis=0)
        if not is_still(raw, bias):
            voice.say("You moved. Let's try that again.")
            continue
        gravity = to_body(acc).mean(axis=0)
        err = cal.orientation_error_deg(gravity)
        print(f"  gyro bias (sensor xyz) {np.round(bias, 2).tolist()} deg/s")
        print(f"  gravity (S, F, L)      {np.round(gravity, 3).tolist()} g   |g| = {np.linalg.norm(gravity):.3f}")
        print(f"  tilt from straight up the shin: {err:.1f} deg")
        problems = []
        if err > MAX_ORIENTATION_DEG:
            problems.append(f"tilt {err:.0f} deg > {MAX_ORIENTATION_DEG:.0f} - is the board mounted the right way up?")
        previous = previous_gravity()
        change = None
        if previous is not None:
            change = cal.angle_between_deg(gravity, previous[1])
            print(f"  change from {previous[0]}'s mount: {change:.1f} deg")
            if change > MAX_ORIENTATION_CHANGE_DEG:
                problems.append(f"mount moved {change:.0f} deg since {previous[0]} - re-strap like last time?")
        if not problems:
            voice.say("Calibration done.")
            return bias, gravity, err, change
        for p in problems:
            print(f"  ! {p}")
        if input("Re-strap and press Enter to redo, or type c to continue anyway: ").strip().lower() == "c":
            return bias, gravity, err, change


def previous_gravity():
    """(session id, gravity_body_g) of the latest real session, or None."""
    for session_dir in sorted(RAW_DIR.glob("S[0-9][0-9][0-9]"), reverse=True):
        try:
            meta = read_session(session_dir)[1]
        except (OSError, ValueError):
            continue
        if meta.get("gravity_body_g"):
            return meta["session"], meta["gravity_body_g"]
    return None


def run_bout(stream, step, bias, voice):
    # The bout starts when the instruction starts, not at "Go": in T001 the walking began
    # during the spoken instruction, and those strides fell outside the labeled bout. Any
    # extra stillness this adds to the bout is harmless.
    start = latest(stream)
    voice.say(step.say)
    voice.beep()
    voice.say("Go.", wait=False)
    # Only movement after "Go" starts the bout. Otherwise walking that was still going on
    # during the instruction (e.g. still on the way to the stairs) counted as the bout, and
    # pausing at the bottom of the stairs then ended it before the climb (2026-11-02).
    go_seq = latest(stream)
    t0 = time.monotonic()
    moved_seq = None  # first sample after "Go" with the leg clearly moving
    nudged = False
    while True:
        time.sleep(0.2)
        if moved_seq is None:
            after_go = stream.since(go_seq)
            moving = np.nonzero(gyro_magnitude_dps(after_go, bias) > MOVING_DPS)[0] if len(after_go) else []
            if len(moving):
                moved_seq = int(after_go[moving[0], 0])
            elif not nudged and time.monotonic() - t0 > NUDGE_AFTER_S:
                voice.say("Start whenever you are ready.", wait=False)
                nudged = True
        else:
            recent = stream.since(stream.latest_seq() - STILL_WINDOW + 1)
            if (len(recent) >= STILL_WINDOW * 0.9 and recent[0, 0] >= moved_seq + MIN_ACTIVE
                    and is_still(recent, bias)):
                end = int(recent[0, 0]) + END_MARGIN
                timed_out = False
                break
        if time.monotonic() - t0 > BOUT_TIMEOUT_S:
            end, timed_out = latest(stream), True
            break
    voice.beep("Glass")
    voice.say("Done." if not timed_out else "Time is up.", wait=False)
    return start, go_seq, end, timed_out


def run_timed(stream, step, voice):
    voice.say(step.say)
    voice.beep()
    start = latest(stream)
    for remaining in range(int(step.seconds), 0, -1):
        if remaining == 5 and step.seconds >= 15:
            voice.say("5 seconds.", wait=False)
        time.sleep(1.0)
    end = latest(stream)
    if step.kind == "hold":
        voice.beep("Glass")
        voice.say("Stop.", wait=False)
    return start, end


def confirm_counts(segments):
    """At the laptop again: the step count of every bout, plus accidental steps / bad blocks."""
    print("\n==== confirm step counts (both feet) ====")
    for seg in segments:
        if seg["kind"] != "bout":
            continue
        target = seg["target_steps"]
        prompt = f"  #{seg['i']:2d} {seg['label']:14s} target {target}: Enter = exactly {target}, or type the real count: " \
            if target is not None else f"  #{seg['i']:2d} {seg['label']:14s} how many steps? "
        while True:
            answer = input(prompt).strip()
            if answer == "" and target is not None:
                seg["steps"] = target
                break
            if answer.isdigit():
                seg["steps"] = int(answer)
                break
            print("    please type a whole number")

    holds = [s for s in segments if s["kind"] == "hold"]
    print("\nNo-step blocks: " + ", ".join(f"#{s['i']} {s['label']}" for s in holds))
    answer = input("Did you take any steps in one of them? Enter = no, or type the block number: ").strip()
    while answer.isdigit():
        seg = next((s for s in holds if s["i"] == int(answer)), None)
        if seg:
            seg["steps"] = int(input(f"  steps in #{seg['i']} {seg['label']}: ").strip() or 0)
        answer = input("Another one? Enter = no, or type the block number: ").strip()

    answer = input("\nAny block that went wrong (tripped, wrong activity, stopped early)? "
                   "Enter = none, or type numbers separated by spaces: ").strip()
    for tok in answer.split():
        for seg in segments:
            if tok.isdigit() and seg["i"] == int(tok):
                seg["ok"] = False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=["A", "B", "test"], required=True)
    ap.add_argument("--stairs", action="store_true", help="include the stairs bouts (protocol A)")
    ap.add_argument("--no-voice", action="store_true", help="printed cues only")
    ap.add_argument("--person", default="julian")
    ap.add_argument("--mount-notes", default="UNO Q + MPU-6050 on the outside of the right ankle")
    ap.add_argument("--notes", default="", help="anything unusual about this session")
    ap.add_argument("--hallway-m", type=lambda v: None if v.lower() == "none" else float(v), default=HALLWAY_M,
                    help="length of the hallway used, in metres, or none if not measured")
    args = ap.parse_args()

    voice = Voice(not args.no_voice)
    accel_cal = cal.load_accel_calibration()
    if accel_cal is None:
        print("! No accelerometer calibration found - run scripts/calibrate_accel.py first for the best "
              "orientation check. Continuing without it.")
    steps = steps_for(args.protocol, enabled={"stairs"} if args.stairs else set())
    session = next_session_id("T" if args.protocol == "test" else "S")

    stream = LiveStream().start()
    print(f"Session {session}, protocol {args.protocol}: {len(steps)} blocks. Waiting for IMU data...")
    if not stream.wait_for_data():
        print("No IMU data. Is the board app running? (scripts/listen.py prints a fix if it's stuck)")
        stream.stop()
        return
    first_seq = stream.latest_seq()
    restarts = stream.tracker.restarts
    segments = []
    bias = None

    try:
        bias, gravity, err, change = calibrate(stream, voice, accel_cal)
        voice.say("Follow my instructions from now on.")
        for i, step in enumerate(steps, start=1):
            if step.kind == "calibrate":
                continue  # done above, at the laptop
            go = None
            if step.kind == "bout":
                start, go, end, timed_out = run_bout(stream, step, bias, voice)
            else:
                (start, end), timed_out = run_timed(stream, step, voice), False
            segments.append({"i": i, "kind": step.kind, "label": step.label, "start_seq": start, "go_seq": go,
                             "end_seq": end, "target_steps": step.target_steps,
                             "steps": 0 if step.kind == "hold" else None, "ok": True, "timed_out": timed_out})
            if stream.tracker.restarts != restarts:
                raise RuntimeError("the board app restarted mid-session (seq numbers reset)")
        time.sleep(LAG_SAMPLES / 100 + 0.5)  # let the last samples arrive
        voice.say("Session complete. Please come back to the laptop.")
    except KeyboardInterrupt:
        print("\nStopped early.")
        if bias is None or input("Save what was recorded? [y/N] ").strip().lower() != "y":
            stream.stop()
            return
    except (RuntimeError, TimeoutError) as exc:
        voice.say("Recording stopped.", wait=False)
        print(f"\n! {exc}. Nothing saved - please record the session again.")
        stream.stop()
        return

    samples = stream.samples()
    stream.stop()
    samples = samples[samples[:, 0] >= first_seq]
    if segments:
        confirm_counts(segments)
    missing = missing_seqs(samples) if len(samples) else []
    meta = {
        "session": session,
        "protocol": args.protocol,
        "person": args.person,
        "ankle": "right",
        "mount_notes": args.mount_notes,
        "notes": args.notes,
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "board_run": stream.run,
        "axis_map": AXIS_MAP,
        "accel_calibration": accel_cal,
        "gyro_bias_dps": np.round(bias, 3).tolist(),
        "gravity_body_g": np.round(gravity, 4).tolist(),
        "orientation_error_deg": round(err, 2),
        "orientation_change_deg": None if change is None else round(change, 2),
        "hallway_m": args.hallway_m,
        "lag_samples": LAG_SAMPLES,
        "samples": int(len(samples)),
        "missing": int(len(missing)),
        "segments": segments,
    }
    session_dir = RAW_DIR / session
    write_session(session_dir, samples, meta)
    total = sum(s["steps"] or 0 for s in segments if s["ok"])
    print(f"\nSaved {session_dir}  ({len(samples)} samples, {len(segments)} blocks, {total} steps)")
    if len(missing):
        app = "/home/arduino/ArduinoApps/stepcounter/python/data"
        print(f"! {len(missing)} samples were lost over MQTT. Fill them from the board's backup log:\n"
              f"    adb pull {app}/stream_{stream.run}.csv /tmp/\n"
              f"    ../../.venv/bin/python scripts/fill_gaps.py {session} /tmp/stream_{stream.run}.csv")
    print(f"Plot it:  ../../.venv/bin/python scripts/plot_session.py {session}")


if __name__ == "__main__":
    main()
