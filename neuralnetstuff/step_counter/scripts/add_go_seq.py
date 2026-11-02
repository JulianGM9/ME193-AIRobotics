"""One-time fix: estimate "Go" for bouts recorded before record.py saved it (PLAN.md change log).

  ../../.venv/bin/python scripts/add_go_seq.py            # every session that needs it
  ../../.venv/bin/python scripts/add_go_seq.py S003 S005

A bout's start_seq is when its spoken instruction began. Go came right after the
instruction, so  go_seq ~= start_seq + (instruction length + ~0.2 s for `say` to start) x 100.
The instruction length is measured by rendering the same text with the same `say` voice and
rate (macOS). The result is marked "go_seq_estimated": true. Only meta.json changes; imu.csv
never does.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stepnet.io.protocol import SAMPLE_RATE_HZ
from stepnet.io.session import RAW_DIR, read_session, write_meta
from stepnet.protocols import PROTOCOLS

SAY_STARTUP_S = 0.2
# Before this, a bout's start_seq already was Go (bouts started at "Go", not at the instruction).
BOUT_START_AT_INSTRUCTION_SINCE = "2026-11-02 14:36:00"


def instruction_seconds(text, cache={}):
    if text not in cache:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "say.aiff"
            subprocess.run(["say", "-r", "185", "-o", str(out), text], check=True)
            info = subprocess.run(["afinfo", str(out)], capture_output=True, text=True, check=True).stdout
        line = next(l for l in info.splitlines() if "estimated duration" in l)
        cache[text] = float(line.split(":")[1].split()[0])
    return cache[text]


def main():
    ids = sys.argv[1:] or sorted(p.name for p in RAW_DIR.glob("[ST][0-9][0-9][0-9]"))
    for sid in ids:
        _, meta = read_session(RAW_DIR / sid)
        if meta["recorded_at"] < BOUT_START_AT_INSTRUCTION_SINCE:
            print(f"{sid}: recorded before bouts started at the instruction - start_seq is Go, nothing to do")
            continue
        say_by_label = {st.label: st.say for st in PROTOCOLS[meta["protocol"]] if st.kind == "bout"}
        changed = 0
        for seg in meta["segments"]:
            if seg["kind"] != "bout" or seg.get("go_seq"):
                continue
            secs = instruction_seconds(say_by_label[seg["label"]]) + SAY_STARTUP_S
            seg["go_seq"] = seg["start_seq"] + int(round(secs * SAMPLE_RATE_HZ))
            seg["go_seq_estimated"] = True
            changed += 1
        if changed:
            write_meta(RAW_DIR / sid, meta)
        print(f"{sid}: estimated Go for {changed} bouts")


if __name__ == "__main__":
    main()
