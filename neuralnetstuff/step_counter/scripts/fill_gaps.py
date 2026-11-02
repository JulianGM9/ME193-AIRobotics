"""Fill samples lost over MQTT from the board's backup log (PLAN.md D1).

  adb pull /home/arduino/ArduinoApps/stepcounter/python/data/stream_<run>.csv /tmp/
  ../../.venv/bin/python scripts/fill_gaps.py S003 /tmp/stream_<run>.csv

record.py prints both commands, with the right run id, whenever a session has gaps.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stepnet.io.session import RAW_DIR, fill_gaps, read_session


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    session_dir, board_csv = RAW_DIR / sys.argv[1], Path(sys.argv[2])
    meta = read_session(session_dir)[1]
    if meta.get("board_run") and meta["board_run"] not in board_csv.name:
        print(f"! {board_csv.name} is not from this session's board run ({meta['board_run']}) - not filling.")
        sys.exit(1)
    filled = fill_gaps(session_dir, board_csv)
    print(f"Filled {filled} samples; {read_session(session_dir)[1]['missing']} still missing.")


if __name__ == "__main__":
    main()
