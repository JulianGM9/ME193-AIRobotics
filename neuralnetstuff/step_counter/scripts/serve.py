"""Start the laptop UI (PLAN.md D17).

  ../../.venv/bin/python scripts/serve.py            # http://127.0.0.1:8000, opens your browser
  ../../.venv/bin/python scripts/serve.py --no-open
"""
import argparse
import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.app import create_app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    print(f"Step Counter UI: {url}   (Ctrl+C to stop)")
    if not args.no_open:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
