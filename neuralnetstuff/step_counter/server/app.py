"""Laptop backend (PLAN.md D17): serves the web UI and a small JSON API.

  GET  /api/sessions                  every recorded session + its label status
  GET  /api/sessions/{id}             signals (body frame), segments, candidates, latest labels
  POST /api/sessions/{id}/labels      {"strides": [seq...], "segment_ok": {"3": false}}
                                      -> saves the next label version, or 422 if a count check fails

Run it with scripts/serve.py.
"""
import re
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from stepnet import labels as lab
from stepnet.io.session import RAW_DIR, missing_seqs, read_session

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


class LabelsIn(BaseModel):
    strides: list[int]
    segment_ok: dict[str, bool] = {}


def create_app(raw_dir=RAW_DIR, labels_dir=lab.LABELS_DIR, web_dir=WEB_DIR) -> FastAPI:
    raw_dir, labels_dir = Path(raw_dir), Path(labels_dir)
    app = FastAPI(title="Step Counter")

    def load(session):
        if not re.fullmatch(r"[ST]\d{3}", session) or not (raw_dir / session / "meta.json").exists():
            raise HTTPException(404, f"no session {session}")
        return read_session(raw_dir / session)

    @app.get("/api/sessions")
    def sessions():
        out = []
        for session_dir in sorted(raw_dir.glob("[ST][0-9][0-9][0-9]")):
            try:
                _, meta = read_session(session_dir)
            except (OSError, ValueError):
                continue
            labels = lab.load_latest_labels(session_dir.name, labels_dir)
            out.append({
                "session": meta["session"],
                "protocol": meta["protocol"],
                "recorded_at": meta["recorded_at"],
                "samples": meta["samples"],
                "label_version": labels["version"] if labels else None,
                "strides": len(labels["strides"]) if labels else None,
            })
        return out

    @app.get("/api/sessions/{session}")
    def session(session: str):
        samples, meta = load(session)
        acc, gyr = lab.body_signals(samples, meta)
        return {
            "meta": meta,
            "seq": samples[:, 0].tolist(),
            "acc": np.round(acc.T, 3).tolist(),  # [S, F, L] x n
            "gyr": np.round(gyr.T, 1).tolist(),
            "missing": int(len(missing_seqs(samples))),
            "candidates": lab.find_candidates(samples, meta),
            "candidate_params": lab.CANDIDATE_PARAMS,
            "omega_params": lab.OMEGA_PARAMS,
            "labels": lab.load_latest_labels(session, labels_dir),
        }

    @app.post("/api/sessions/{session}/labels")
    def save(session: str, body: LabelsIn):
        samples, meta = load(session)
        try:
            return lab.save_labels(session, meta, body.strides, body.segment_ok, labels_dir=labels_dir,
                                   candidates=lab.find_candidates(samples, meta))
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app
