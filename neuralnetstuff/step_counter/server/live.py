"""WebSocket feed for the live view (milestone M7, PLAN.md D17).

  ws://127.0.0.1:8000/ws/live?source=replay&session=S005&speed=1    replay a recorded session
  ws://127.0.0.1:8000/ws/live?source=board                          the board, live over MQTT

Every 0.25 s the browser gets the new body-frame samples, the latest stretch of the stitched
probability curve (ensemble mean and spread), the strides that became final, the running
count with its interval, and (after each inference step) the first CNN's layer activations.
The browser can send {"level": 0.8} to change the interval level.

Replaying the locked test sessions is refused until the final test has been run (D9).
"""
import asyncio
import json
import queue

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from stepnet.data.blocks import TEST_SESSIONS
from stepnet.infer import LiveEngine
from stepnet.io.session import RAW_DIR, read_session
from stepnet.tracking import ROOT

TICK_S = 0.25
FINAL_REPORT = ROOT / "reports" / "m10_final_test.md"


async def live_socket(ws: WebSocket, ensemble):
    await ws.accept()
    params = ws.query_params
    if ensemble is None:
        await ws.send_json({"type": "error", "message": "No trained ensemble yet (models/cnn_ensemble_v1). Run scripts/uncertainty.py --train-final."})
        await ws.close()
        return
    members, cal = ensemble
    source = params.get("source", "replay")
    level = float(params.get("level", 0.9))

    if source == "replay":
        session = params.get("session", "")
        if session in TEST_SESSIONS and not FINAL_REPORT.exists():
            await ws.send_json({"type": "error", "message": f"{session} is a locked test session: replay it after the final test (M10)."})
            await ws.close()
            return
        try:
            samples, meta = read_session(RAW_DIR / session)
        except (OSError, ValueError):
            await ws.send_json({"type": "error", "message": f"no session {session}"})
            await ws.close()
            return
        engine = LiveEngine(members, cal, gyro_bias_dps=meta["gyro_bias_dps"], accel_cal=meta.get("accel_calibration"), level=level)
        speed = max(0.25, min(8.0, float(params.get("speed", 1))))
        per_tick = int(round(100 * TICK_S * speed))
        chunks = (samples[k:k + per_tick] for k in range(0, len(samples), per_tick))
        segments = [s for s in meta["segments"] if s["kind"] != "calibrate"]
        source_info = {"source": "replay", "session": session, "protocol": meta["protocol"], "segments": segments}
        next_chunk = lambda: next(chunks, None)
    else:
        from stepnet.io.mqtt_link import BoardLink
        link = BoardLink().start()
        engine = LiveEngine(members, cal, level=level)
        source_info = {"source": "board", "segments": []}

        def next_chunk():
            rows = []
            while True:
                try:
                    event = link.events.get_nowait()
                except queue.Empty:
                    break
                if event[0] == "imu":
                    rows.append(event[3])
            return np.concatenate(rows) if rows else np.empty((0, 8), dtype=np.int64)

    await ws.send_json({"type": "hello", **source_info, "window": engine.window, "latency": engine.latency,
                        "level": level, "levels": sorted(float(k) for k in cal["q"]),
                        "layers": [{"channels": c, "dilation": d} for c, d in
                                   zip(cal["config"]["model"]["channels"], cal["config"]["model"]["dilations"])]})
    sent_strides, sent_len = 0, 0
    try:
        while True:
            chunk = next_chunk()
            if chunk is None:
                await ws.send_json({"type": "end"})
                break
            stepped = False
            if len(chunk):
                if engine.feed(chunk):
                    engine.step()
                    stepped = True
            msg = {"type": "update", "calibrating": engine.calibrating}
            n = len(engine.seq)
            if n > sent_len:
                x = np.asarray(engine.x[sent_len:n])
                msg["imu"] = {"seq": engine.seq[sent_len:n], "x": np.round(x, 2).T.tolist()}
                sent_len = n
            if n >= engine.window:
                curves = engine.curves()[:, -engine.window:]
                msg["prob"] = {"start_seq": engine.seq[n - engine.window],
                               "mean": np.round(curves.mean(axis=0), 3).tolist(),
                               "spread": np.round(curves.std(axis=0), 3).tolist()}
                strides, heights = engine.strides()
                msg["strides"] = {"seq": strides[sent_strides:].tolist(), "p": np.round(heights[sent_strides:], 3).tolist()}
                sent_strides = len(strides)
                c = engine.count()
                msg["count"] = {k: c[k] for k in ("n_hat", "low", "high", "level", "sigma", "var_pb", "var_ens", "var_dbl", "bouts")}
                msg["final_seq"] = engine.seq[engine.final_len() - 1] if engine.final_len() else None
            if stepped:
                msg["activations"] = engine.latest_activations()
            await ws.send_json(msg)
            try:
                incoming = await asyncio.wait_for(ws.receive_text(), timeout=TICK_S)
                data = json.loads(incoming)
                if "level" in data and str(float(data["level"])) in cal["q"]:
                    engine.level = float(data["level"])
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        if source != "replay":
            link.stop()
