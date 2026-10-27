"""Gait classifier - Linux side. Pick what the app does with MODE:

  "live"    - classify activity in real time with both KNN models (live.py).
              Choose which model drives the LED matrix with MODEL in live.py.
  "collect" - record one labeled training session (collect.py).
"""
from arduino.app_utils import *

MODE = "live"  # "live" or "collect"

if MODE == "collect":
    import collect as app
else:
    import live as app

App.run(user_loop=app.loop)
