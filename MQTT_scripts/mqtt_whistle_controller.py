import queue

import numpy as np
import pyaudio
import paho.mqtt.client as mqtt

BROKER = "test.mosquitto.org"
LISTEN_TOPIC = "ME193/Rogers"
DRIVE_TOPIC = "ME193/Julian"

SAMPLE_RATE = 44100
CHUNK_SIZE = 2048

FREQ_MIN = 500.0
FREQ_MAX = 3000.0

# How many RMS units above the measured ambient noise floor counts as a
# "full speed" whistle. Edit this to tune how hard you need to whistle.
AMPLITUDE_RANGE = 300.0

# The quiet gate sits at max(floor * this multiplier, floor + this margin).
# Edit these if quiet ambient noise is still triggering movement.
SILENCE_MARGIN_MULT = 1.5
SILENCE_MARGIN_ADD = 40.0

# A whistle is a narrow spike in the frequency spectrum; background noise
# (fans, talking, room hum) is spread out. This is how many times louder
# the peak frequency must be than the average of the band to count as a
# whistle rather than noise. Edit this if noise still causes movement, or
# if real whistles are being ignored.
TONALITY_THRESHOLD = 4.0

CALIBRATION_CHUNKS = 20

incoming_messages = queue.Queue()


def on_connect(client, userdata, flags, reason_code, properties):
    client.subscribe(LISTEN_TOPIC)


def on_message(client, userdata, message):
    incoming_messages.put(message.payload.decode().strip())


def clamp_map(value, in_min, in_max, out_min=-1.0, out_max=1.0):
    if value <= in_min:
        return out_min
    if value >= in_max:
        return out_max
    fraction = (value - in_min) / (in_max - in_min)
    return out_min + fraction * (out_max - out_min)


def choose_role(client):
    while True:
        choice = input("Choose role - (g)oalie or (s)hooter: ").strip().lower()
        if choice in ("g", "goalie"):
            role = "goalie"
            break
        if choice in ("s", "shooter"):
            role = "shooter"
            break
        print("Please enter 'g' for goalie or 's' for shooter.")
    client.publish(DRIVE_TOPIC, role)
    print(f"Sent role '{role}' on '{DRIVE_TOPIC}'.")
    return role


def wait_for_start():
    print(f"Waiting for 'start' on '{LISTEN_TOPIC}'...")
    while True:
        payload = incoming_messages.get()
        if payload.lower() == "start":
            print("Received 'start'. Entering controller state.")
            return


def analyze_chunk(samples, sample_rate):
    """Return (rms, dominant_frequency, tonality) for one audio chunk.

    tonality is how many times louder the peak frequency in the whistle
    band is compared to the average of that band - high for a pure
    whistle tone, low for broadband noise.
    """
    rms = float(np.sqrt(np.mean(samples ** 2)))

    windowed = samples * np.hanning(len(samples))
    magnitudes = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(samples), d=1.0 / sample_rate)

    band = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    band_magnitudes = magnitudes[band]
    band_freqs = freqs[band]

    peak_index = np.argmax(band_magnitudes)
    peak_magnitude = float(band_magnitudes[peak_index])
    frequency = float(band_freqs[peak_index])
    mean_magnitude = float(np.mean(band_magnitudes))
    tonality = peak_magnitude / (mean_magnitude + 1e-9)

    return rms, frequency, tonality


def calibrate_noise_floor(stream):
    print("Calibrating ambient noise level - stay quiet for a second...")
    floor_samples = []
    for _ in range(CALIBRATION_CHUNKS):
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
        floor_samples.append(float(np.sqrt(np.mean(samples ** 2))))
    floor = float(np.mean(floor_samples))
    print(f"Ambient noise floor measured at rms={floor:.1f}")
    return floor


def run_controller(client):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                         input=True, frames_per_buffer=CHUNK_SIZE)

    noise_floor = calibrate_noise_floor(stream)
    amp_min = max(noise_floor * SILENCE_MARGIN_MULT, noise_floor + SILENCE_MARGIN_ADD)
    amp_max = amp_min + AMPLITUDE_RANGE
    print(f"Whistle gate: quiet below rms={amp_min:.0f}, full speed at rms={amp_max:.0f}")
    print("Listening for whistles. Ctrl+C to quit.")

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
            rms, frequency, tonality = analyze_chunk(samples, SAMPLE_RATE)

            is_whistling = rms > amp_min and tonality > TONALITY_THRESHOLD
            if not is_whistling:
                speed, turn = 0.0, 0.0
            else:
                speed = clamp_map(rms, amp_min, amp_max)
                turn = clamp_map(frequency, FREQ_MIN, FREQ_MAX)

            payload = f"({speed:.2f}, {turn:.2f})"
            client.publish(DRIVE_TOPIC, payload)
            print(f"rms={rms:7.1f}  tonality={tonality:5.1f}  whistling={is_whistling!s:5}  "
                  f"speed={speed:+.2f}  turn={turn:+.2f}  -> {payload}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER)
    client.loop_start()

    try:
        choose_role(client)
        wait_for_start()
        run_controller(client)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
