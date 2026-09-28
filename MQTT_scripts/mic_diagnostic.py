import numpy as np
import pyaudio

SAMPLE_RATE = 44100
CHUNK_SIZE = 2048


def list_input_devices(audio):
    default_index = audio.get_default_input_device_info()["index"]
    print("Available input devices:")
    for index in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(index)
        if info.get("maxInputChannels", 0) > 0:
            marker = " (default)" if index == default_index else ""
            print(f"  [{index}] {info['name']}{marker}")


def main():
    audio = pyaudio.PyAudio()
    list_input_devices(audio)

    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                         input=True, frames_per_buffer=CHUNK_SIZE)
    print("\nListening... talk, clap, or whistle. Ctrl+C to stop.\n")

    try:
        while True:
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float64)
            rms = float(np.sqrt(np.mean(samples ** 2)))
            peak = float(np.max(np.abs(samples)))
            print(f"rms={rms:8.2f}  peak={peak:8.0f}")
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


if __name__ == "__main__":
    main()
