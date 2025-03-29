import sounddevice as sd
from scipy.io.wavfile import write
import argparse
from predict import predict

def record_audio(duration=1.5, fs=22050, channels=1, filename="live_input.wav"):
    print(f"Recording audio for {duration} seconds at {fs} Hz...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()  # Wait until recording is finished
    write(filename, fs, audio)
    print(f"Audio recorded and saved to {filename}")
    return filename

def main():
    parser = argparse.ArgumentParser(
        description="Record audio from the microphone and classify chicken sound."
    )
    parser.add_argument("--duration", type=float, default=1.5,
                        help="Recording duration in seconds (default: 1.5)")
    parser.add_argument("--fs", type=int, default=22050,
                        help="Sampling rate (default: 22050)")
    parser.add_argument("--channels", type=int, default=1,
                        help="Number of audio channels (default: 1)")
    parser.add_argument("--model", default="models/trained_custom_model.keras",
                        help="Path to the trained model")
    args = parser.parse_args()

    wav_file = record_audio(duration=args.duration, fs=args.fs, channels=args.channels)
    predict(wav_file, args.model)

if __name__ == "__main__":
    main()
