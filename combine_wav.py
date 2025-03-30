import os
import argparse
from pydub import AudioSegment

# 🔧 Hardcode the ffmpeg executable path
AudioSegment.converter = r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

def combine_wav_files(folder_path, output_filename="combined_output.wav"):
    # List all .wav files in folder
    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    wav_files.sort()  # Optional: sort alphabetically

    if not wav_files:
        print("No .wav files found in the specified folder.")
        return

    # Initialize empty audio segment
    combined = AudioSegment.empty()

    print(f"\nFound {len(wav_files)} .wav file(s). Combining...\n")

    for file in wav_files:
        file_path = os.path.join(folder_path, file)
        print(f"➕ Adding: {file}")
        sound = AudioSegment.from_wav(file_path)
        combined += sound

    # Export combined audio
    output_path = os.path.join(folder_path, output_filename)
    combined.export(output_path, format="wav")
    print(f"\n✅ Combined file saved as: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Combine all .wav files in a folder into one .wav file.")
    parser.add_argument("folder", help="Path to the folder containing .wav files")
    parser.add_argument("--output", default="combined_output.wav", help="Name of the output .wav file")

    args = parser.parse_args()
    combine_wav_files(args.folder, args.output)

if __name__ == "__main__":
    main()
