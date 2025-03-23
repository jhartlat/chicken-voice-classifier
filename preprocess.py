# preprocess.py

import librosa
import numpy as np

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 256
MEL_BINS = 128
WAV_SIZE = 33075
WINDOW = 'hann'

def audio_to_log_mel_spec(filename):
    wav, sr = librosa.load(filename, sr=SAMPLE_RATE)
    wav = librosa.util.normalize(wav)

    if len(wav) > WAV_SIZE:
        wav = wav[:WAV_SIZE]
    else:
        wav = librosa.util.pad_center(wav, size=WAV_SIZE)

    mel_spec = librosa.feature.melspectrogram(
        y=wav,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=WINDOW,
        n_mels=MEL_BINS,
        power=2.0
    )
    return librosa.power_to_db(mel_spec, ref=np.max)
