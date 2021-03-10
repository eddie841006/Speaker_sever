# Third Party
import librosa
import numpy as np

# ===============================================
#       code from Arsha for loading data.
# ===============================================
def load_wav(wav, sr):
    sr_ret = 16000
    if not sr==sr_ret:
        wav = librosa.core.resample(wav, sr, sr_ret)
    extended_wav = np.append(wav, wav[::-1])
    return extended_wav


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250):
    wav = load_wav(path, sr=sr)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    spec_mag = mag.T
    freq, time = spec_mag.shape
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)


