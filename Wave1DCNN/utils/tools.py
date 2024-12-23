import numpy as np
import torch
from scipy import signal
from nnAudio.Spectrogram import STFT
import noisereduce as nr
import librosa
import scipy
import matplotlib.pyplot as plt



class STFTtransform:
    def __init__(self, n_fft, sr, fmin, fmax, hop_length):
        self.stft_transform = STFT(n_fft = n_fft, fmin=fmin, fmax=fmax, hop_length=hop_length)
        
    def apply(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.ndim == 1:
            wave = wave.unsqueeze(0)  # Agrega una dimensión de batch
        spectrogram = self.stft_transform(wave)
        # if spectrogram size is like torch.Size([1025, 7119, 2])
        magnitude = torch.sqrt(spectrogram[..., 0]**2 + spectrogram[..., 1]**2)
        return magnitude.squeeze(0)

    def show_stft(self, spectrogram: torch.Tensor):
        spectrogram_np = spectrogram.detach().cpu().numpy()
        spectrogram_np = np.resize(spectrogram_np, (7119, 1025))
        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram_np, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Amplitud')
        plt.xlabel('Frames de Tiempo')
        plt.ylabel('Frecuencias')
        plt.title('Espectrograma STFT')
        plt.show()
        
        
def get_cwt_morlet(signal):
    return np.abs(scipy.signal.cwt(signal, scipy.signal.morlet, np.arange(1, 10)))


def segment_wav(wav, window_size = 256, stride = 128):
    if len(wav.shape) == 2:
        wav = wav.mean(axis=1)
    segments = [wav[i:i+window_size] for i in range(0, len(wav) - window_size, stride)]
    return np.array(segments)


def remove_noise(samples, rate):
    # Noise reduction with noisereduce
    return nr.reduce_noise(y=samples, sr=rate)


def apply_bandpass(x, lf=1, hf=100, order=16, sr=30000):
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt((hf - lf) / (sr / 2))
    x = signal.sosfiltfilt(sos, x) / normalization
    return x


   
def normalize_waveform(wave, eps=1e-9):
    """
    Normalize an audio waveform to have zero mean and unit variance.

    Args:
        waveform (numpy): The input audio waveform (shape: [num_channels, num_samples]).
        eps (float): A small value to avoid division by zero.

    Returns:
        numpy: The normalized audio waveform.
    """
    mean = np.mean(wave, axis=-1, keepdims=True)
    std = np.std(wave, axis=-1, keepdims=True)
    normalized_waveform = (wave - mean) / (std + eps)
    return normalized_waveform


# Plot the waveform
def plot_waveform_numpy(wave, sample_rate, title="Waveform"):
    if wave.ndim == 1:
        # Mono audio
        time = np.linspace(0, len(wave) / sample_rate, num=len(wave))
        plt.figure(figsize=(10, 4))
        plt.plot(time, wave, label="Mono")
    else:
        # Stereo or multichannel audio
        time = np.linspace(0, wave.shape[0] / sample_rate, num=wave.shape[0])
        plt.figure(figsize=(10, 4))
        for channel in range(wave.shape[1]):
            plt.plot(time, wave[:, channel], label=f"Channel {channel+1}")

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.show()
