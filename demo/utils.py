import numpy as np
import matplotlib.pyplot as plt
import librosa
import noisereduce as nr
import scipy
import sys
import torch
from nnAudio.Spectrogram import CQT1992v2, STFT

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

class CQTtransform:
    def __init__(self, sr=8000, fmin=2000, fmax=4000, hop_length=248):
        self.cqt_transform = CQT1992v2(sr=sr, fmin=fmin, fmax=fmax, hop_length=hop_length)
        
    def cqt_transform(self, wave: torch.Tensor) -> torch.Tensor:
        #waves = torch.from_numpy(wave).float()
        if len(waves.shape) > 1:
            waves = waves.mean(dim=1)
        image = self.cqt_transform(waves)
        return image.transpose(0, 1).transpose(1, 2)
        
    def show_cqt(self, image, figsize=(15,7)):
        plt.figure(figsize=figsize)
        for j in range(image.shape[2]):
            plt.subplot(3, 3, j + 1)
            plt.imshow(image[:,:,j], aspect="auto")
            plt.colorbar()
        plt.tight_layout()
        plt.show()

class STFTtransform:
    def __init__(self, n_fft, sr, fmin, fmax, hop_length):
        self.stft_transform = STFT(n_fft = n_fft, sr=sr, fmin=fmin, fmax=fmax, hop_length=hop_length)
        
    def apply(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.ndim == 1:
            wave = wave.unsqueeze(0)  # Agrega una dimensión de batch
        spectrogram = self.stft_transform(wave)
        print(spectrogram.shape)
        # if spectrogram size is like torch.Size([1025, 7119, 2])
        magnitude = torch.sqrt(spectrogram[..., 0]**2 + spectrogram[..., 1]**2)
        return magnitude.squeeze(0)

    def show_stft(self, spectrogram: torch.Tensor):
        spectrogram_np = spectrogram.detach().cpu().numpy()
        spectrogram_np = np.resize(spectrogram_np, (6897, 250))
        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram_np, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Amplitud')
        plt.xlabel('Frames de Tiempo')
        plt.ylabel('Frecuencias')
        plt.title('Espectrograma STFT')
        plt.show()

        
def remove_noise(samples, rate):
    # Noise reduction with noisereduce
    return nr.reduce_noise(y=samples, sr=rate)


def wav2mel_vector(y,
                         sr=16000,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=1.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    vector_array = log_mel_spectrogram.T

    # normalize from 0 to 1
    vector_array -= vector_array.min()
    vector_array /= vector_array.max()
    return vector_array
    
    

