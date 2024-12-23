import sys
import numpy as np
from scipy import signal
from nnAudio.Spectrogram import CQT1992v2
import noisereduce as nr
import torch
import librosa
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data
    """
    y, sr = librosa.load(wav_name, sr=None, mono=mono)
    return y, sr

def wave2image(waveimg):
    """
    Plot wave data as a grid of subplots in memory and convert it to a tensor.
    This implementation uses the multi-channel plotting logic similar to show_cqt.
    """
    image = waveimg.squeeze()
    plt.figure(figsize=(10, 4))  # Set the figure size
    plt.imshow(image, aspect='auto')
    plt.axis('off')  # Turn off axis
    # Save the figure to a BytesIO buffer
    buf = BytesIO()
    plt.savefig(buf, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    buf.seek(0)

    image = Image.open(buf).convert("RGB")
    return image

class CQTtransform:
    def __init__(self, sr=8000, fmin=2000, fmax=4000, hop_length=248):
        self.cqt_transform = CQT1992v2(sr=sr, fmin=fmin, fmax=fmax, hop_length=hop_length)
        
    def transform(self, waves: torch.Tensor) -> torch.Tensor:
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

def wav2mel_vector(y, scaler=None, 
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

def remove_noise(samples, rate):
    # Noise reduction with noisereduce
    return nr.reduce_noise(y=samples, sr=rate)


def apply_bandpass(x, lf=1, hf=100, order=16, sr=30000):
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt((hf - lf) / (sr / 2))
    x = signal.sosfiltfilt(sos, x) / normalization
    return x

