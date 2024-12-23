import numpy as np
import torch
from scipy import signal
from nnAudio.Spectrogram import CQT1992v2
import noisereduce as nr


class ConstantQTransform:
    """
    Implements the Constant-Q Transform for audio signal processing.
    """
    def __init__(self, sr=8000, fmin=2000, fmax=4000, hop_length=248):
        self.cqt_transform = CQT1992v2(sr=sr, fmin=fmin, fmax=fmax, hop_length=hop_length)
        
    def transform(self, waves: torch.Tensor) -> torch.Tensor:
        """Transforms an audio waveform to its CQT representation."""
        #waves = torch.from_numpy(wave).float()
        if len(waves.shape) > 1:
            waves = waves.mean(dim=1)
        cqt_image = self.cqt_transform(waves)
        return cqt_image.transpose(0, 1).transpose(1, 2)
        
    def visualize_cqt(self, cqt_image, figure_size=(15,7)):
        import matplotlib.pyplot as plt
        plt.figure(figsize=figure_size)
        for j in range(cqt_image.shape[2]):
            plt.subplot(3, 3, j + 1)
            plt.imshow(cqt_image[:,:,j], aspect="auto")
            plt.colorbar()
        plt.tight_layout()
        plt.show()




def denoise_audio(waveform, sampling_rate):
    """Reduces noise from an audio waveform."""
    return nr.reduce_noise(y=waveform, sr=sampling_rate)


def apply_bandpass(x, lf=1, hf=100, order=16, sr=30000):
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt((hf - lf) / (sr / 2))
    x = signal.sosfiltfilt(sos, x) / normalization
    return x


