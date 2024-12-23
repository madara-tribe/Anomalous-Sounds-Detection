import sys
import numpy as np
from scipy import signal
import librosa
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def load_audio_file(file_path, mono=False):
    """
    Load a WAV audio file.

    Args:
        file_path (str): Path to the WAV file.
        mono (bool): If True, convert multi-channel audio to mono.

    Returns:
        tuple: Audio signal (numpy array) and sampling rate (int).
    """
    audio_signal, sampling_rate = librosa.load(file_path, sr=None, mono=mono)
    return audio_signal, sampling_rate


def convert_waveform_to_image(waveform):
    """
    Plot wave data as a grid of subplots in memory and convert it to a tensor.
    This implementation uses the multi-channel plotting logic similar to show_cqt.
    """
    waveform = waveform.squeeze()
    plt.figure(figsize=(10, 4))  # Set the figure size
    plt.imshow(waveform, aspect='auto')
    plt.axis('off')  # Turn off axis
    # Save the figure to a BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    buffer.seek(0)

    image = Image.open(buffer).convert("RGB")
    return image

def wave_to_mel_features(y, sr=16000,
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

def apply_bandpass(x, lf=1, hf=100, order=16, sr=30000):
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt((hf - lf) / (sr / 2))
    x = signal.sosfiltfilt(sos, x) / normalization
    return x

