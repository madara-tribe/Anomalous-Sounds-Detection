import argparse
import librosa
import audiomentations as A
import soundfile as sf
import sys


# Helper Functions
def load_audio(file_path):
    """Load an audio file and return the waveform and sample rate."""
    waveform, sample_rate = librosa.load(file_path, sr=None)
    return waveform, sample_rate


# Audio Augmentation Functions
def apply_gaussian_noise(input_path, output_path, min_amplitude=0.001, max_amplitude=0.045, p=1.0):
    """Apply Gaussian noise to the audio and save the processed version."""
    waveform, sample_rate = load_audio(input_path)
    transform = A.Compose([
        A.AddGaussianNoise(min_amplitude=min_amplitude, max_amplitude=max_amplitude, p=p)
    ])
    processed_audio = transform(waveform, sample_rate=sample_rate)
    sf.write(output_path, processed_audio, sample_rate)


def apply_color_noise(input_path, output_path):
    """Apply colored noise (white noise variant) to the audio and save the processed version."""
    waveform, sample_rate = load_audio(input_path)
    transform = A.Compose([
        A.AddColorNoise(p=1.0, min_snr_db=5, max_snr_db=20, min_f_decay=0, max_f_decay=0)  # White noise
    ])
    processed_audio = transform(waveform, sample_rate=sample_rate)
    sf.write(output_path, processed_audio, sample_rate)


def apply_short_noise(input_path, output_path):
    """Add short random noises to the audio and save the processed version."""
    waveform, sample_rate = load_audio(input_path)
    transform = A.Compose([
        A.AddShortNoises("./archive", p=1.0)
    ])
    processed_audio = transform(waveform, sample_rate=sample_rate)
    sf.write(output_path, processed_audio, sample_rate)


# Main Execution Logic
def main():
    parser = argparse.ArgumentParser(description="Audio augmentation tool")
    parser.add_argument('-i', '--input_path', type=str, default="input.wav", help='Input audio file path')
    parser.add_argument('-o', '--output_path', type=str, default="output.wav", help='Output audio file path')
    parser.add_argument('-m', '--mode', type=str, choices=['gaussian', 'color', 'short'], required=True,
                        help='Augmentation mode: gaussian (Gaussian noise), color (Colored noise), short (Short noises)')
    args = parser.parse_args()

    if args.mode == "gaussian":
        apply_gaussian_noise(args.input_path, args.output_path)
    elif args.mode == "color":
        apply_color_noise(args.input_path, args.output_path)
    elif args.mode == "short":
        apply_short_noise(args.input_path, args.output_path)
    else:
        print("Invalid mode. Choose from 'gaussian', 'color', or 'short'.")
        sys.exit(1)


if __name__ == '__main__':
    main()
