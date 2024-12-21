import argparse
import librosa
import librosa.display
import sys
import audiomentations as A
import soundfile as sf


def load_wave(fle_path):
    wav, sr = librosa.load(fle_path, sr=None)
    return wav, sr

def add_GaussianNoise(input_path, output_path, min_amplitude=0.001, max_amplitude=0.045, p=1.0):
    wav, sr = load_wave(input_path)
    transform = A.Compose([
        A.AddGaussianNoise(min_amplitude=min_amplitude, max_amplitude=max_amplitude, p=p),
    ])
    augmented_wave = transform(wav, sample_rate=sr)
    sf.write(output_path,augmented_wave,sr)

def add_WhiteNoise(input_path, output_path):
    wav, sr = load_wave(input_path)
    transform = A.Compose([
        A.AddColorNoise(p=1.0, min_snr_db=5, max_snr_db=20, min_f_decay=0, max_f_decay=0),#white
    ])
    augmented_wave = transform(wav, sample_rate=sr)
    sf.write(output_path,augmented_wave,sr)

def add_ShortNoise(input_path, output_path):
    wav, sr = load_wave(input_path)
    transform = A.Compose([
        A.AddShortNoises("./archive", p=1.0)
    ])
    augmented_wave = transform(wav, sample_rate=sr)
    sf.write(output_path, augmented_wave,sr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default="input.wav", help='output wave file name')
    parser.add_argument('-o', '--output_path', type=str, default="MCU03_正常_AddColorNoise_max_snr_db_20.wav", help='output wave file name')
    parser.add_argument('-m', '--mode', type=str, default='train', help='type1/type2/type3')
    opt = parser.parse_args()
    if opt.mode == "type1":
        add_GaussianNoise(opt.input_path, opt.output_path)
    elif opt.mode == "type2":
        add_WhiteNoise(opt.input_path, opt.output_path)
    elif opt.mode == "type3":
        add_ShortNoise(opt.input_path, opt.output_path)
    else:
        print('specify mode type like type1 or type2 or type3')
        sys.exit()
