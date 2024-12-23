import torch.utils.data as data
import numpy as np
import glob
from .signal_utils import load_audio_file, convert_waveform_to_image, wave_to_mel_features


class AudioImageDataset(data.Dataset):
    def __init__(self, config, transform=None, is_validation=None):
        #self.low_cut = config.low_cut
        #self.high_cut = config.high_cut
        #self.bandpass_sr = config.bandpass_sr
        self.is_validation = is_validation

        self.transform = transform 
        # Define paths based on training or validation mode
        if not self.is_validation:
            normal_paths = glob.glob(config.train_normal_path)
            anomaly_paths = glob.glob(config.train_anomaly_path)
        else:
            normal_paths = glob.glob(config.val_normal_path)
            anomaly_paths = glob.glob(config.val_anomaly_path)
        
        # Combine normal and anomaly paths
        self.paths = normal_paths + anomaly_paths
        self.labels = [0] * len(normal_paths) + [1] * len(anomaly_paths)
        
        assert len(self.paths) == len(self.labels)

    def __len__(self):
        return len(self.paths)

    def preprocess(self, x, rate):
        """
        Preprocess the signal data by applying the CQT transform
        and reshaping to (1, CHANNEL, TIME_STEP).
        """
        #x = apply_bandpass(x, lf=self.low_cut, hf=self.high_cut, order=16, sr=self.bandpass_sr)
        x = wave_to_mel_features(x, sr=rate)
        x = np.resize(x, (100, 64*3))
        x = convert_waveform_to_image(x)
        return x
    
    def __getitem__(self, index):
        path = self.paths[index]
        wave, rs = load_audio_file(path)
        x = self.preprocess(wave, rs)
#        x.save("img/{}.png".format(index))
        if self.transform is not None:
            x = self.transform(image=np.array(x))['image']
        return x, self.labels[index]


