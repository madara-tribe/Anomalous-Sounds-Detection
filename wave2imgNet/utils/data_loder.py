import torch.utils.data as data
import torch
import numpy as np
import glob
import soundfile as sf
from .signal_utils import CQTtransform, apply_bandpass, wave2image


class DataLoader(data.Dataset):
    def __init__(self, config, transform=None, valid=None):
        self.low_cut = config.low_cut
        self.high_cut = config.high_cut
        self.bandpass_sr = config.bandpass_sr
        self.valid = valid
        self.cqt_ = CQTtransform(sr=8000, fmin=2000, fmax=4000, hop_length=124)

        self.transform = transform 
        # Define paths based on training or validation mode
        if not self.valid:
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
        x = apply_bandpass(x, lf=self.low_cut, hf=self.high_cut, order=16, sr=self.bandpass_sr)
        x = torch.from_numpy(x).float()
        x = self.cqt_.transform(x)
        x = wave2image(x.to('cpu').detach().numpy().copy())
        return x
    def __getitem__(self, index):
        path = self.paths[index]
        wave, rs = sf.read(path)
        x = self.preprocess(wave, rs)
        #x.save("img/{}.png".format(index))
        if self.transform is not None:
            x = self.transform(image=np.array(x))['image']
        #print(x.shape, x.min(), x.max())
        return x, self.labels[index]


