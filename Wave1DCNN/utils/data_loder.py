import torch.utils.data as data
import torch
import numpy as np
import glob
import soundfile as sf
from .signal_utils import ConstantQTransform, apply_bandpass
#from .tools import STFTtransform
#stft = STFTtransform(n_fft = 512, sr=16000, fmin=24, fmax=1024, hop_length=None)
#N = 6800

TIME_STEP = 3560
class AudioDataset(data.Dataset):
    def __init__(self, config, transform=None, is_validation=False):
        self.low_cut_freq = config.low_cut
        self.high_cut_freq = config.high_cut
        self.sampling_rate = config.bandpass_sr
        self.channel = config.CHANNEL
        self.is_validation = is_validation
        self.cqt_transformer = ConstantQTransform(sr=8000, fmin=2000, fmax=4000, hop_length=248)
        
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
    

    def sftf_preprocess(self, x, rate, lf, hf, sr):
        x = apply_bandpass(x, lf=lf, hf=hf, order=16, sr=sr)
        x = torch.from_numpy(x).float()
        x = stft.apply(x)
        x = x.to('cpu').detach().numpy().copy()
        x = np.resize(x, (N, self.channel))# (N, self.time_step))
        #print(x.shape)
        return np.transpose(x, (1, 0))
        
        
    def cqt_preprocess(self, x, rate, lf, hf, sr):
        """
        Preprocess the signal data by applying the CQT transform
        and reshaping to (1, CHANNEL, TIME_STEP).
        """
        x = apply_bandpass(x, lf=lf, hf=hf, order=16, sr=sr)
        x = torch.from_numpy(x).float()
        x = self.cqt_transformer.transform(x)
        x = x.to('cpu').detach().numpy().copy()
        x = np.resize(x, (TIME_STEP, self.channel))
        return np.transpose(x, (1, 0))
        
    def __getitem__(self, index):
        path = self.paths[index]
        x, rs = sf.read(path)
        x = self.cqt_preprocess(x, rs, lf=self.low_cut_freq, hf=self.high_cut_freq, sr=self.sampling_rate)
        return x, self.labels[index]

