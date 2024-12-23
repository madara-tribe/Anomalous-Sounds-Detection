import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.train_batch = 4
Cfg.val_batch = 1
Cfg.input_size = 228
Cfg.lr = 0.001
Cfg.epochs = 100
Cfg.num_worker = 16
Cfg.val_interval = 400
Cfg.train_interval = 50
Cfg.gpu_id = '0'
Cfg.weight_decay = 1e-4
Cfg.momentum = 0.9
Cfg.TRAIN_OPTIMIZER = 'adam'
Cfg.num_class = 2

# bandpass param
Cfg.low_cut = 10000
Cfg.high_cut = 14000
Cfg.bandpass_sr = 30000

# Dataset paths
Cfg.train_normal_path = '../../dataset1209/train/0/*.wav'
Cfg.train_anomaly_path = '../../dataset1209/train/1/*.wav'
Cfg.val_normal_path = '../../dataset1209/test/0/*.wav'
Cfg.val_anomaly_path = '../../dataset1209/test/1/*.wav'
Cfg.save_checkpoint = True
Cfg.TRAIN_TENSORBOARD_DIR = './logs'
Cfg.ckpt_dir = os.path.join(_BASE_DIR, 'checkpoints')

