import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.train_batch = 2
Cfg.val_batch = 1
Cfg.lr = 0.001
Cfg.epochs = 70
Cfg.val_interval = 500
Cfg.train_interval = 50
Cfg.gpu_id = '0'
Cfg.weight_decay = 1e-4
Cfg.momentum = 0.9
Cfg.TRAIN_OPTIMIZER = 'adam'
Cfg.CHANNEL = 1
Cfg.num_worker = 4

# bandpass param
Cfg.low_cut = 10000
Cfg.high_cut = 14000
Cfg.bandpass_sr = 30000
# Dataset paths
Cfg.train_normal_path = '../../dataset/train/0/*.wav'
Cfg.train_anomaly_path = '../../dataset/train/1/*.wav'
Cfg.val_normal_path = '../../dataset/test/0/*.wav'
Cfg.val_anomaly_path = '../../dataset/test/1/*.wav'
Cfg.save_checkpoint = True
Cfg.TRAIN_TENSORBOARD_DIR = './logs'
Cfg.output_dir = os.path.join(_BASE_DIR, 'results')
Cfg.ckpt_dir = os.path.join(_BASE_DIR, 'checkpoints')

