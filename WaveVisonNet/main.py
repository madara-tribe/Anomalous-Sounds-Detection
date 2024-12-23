import argparse
from pathlib import Path
import os
import torch
from cfg import Cfg
from models.model import WaveVisonNet
from utils.data_loder import AudioImageDataset
from utils.augmentations import transforms_
from train import Trainer
from predict import Predictor

def main(config, weight_path, opt):
    train_transform, val_transform = transforms_(config)
    train_dataset = AudioImageDataset(config, transform=train_transform, is_validation=None)
    val_dataset = AudioImageDataset(config, transform=val_transform, is_validation=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WaveVisonNet(output_dim=config.num_class).to(device)
    
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    
    if opt.train:
        trainer = Trainer(model, train_dataset, val_dataset, device, config)
        trainer.train(config)
    else:
        Predictor(model, val_dataset, device, config)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='train')
    parser.add_argument('-p', '--predict', action='store_true', help='inference')
    parser.add_argument('-w', '--weight', type=str, default=None, help='model weight')
    opt = parser.parse_args()
    cfg = Cfg
    weight_path = os.path.join(cfg.ckpt_dir, opt.weight) if opt.weight else None
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    main(cfg, weight_path=weight_path, opt=opt)

