import argparse
from pathlib import Path
import os
import torch
from cfg import Cfg
from models.model import CNN1D
from utils.data_loder import DataLoader
from train import Trainer
from predict import Predictor

def main(cfg, weight_path, mode="train"):
    train_dataset = DataLoader(cfg, transform=True, valid=None)
    val_dataset = DataLoader(cfg, transform=True, valid=True)
    model = CNN1D(in_channels=cfg.CHANNEL)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path))
    if mode=="train":
        trainer = Trainer(model, train_dataset, val_dataset, device, cfg)
        trainer.train(cfg)
    else:
        Predictor(model, val_dataset, device, cfg)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', help='train')
    parser.add_argument('-p', '--predict', action='store_true', help='inference')
    parser.add_argument('-w', '--weight', type=str, default=None, help='model weight')
    opt = parser.parse_args()
    cfg = Cfg
    weight_path = os.path.join(cfg.ckpt_dir, opt.weight) if opt.weight else None
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    if opt.train:
        main(cfg, weight_path=weight_path, mode="train")
    elif opt.predict:
        main(cfg, weight_path=weight_path, mode="predict")
    else:
        pass
