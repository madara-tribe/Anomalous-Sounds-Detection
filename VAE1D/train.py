import logging
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from utils import scheduler
from models.ema import EMA

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
class Trainer:
    def __init__(self, model, train_dataset, val_dataset, device, config):
        self.model = model.to(device)
        self.tfwriter = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR)
        self.criterion = nn.MSELoss()
        self.device = device
        self.epochs = config.epochs
        self.train_loader = data.DataLoader(train_dataset, batch_size=config.train_batch, shuffle=True, num_workers=config.num_worker, pin_memory=True)
        self.val_loader = data.DataLoader(val_dataset, batch_size=config.val_batch, shuffle=True, num_workers=0, pin_memory=None)
        
        print("Train set: {}, Val set: {}".format(len(train_dataset), len(val_dataset)))
   
        self.optimizer = torch.optim.Adam(params=[
            {'params': self.model.parameters(), 'lr': 0.1*config.lr}], lr=config.lr, betas=(0.9, 0.999), eps=1e-08)
        self.scheduler = scheduler.CosineWithRestarts(self.optimizer, t_max=10)
        self.global_step = 0
        self.ema = EMA(self.model.parameters(), decay_rate=0.995, num_updates=0)
        # (Initialize logging)
        logging.info(f'''Starting training:
            Epochs:          {config.epochs}
            Device:          {self.device}
            Learning rate:   {config.lr}
            Training size:   {len(train_dataset)}
            Validation size: {len(val_dataset)}
        ''')
        
    def train_one_epoch(self, config, epoch):
        train_loss = 0
        for inputs, targets in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, targets = inputs.to(self.device, dtype=torch.float32), targets.to(self.device, dtype=torch.float32)
            self.optimizer.zero_grad()
            output = self.model(inputs)
            loss = self.criterion(output, inputs)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
                    
            self.ema.update(self.model.parameters())
            self.global_step += 1
            if self.global_step % config.train_interval == 0:
                self.tfwriter.add_scalar('train/train_loss', train_loss/config.train_interval, self.global_step)
                print(f"Epoch {epoch}, global step: {self.global_step}, train_Loss: {train_loss/config.train_interval:.4f}")
                train_loss = 0
            if self.global_step % config.val_interval == 0:
                val_loss = self.validate()
                self.model.train()
                if val_loss < self.best_loss:
                    self.best_loss = np.round(val_loss, decimals=4)
                    torch.save(self.model.state_dict(), config.ckpt_dir + "/"+ f"checkpoint_epoch{epoch}_{self.best_loss}.pth")
                    print("Model saved!")
                        
    def validate(self):
        val_loss = 0
        self.model.eval()
        print("validating .....")
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs, targets = inputs.to(self.device, dtype=torch.float32), targets.to(self.device, dtype=torch.float32)
                self.ema.store(self.model.parameters())
                self.ema.copy(self.model.parameters())
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                val_loss += loss.item() / len(self.val_loader)
            self.tfwriter.add_scalar('valid/interval_loss', val_loss, self.global_step)
            print(f"global step: {self.global_step}, valid_Loss: {val_loss:.4f}")
        return val_loss
    
    def train(self, config):
        self.best_loss = float('inf')
        for epoch in range(1, self.epochs+1):
            self.train_one_epoch(config, epoch)
            self.scheduler.step()
        torch.save(self.model.state_dict(), config.ckpt_dir + "/"+ "model_last.pth")


