import sys
import os
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils import data
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


class Predictor:
    def __init__(self, model, val_dataset, device, config):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.val_loader = data.DataLoader(val_dataset, batch_size=config.val_batch, shuffle=True, num_workers=0, pin_memory=None)
        self.th = 0.5
        self.predict(config)
        print("Test set: %d" % len(val_dataset))
        y_test, y_pred = self.predict(config)
        self.evaluate(y_test, y_pred, config)

    def predict(self, config):
        y_pred, y_test = [], []
        start = time.time()
        with torch.no_grad():
            for input, target in tqdm(self.val_loader):
                input = input.to(device=self.device, dtype=torch.float32)
                pred = self.model(input)
                label = int(target.to('cpu').detach().numpy().copy())
                pred = int(torch.argmax(pred.softmax(dim=-1)).to('cpu').detach().numpy().copy())
                y_test.append(label)
                y_pred.append(pred)
            
        latency = time.time() - start
        print(f'Prediction Latency: {latency}')
        return y_test, y_pred

    def evaluate(self, y_test, y_pred, config): 

        y_pred = np.array(y_pred)
        y_test = np.array(y_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='micro')

        # Calculate ROC and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC (ROC): {roc_auc:.4f}")

        # Save ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.4f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        roc_path = os.path.join(config.ckpt_dir, 'roc_curve.png')
        plt.savefig(roc_path)
        print(f"ROC curve saved at {roc_path}")
        
        # Compute and plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(config.ckpt_dir, 'confusion_matrix.png'))
        print("Confusion matrix saved as 'confusion_matrix.png'")     


