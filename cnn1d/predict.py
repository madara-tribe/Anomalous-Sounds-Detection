import os
from tqdm import tqdm
import torch
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, ConfusionMatrixDisplay



class Predictor:
    def __init__(self, model, val_dataset, device, config):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.val_loader = data.DataLoader(val_dataset, batch_size=config.val_batch, shuffle=True, num_workers=0, pin_memory=None)
        self.th = 0.5
        self.predict(config)
        
    def predict(self, config):
        y_label, y_preds = [], []
        acc, nums = 0, 0
        print("predicting  .....")
        with torch.no_grad():
            for input, target in tqdm(self.val_loader):
                input = input.to(device=self.device, dtype=torch.float32)
                pred = self.model(input)
                target = int(target.to('cpu').detach().numpy().copy())
                #print(y_preds, y_preds.sigmoid().to('cpu').numpy(), target)
                pred = pred.sigmoid().to('cpu').numpy()
                pred = 1 if pred > self.th else 0
                #print(str(target), str(pred_idx), target, pred_idx)
                acc += 1 if pred==target else 0
                y_label.append(target)
                y_preds.append(pred)
                nums += 1
        # Compute metrics
        y_label = np.array(y_label)
        y_preds = np.array(y_preds)

        fpr, tpr, _ = roc_curve(y_label, y_preds)
        roc_auc = auc(fpr, tpr)
        f1 = f1_score(y_label, (y_preds > self.th).astype(int))

        print("Accuracy is: {:.4f}".format(acc / nums))
        print("AUC (ROC): {:.4f}".format(roc_auc))
        print("F1 Score: {:.4f}".format(f1))
     
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
        plt.savefig(os.path.join(config.ckpt_dir, 'roc_curve.png'))
        print("ROC curve saved as 'roc_curve.png'")

        # Compute and plot confusion matrix
        cm = confusion_matrix(y_label, y_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(config.ckpt_dir, 'confusion_matrix.png'))
        print("Confusion matrix saved as 'confusion_matrix.png'")
 

