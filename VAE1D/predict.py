import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, ConfusionMatrixDisplay

class Predictor:
    def __init__(self, model_instance, validation_data, processing_device, configuration):
        self.neural_model = model_instance.to(processing_device)
        self.device = processing_device
        self.validation_loader = data.DataLoader(validation_data, batch_size=configuration.val_batch, shuffle=True, num_workers=0)
        self.results_directory = configuration.output_dir
        os.makedirs(self.results_directory, exist_ok=True)
        self.predict()

    def calculate_thresholds(self, probabilities, target_labels):
        error_minimum = min(probabilities)
        error_maximum = max(probabilities)
        threshold_values = np.linspace(error_minimum, error_maximum, 500)

        optimal_threshold = 0
        highest_f1 = 0
        for threshold in threshold_values:
            predicted_labels = [1 if prob > threshold else 0 for prob in probabilities]
            f1_score_value = f1_score(target_labels, predicted_labels)
            if f1_score_value > highest_f1:
                highest_f1 = f1_score_value
                optimal_threshold = threshold

        return optimal_threshold, highest_f1

    def predict(self):
        error_metrics = []
        target_labels = []

        with torch.no_grad():
            for data_input, data_target in tqdm(self.validation_loader):
                data_input = data_input.to(device=self.device, dtype=torch.float32)
                predictions = self.neural_model(data_input)
                data_target = int(data_target.to('cpu').detach().numpy().copy())
                mse_value = nn.functional.mse_loss(predictions, data_input, reduction='mean').item()
                error_metrics.append(mse_value)
                target_labels.append(data_target)

        probabilities = error_metrics
        optimal_threshold, highest_f1 = self.calculate_thresholds(probabilities, target_labels)

        false_positive_rate, true_positive_rate, _ = roc_curve(target_labels, probabilities, pos_label=1)
        area_under_curve = auc(false_positive_rate, true_positive_rate)

        plt.figure(figsize=(6, 5))
        plt.plot(false_positive_rate, true_positive_rate, label=f'ROC curve (AUC = {area_under_curve:.4f})')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.results_directory, 'roc_curve.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist([probabilities[i] for i in range(len(probabilities)) if target_labels[i] == 0], bins=50, alpha=0.5, label="Normal")
        plt.hist([probabilities[i] for i in range(len(probabilities)) if target_labels[i] == 1], bins=50, alpha=0.5, label="Anomaly")
        plt.axvline(optimal_threshold, color='red', linestyle='--', label='Optimal Threshold')
        plt.xlabel("Probability of Anomaly")
        plt.ylabel("Frequency")
        plt.title("Histogram of Anomaly Probabilities")
        plt.legend()
        plt.savefig(os.path.join(self.results_directory, 'anomaly_histogram.png'))
        plt.close()

        final_predictions = [1 if prob > optimal_threshold else 0 for prob in probabilities]
        confusion_mat = confusion_matrix(target_labels, final_predictions)
        ConfusionMatrixDisplay(confusion_mat).plot()
        plt.savefig(os.path.join(self.results_directory, 'confusion_matrix.png'))
        plt.close()


