import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, 'results')
output_dir = os.path.join(base_dir, 'confusion_matrixes')

os.makedirs(output_dir, exist_ok=True)

def plot_confusion_matrix(conf_matrix, title, output_path):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

for split in os.listdir(results_dir):
    split_dir = os.path.join(results_dir, split, 'neural_networks')
    if os.path.isdir(split_dir):
        for model_file in os.listdir(split_dir):
            if model_file.endswith('.json'):
                model_path = os.path.join(split_dir, model_file)
                with open(model_path, 'r') as f:
                    data = json.load(f)
                    conf_matrix = np.array(data['confusion_matrix'])
                    model_name = data['metadata']['model']
                    split_strategy = data['metadata']['split_strategy']
                    title = f"{model_name} - {split_strategy}"
                    output_path = os.path.join(output_dir, f"{model_name}_{split_strategy}.png")
                    plot_confusion_matrix(conf_matrix, title, output_path)