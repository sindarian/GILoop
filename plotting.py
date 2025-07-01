import json
import matplotlib.pyplot as plt
import os
import logging
from logger import Logger

LOGGER = Logger(name='plotting', level=logging.DEBUG).get_logger()

def plot_training_history(metric_dir):
    files = [f for f in os.listdir(metric_dir) if os.path.isfile(os.path.join(metric_dir, f)) and f.endswith('.json')]
    LOGGER.debug(f'Given metric dir: {metric_dir}')
    LOGGER.debug(f'files in {metric_dir}: {files}')

    histories = []
    for file in files:
        with open(os.path.join(metric_dir, file), 'r') as f:
            history = json.load(f)
            histories.append(history)

    # 4 rows x 2 columns
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    colors = ['blue', 'green', 'red', 'purple', 'orange']
    labels = [file.split('_')[0] for file in files]
    metrics = [key for key in histories[0].keys()]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, history in enumerate(histories):
            ax.plot(history[metric], label=labels[j], color=colors[j], alpha=0.7)
        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True)
        if i == 0:
            ax.legend(loc='upper right')

    plt.tight_layout()
    # plt.show()
    plt.savefig('plots/fig.png')