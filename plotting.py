import json
import matplotlib.pyplot as plt
import os
import logging
from logger import Logger

LOGGER = Logger(name='plotting', level=logging.DEBUG).get_logger()

PLOT_DIR = 'plots/'

def plot_training_history(metric_dir):
    # files = [f for f in os.listdir(metric_dir) if os.path.isfile(os.path.join(metric_dir, f)) and f.endswith('.json')]
    # LOGGER.debug(f'Given metric dir: {metric_dir}')
    # LOGGER.debug(f'files in {metric_dir}: {files}')

    # histories, model_names = load_histories(metric_dir, files)
    # paired_metrics = create_metric_pairs(histories)
    # plot_metrics(histories, paired_metrics, model_names)

    for f in os.listdir(metric_dir):
        if os.path.isfile(os.path.join(metric_dir, f)) and f.endswith('.json'):
            histories, model_names = load_histories(metric_dir, [f])
            paired_metrics = create_metric_pairs(histories)
            plot_metrics(histories, paired_metrics, model_names)


def load_histories(metric_dir, files):
    # load the saved model histories into a dictionary structure
    histories = []
    model_names = []
    for file in files:
        with open(os.path.join(metric_dir, file), 'r') as f:
            history = json.load(f)
            histories.append(history)
        model_names.append(file.split('_')[0])
    return histories, model_names

def create_metric_pairs(histories):
    # gather all the similar metrics into a list of tuples: [(loss, val_loss), ...]
    metrics = list(histories[0].keys())
    paired_metrics = []
    for i, metric in enumerate(metrics[:int(len(metrics) / 2)]):
        # the index of the corresponding validation metric is:
        # the current index (the training metric) + half the size of the total number of metrics
        v = int(i + (len(metrics) / 2))
        paired_metrics.append((metric, metrics[v]))
    LOGGER.debug(f'paired_metrics: {paired_metrics}')
    return paired_metrics

def plot_metrics(histories, paired_metrics, model_names):
    LOGGER.debug(f'Plotting metrics for {model_names}')
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    axes = axes.flatten()

    # iterate over all histories and plot the training and validation metrics for that axis
    for model_index, history in enumerate(histories):
        for i, pair in enumerate(paired_metrics):
            ax = axes[i]
            ax.plot(history[pair[0]], label='train')
            ax.plot(history[pair[1]], label='val')
            ax.set_title(pair[0])
            ax.set_ylabel(pair[0])
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(True)

        # turn off any plots that weren't used
        for ax in axes:
            if not ax.lines:
                ax.axis('off')

        LOGGER.info(f'Saving plot to {PLOT_DIR + model_names[model_index]}.png')
        plt.savefig(PLOT_DIR + f'{model_names[model_index]}.png')
