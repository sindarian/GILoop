from attention_u_net import attention_net_run
from cnn import cnn_run
from efficient_net import efficient_net_run
from tensorflow.keras import backend as K
from logger import Logger
import logging

LOGGER = Logger(name='train', level=logging.DEBUG).get_logger()


def train_run_cnn(chroms, run_id, seed, source_dataset_name, epochs=50):
    cnn_run(chroms, run_id, seed, source_dataset_name, epoch=epochs)
    K.clear_session()

def train_efficient_net(chroms, seed, dataset_name, epochs=50):
    # efficient net ids to iterate over and train
    efficient_nets = [0]
    LOGGER.debug(f'Efficient nets to train: {efficient_nets}')

    for net in efficient_nets:
        LOGGER.debug(f'Training efficient net {net}')
        efficient_net_run(chroms,
                          seed,
                          dataset_name,
                          efficient_net_id=net,
                          adjust_weights=True,
                          epoch=epochs)
        K.clear_session()

def train_attention_u_net(chroms, seed, dataset_name, epochs=50):
    LOGGER.debug(f'Training AttentionNet')
    attention_net_run(chroms, seed, dataset_name, epochs)