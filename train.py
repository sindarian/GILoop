from attention_u_net import attention_net_run
from cnn import cnn_run
from efficient_net import run_efficient_net, run_cbam_efficient_net, run_attention_efficient_net
from tensorflow.keras import backend as K
from logger import Logger
import logging

LOGGER = Logger(name='train', level=logging.DEBUG).get_logger()


def train_run_cnn(chroms, run_id, seed, source_dataset_name, epochs=50):
    cnn_run(chroms, run_id, seed, source_dataset_name, epoch=epochs)
    K.clear_session()

def train_efficient_net(chroms, seed, dataset_name, epochs=50):
    LOGGER.debug(f'Training EfficientNetB0')
    run_efficient_net(chroms,
                      seed,
                      dataset_name,
                      efficient_net_id=0,
                      adjust_weights=True,
                      epoch=epochs)
    K.clear_session()

def train_cbam_efficient_net(chroms, seed, dataset_name, epochs=50):
    LOGGER.debug(f'Training CBAM_EfficientNetB0')
    run_cbam_efficient_net(chroms,
                           seed,
                           dataset_name,
                           efficient_net_id=0,
                           adjust_weights=True,
                           epoch=epochs)
    K.clear_session()

def train_attention_efficient_net(chroms, seed, dataset_name, epochs=50):
    LOGGER.debug(f'Training Attention_EfficientNetB0')
    run_attention_efficient_net(chroms,
                               seed,
                               dataset_name,
                               efficient_net_id=0,
                               adjust_weights=True,
                               epoch=epochs)
    K.clear_session()