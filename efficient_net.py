#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import json
import textwrap

from keras.layers import Conv2DTranspose, BatchNormalization, Conv2D
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (Input, Activation, Reshape, GaussianNoise, Dense, Rescaling, ReLU)
from tensorflow.keras.layers import concatenate
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Resizing
from tensorflow.keras.models import Model
from custom_layers import ClipByValue
from metric_configs import LOSS_WEIGHTS, LOSS, OPTIMIZER, METRICS
from utils import PATCH_SIZE, get_split_imageset, FLATTENED_PATCH_SIZE
from metrics import compute_auc
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.regularizers import l2
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from tensorflow.keras.optimizers import Adam
from logger import Logger
import logging

LOGGER = Logger(name='efficient_net', level=logging.DEBUG).get_logger()

efficient_nets = \
{
    0 : {
        'base_model': EfficientNetB0,
        'name' : 'efn0',
        'in_shape' : (224, 224)
    }
}

def efficient_net_run(chroms, seed, dataset_name, efficient_net_id, adjust_weights=False, epoch=50):
    LOGGER.info(f'Building/Training EfficientNet{efficient_net_id}')

    # dataset_dir = os.path.join('dataset', dataset_name)
    dataset_dir = 'dataset/' + dataset_name

    (x_train, y_train,
     x_val, y_val,
     x_test, y_test) = get_split_imageset(dataset_dir, PATCH_SIZE, seed, chroms)
    LOGGER.debug('Data shapes:\n'
                 f'x_train.shape: {x_train.shape}\n'
                 f'y_train.shape: {y_train.shape}\n'
                 f'x_val.shape: {x_val.shape}\n'
                 f'y_val.shape: {y_val.shape}\n'
                 f'x_test.shape: {x_test.shape}\n'
                 f'y_test.shape: {y_test.shape}')

    # x_train_tensors = [(17308, 64, 64)]
    # x_train_tensors[0] = (17308, 64, 64)
    # [x_train_tensors[0]] = [(17308, 64, 64)]

    (x_train_tensors, x_val_tensors,
     y_train, y_val,
     y_train_flatten, y_val_flatten) = reshape_train_val(x_train, y_train, x_val, y_val)
    LOGGER.debug('Data Shapes:\n'
                 f'x_train_tensors.shape: {x_train_tensors[0].shape}\n'
                 f'x_val_tensors.shape: {x_val_tensors[0].shape}\n'
                 f'y_train.shape: {y_train.shape}\n'
                 f'y_val.shape: {y_val.shape}\n'
                 f'y_train_flatten.shape: {y_train_flatten.shape}\n'
                 f'y_val_flatten.shape: {y_val_flatten.shape}')

    LOGGER.debug('Target Value Distribution:\n'
                 f'y_train_flatten: {np.unique(y_train_flatten, return_counts=True)}\n'
                 f'y_val_flatten: {np.unique(y_val_flatten, return_counts=True)}\n'
                 f'y_test: {np.unique(y_test, return_counts=True)}\n')

    LOGGER.debug('Digging into x_train_tensors:\n'
                 f'len(x_train_tensors): {len(x_train_tensors)}\n'
                 f'x_train_tensors.shape: {x_train_tensors[0].shape}\n')

    # find the value that 99.6% of the values fall below
    # likely done to minimize the impact of outliers
    x_train_upper_bound = np.quantile(a=x_train, q=0.996)

    LOGGER.debug(f'Building the model')
    model = build_efficient_net(x_train_upper_bound,
                                efficient_net=efficient_nets[efficient_net_id],
                                adjust_weights=adjust_weights)
    # model = build_efficient_net2(x_train_upper_bound,
    #                             efficient_net=efficient_nets[efficient_net_id],
    #                             adjust_weights=adjust_weights)

    LOGGER.debug(f'Fitting the model')
    history = model.fit(x=x_train_tensors[0],
                        y=y_train_flatten,
                        batch_size=8,
                        epochs=epoch,
                        validation_data=(x_val_tensors[0], y_val_flatten),
                        verbose=1)
    # history = model.fit(x=x_train_tensors[0],
    #                     y=y_train[..., np.newaxis],
    #                     batch_size=8,
    #                     epochs=epoch,
    #                     validation_data=(x_val_tensors[0], y_val[..., np.newaxis]),
    #                     verbose=1)

    # make predictions
    y_pred = np.asarray(model.predict([x_test]))
    LOGGER.debug(f'y_pred.shape: {y_pred.shape}')

    test_auc, test_ap = compute_auc(y_pred, y_test.astype('bool'))
    LOGGER.info(f'test_auc: {test_auc}\n'
                f'test_ap: {test_ap}')
    LOGGER.debug(f'Prediction stats: {np.min(y_pred)}, {np.max(y_pred)}, {np.mean(y_pred)}')

    model_name = efficient_nets[efficient_net_id]['name']
    with open(f'metrics/{model_name}_training_metrics.json', 'w') as f:
        json.dump(history.history, f)
    model.save(f'models/{model_name}.h5')
    LOGGER.info('Test AUC is {}. Test AP is {}.'.format(test_auc, test_ap))

def reshape_train_val(x_train, y_train, x_val, y_val):
    # convert the data to tensors of type float32
    x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_val_tensor = tf.convert_to_tensor(x_val, dtype=tf.float32)

    # add a dimension to the training data tensors
    train_x_tensors = [x_train_tensor]
    val_x_tensors = [x_val_tensor]

    # flatten the training and validation labels to 1D array of size IMAGE_SIZE * IMAGE_SIZE
    # then add a third dimension to make it a 3D tensor
    y_train_flattened = y_train.reshape((-1, PATCH_SIZE * PATCH_SIZE))[..., np.newaxis]
    y_val_flattened = y_val.reshape((-1, PATCH_SIZE * PATCH_SIZE))[..., np.newaxis]

    return train_x_tensors, val_x_tensors, y_train, y_val, y_train_flattened, y_val_flattened

def build_efficient_net(x_train_upper_bound, efficient_net, adjust_weights=False):
    # pull the base model from the dictionary and configure it
    base_model = efficient_net['base_model'](include_top=False,
                                             weights='imagenet',
                                             pooling='max')
                                             #input_shape=(64, 64, 3))
    base_model.trainable = adjust_weights

    # input layers
    I = Input(shape=(PATCH_SIZE, PATCH_SIZE),
              name='input')
    x = ClipByValue(max_hic_value=x_train_upper_bound, # limit the data to a lower and upper bound (removes the outliers)
                    name="clip")(I)
    x = Rescaling(scale=1.0 / x_train_upper_bound,
                  name="rescale")(x)
    x = Reshape(target_shape=(PATCH_SIZE, PATCH_SIZE, 1),
                name="reshape")(x)
    x = GaussianNoise(stddev=.05,
                      name="noise")(x)
    x = concatenate(inputs=[x, x, x],
                    axis=-1)

    # encoder layer
    x = base_model(x)

    # prediction layers
    x = Dense(FLATTENED_PATCH_SIZE, activation='relu')(x)
    x = Reshape((FLATTENED_PATCH_SIZE, 1), name='reshape2')(x)
    out = Activation('sigmoid', name='sigmoid')(x)

    # instantiate the model
    model = Model(inputs=[I], outputs=[out])

    # compile the model
    model.compile(
        loss_weights=LOSS_WEIGHTS,
        loss=LOSS,
        optimizer=OPTIMIZER,
        metrics=METRICS
    )

    # print the network architecture
    model.summary()

    return model

def build_efficient_net2(x_train_upper_bound, efficient_net, adjust_weights=False):
    # pull the base model from the dictionary and configure it
    base_model = efficient_net['base_model'](include_top=False,
                                             weights='imagenet',
                                             pooling='max')
    # input_shape=(64, 64, 3))
    base_model.trainable = adjust_weights

    # input layers
    I = Input(shape=(PATCH_SIZE, PATCH_SIZE),
              name='input')
    x = ClipByValue(max_hic_value=x_train_upper_bound,
                    # limit the data to a lower and upper bound (removes the outliers)
                    name="clip")(I)
    x = Rescaling(scale=1.0 / x_train_upper_bound,
                  name="rescale")(x)
    x = Reshape(target_shape=(PATCH_SIZE, PATCH_SIZE, 1),
                name="reshape")(x)
    x = GaussianNoise(stddev=.05,
                      name="noise")(x)
    x = concatenate(inputs=[x, x, x],
                    axis=-1)

    # encoder layer
    x = base_model(x)
    x = Dense(64 * 64, activation='relu')(x)
    x = Reshape((64, 64, 1),
                name="reshape2")(x)
    out = Activation('sigmoid', name='sigmoid')(x)

    # instantiate the model
    model = Model(inputs=[I], outputs=[out])

    # compile the model
    model.compile(
        loss_weights=LOSS_WEIGHTS,
        loss=LOSS,
        optimizer=OPTIMIZER,
        metrics=METRICS
    )

    # print the network architecture
    model.summary()

    return model