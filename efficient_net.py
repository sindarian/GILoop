#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import json
import textwrap

from keras.layers import Conv2DTranspose, BatchNormalization, Conv2D, Flatten, UpSampling2D, GlobalAveragePooling2D, \
    Multiply, Concatenate, Attention, Add
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (Input, Activation, Reshape, GaussianNoise, Dense, Rescaling, ReLU)
from tensorflow.keras.layers import concatenate
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Resizing
from tensorflow.keras.models import Model

from CBAM import CBAM
from attention_u_net import AttentionGate
from custom_layers import ClipByValue
from metric_configs import LOSS_WEIGHTS, LOSS, OPTIMIZER, METRICS, EARLY_STOP
from utils import PATCH_SIZE, get_split_imageset, FLATTENED_PATCH_SIZE
from metrics import compute_auc
from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision, Recall
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

def run_efficient_net(chroms, seed, dataset_name, efficient_net_id, adjust_weights=False, epoch=50):
    LOGGER.info(f'Building/Training EfficientNet{efficient_net_id}')

    dataset_dir = 'dataset_ps64_res10000/' + dataset_name

    (x_train, y_train,
     x_val, y_val,
     x_test, y_test) = get_split_imageset(dataset_dir, PATCH_SIZE, seed, chroms)

    (x_train_tensors, x_val_tensors,
     y_train, y_val,
     y_train_flatten, y_val_flatten) = split_shape_inspect_data(dataset_dir, seed, chroms)

    # find the value that 99.6% of the values fall below
    # likely done to minimize the impact of outliers
    x_train_upper_bound = np.quantile(a=x_train, q=0.996)

    LOGGER.debug(f'Building the model')
    model = build_efficient_net(x_train_upper_bound,
                                    efficient_net=efficient_nets[efficient_net_id],
                                    adjust_weights=adjust_weights)

    LOGGER.debug(f'Fitting the model')
    history = model.fit(x=x_train_tensors[0],
                        y=y_train_flatten,
                        batch_size=8,
                        epochs=epoch,
                        validation_data=(x_val_tensors[0], y_val_flatten),
                        callbacks=[EARLY_STOP],
                        verbose=1)

    # # make predictions
    model_name = efficient_nets[efficient_net_id]['name']
    predict_and_save(model, x_test, y_test, model_name, history)

def build_efficient_net(x_train_upper_bound, efficient_net, adjust_weights=False):
    # pull the base model from the dictionary and configure it
    base_model = efficient_net['base_model'](include_top=False,
                                             weights='imagenet',
                                             pooling='max')
    base_model.trainable = adjust_weights

    # input layers
    I = Input(shape=(PATCH_SIZE, PATCH_SIZE))
    x = ClipByValue(max_hic_value=x_train_upper_bound)(I) # limit the data to a lower and upper bound (removes the outliers)
    x = Rescaling(scale=1.0 / x_train_upper_bound, )(x)
    x = Reshape(target_shape=(PATCH_SIZE, PATCH_SIZE, 1), )(x)
    x = GaussianNoise(stddev=.05, )(x)
    x = concatenate(inputs=[x, x, x], axis=-1)

    # efficient net b0
    x = base_model(x)

    # prediction layers
    x = Dense(FLATTENED_PATCH_SIZE, activation='relu')(x)
    x = Reshape((FLATTENED_PATCH_SIZE, 1))(x)
    out = Activation('sigmoid', name='sigmoid')(x)

    # instantiate the model
    model = Model(inputs=[I], outputs=[out])

    # compile the model
    model.compile(
        loss_weights={'sigmoid': PATCH_SIZE * PATCH_SIZE},
        loss={'sigmoid': SigmoidFocalCrossEntropy(from_logits=False,
                                                  alpha=0.98,
                                                  gamma=2.0,
                                                  reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)},
        optimizer=Adam(learning_rate=PolynomialDecay(initial_learning_rate=0.00001,
                                                     decay_steps=2000 * 20,
                                                     end_learning_rate=0.00005,
                                                     power=2.0)),
        metrics={'sigmoid': [
                                BinaryAccuracy(name='binary_accuracy', threshold=0.9),
                                Precision(name='precision'),
                                Recall(name='recall'),
                                AUC(curve="ROC", name='ROC_AUC'),
                                AUC(curve="PR", name='PR_AUC')
                             ]
                 }
    )

    # print the network architecture
    model.summary()

    return model

def run_cbam_efficient_net(chroms, seed, dataset_name, efficient_net_id, adjust_weights=False, epoch=50):
    dataset_dir = 'dataset_ps64_res10000/' + dataset_name

    (x_train, y_train,
     x_val, y_val,
     x_test, y_test) = get_split_imageset(dataset_dir, PATCH_SIZE, seed, chroms)
    (x_train_tensors, x_val_tensors,
     y_train, y_val,
     y_train_flatten, y_val_flatten) = split_shape_inspect_data(dataset_dir, seed, chroms)

    # find the value that 99.6% of the values fall below
    # likely done to minimize the impact of outliers
    x_train_upper_bound = np.quantile(a=x_train, q=0.996)

    LOGGER.debug(f'Building the model')
    model = build_cbam_efficient_net(x_train_upper_bound,
                                     efficient_net=efficient_nets[efficient_net_id],
                                     adjust_weights=adjust_weights)

    LOGGER.debug(f'Fitting the model')
    history = model.fit(x=x_train_tensors[0],
                        y=tf.reshape(y_train, (-1, 64, 64, 1)),
                        batch_size=8,
                        epochs=epoch,
                        validation_data=(x_val_tensors[0], tf.reshape(y_val, (-1, 64, 64, 1))),
                        callbacks=[EARLY_STOP],
                        verbose=1)

    # make predictions
    model_name = 'cbam_' + efficient_nets[efficient_net_id]['name']
    predict_and_save(model, x_test, y_test, model_name, history)

def build_cbam_efficient_net(x_train_upper_bound, efficient_net, adjust_weights=False):
    # pull the base model from the dictionary and configure it
    base_model = efficient_net['base_model'](include_top=False,
                                             weights='imagenet')
                                             #pooling=None) # pooling flattens the shape
    base_model.trainable = adjust_weights

    # Inputs
    I = Input(shape=(64, 64, 1))
    x = ClipByValue(max_hic_value=x_train_upper_bound)(I) # limit the data to a lower and upper bound (removes the outliers)
    x = Rescaling(scale=1.0 / x_train_upper_bound)(x)
    x = Reshape(target_shape=(PATCH_SIZE, PATCH_SIZE, 1))(x)
    x = GaussianNoise(stddev=.05)(x)
    x = Concatenate()([x, x, x])

    # EfficientNetB0
    x = base_model(x)
    # CBAM
    x = CBAM()(x)

    # Decoder
    x = Conv2DTranspose(512, 3, strides=2, padding='same')(x)
    x = CBAM()(x)
    x = Conv2DTranspose(256, 3, strides=2, padding='same')(x)
    x = CBAM()(x)
    x = Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = CBAM()(x)
    x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    outputs = Conv2D(1, kernel_size=1, activation='sigmoid')(x)

    model = Model(inputs=I, outputs=outputs)

    # compile the model
    model.compile(
        loss_weights=PATCH_SIZE * PATCH_SIZE,
        loss=SigmoidFocalCrossEntropy(from_logits=False,
                                            alpha=0.98,
                                            gamma=2.0,
                                            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
        optimizer=Adam(learning_rate=PolynomialDecay(initial_learning_rate=0.00001,
                                                     decay_steps=2000 * 20,
                                                     end_learning_rate=0.00005,
                                                     power=2.0)),
        metrics=[BinaryAccuracy(name='binary_accuracy', threshold=0.9),
                             AUC(curve="ROC", name='ROC_AUC'),
                             AUC(curve="PR", name='PR_AUC')
                ]

    )

    # print the network architecture
    model.summary()

    return model

def run_attention_efficient_net(chroms, seed, dataset_name, efficient_net_id, adjust_weights=False, epoch=50):
    LOGGER.info(f'Building/Training Attention Enhanced EfficientNet{efficient_net_id}')
    dataset_dir = 'dataset_ps64_res10000/' + dataset_name

    (x_train, y_train,
     x_val, y_val,
     x_test, y_test) = get_split_imageset(dataset_dir, PATCH_SIZE, seed, chroms)

    (x_train_tensors, x_val_tensors,
     y_train, y_val,
     y_train_flatten, y_val_flatten) = split_shape_inspect_data(dataset_dir, seed, chroms)

    # find the value that 99.6% of the values fall below
    # likely done to minimize the impact of outliers
    x_train_upper_bound = np.quantile(a=x_train, q=0.996)

    LOGGER.debug(f'Building the model')
    model = build_attention_efficient_net(x_train_upper_bound,
                                    efficient_net=efficient_nets[efficient_net_id],
                                    adjust_weights=adjust_weights)

    LOGGER.debug(f'Fitting the model')
    history = model.fit(x=x_train_tensors[0],
                        y=y_train_flatten,
                        batch_size=8,
                        epochs=epoch,
                        validation_data=(x_val_tensors[0], y_val_flatten),
                        callbacks=[EARLY_STOP],
                        verbose=1)

    # # make predictions
    model_name = 'attention_' + efficient_nets[efficient_net_id]['name']
    predict_and_save(model, x_test, y_test, model_name, history)

def build_attention_efficient_net(x_train_upper_bound, efficient_net, adjust_weights=False):
    # pull the base model from the dictionary and configure it
    base_model = efficient_net['base_model'](include_top=False,
                                             weights='imagenet',
                                             pooling='max')
    base_model.trainable = adjust_weights

    # input layers
    I = Input(shape=(PATCH_SIZE, PATCH_SIZE))
    x = ClipByValue(max_hic_value=x_train_upper_bound)(I) # limit the data to a lower and upper bound (removes the outliers)
    x = Rescaling(scale=1.0 / x_train_upper_bound)(x)
    x = Reshape(target_shape=(PATCH_SIZE, PATCH_SIZE, 1))(x)
    x = GaussianNoise(stddev=.05)(x)
    x = concatenate(inputs=[x, x, x], axis=-1)

    # efficient net b0
    x = base_model(x)
    x = Dense(FLATTENED_PATCH_SIZE)(x)
    x = tf.reshape(x, (-1, PATCH_SIZE, PATCH_SIZE, 1))

    x = Flatten()(x)
    x = Reshape((FLATTENED_PATCH_SIZE, 1))(x)
    x = Attention()([x, x])
    out = Activation('sigmoid', name='sigmoid')(x)

    # instantiate the model
    model = Model(inputs=[I], outputs=[out])

    # compile the model
    model.compile(
        loss_weights={'sigmoid': PATCH_SIZE * PATCH_SIZE},
        loss={'sigmoid': SigmoidFocalCrossEntropy(from_logits=False,
                                                  alpha=0.98,
                                                  gamma=2.0,
                                                  reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)},
        optimizer=Adam(learning_rate=PolynomialDecay(initial_learning_rate=0.00001,
                                                     decay_steps=2000 * 20,
                                                     end_learning_rate=0.00005,
                                                     power=2.0)),
        metrics={'sigmoid': [
                                BinaryAccuracy(name='binary_accuracy', threshold=0.9),
                                AUC(curve="ROC", name='ROC_AUC'),
                                AUC(curve="PR", name='PR_AUC')
                             ]
                 }
    )

    # print the network architecture
    model.summary()

    return model

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

def split_shape_inspect_data(dataset_dir, seed, chroms):
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

    return x_train_tensors, x_val_tensors, y_train, y_val, y_train_flatten, y_val_flatten

def predict_and_save(model, x_test, y_test, model_name, history):
    # make predictions
    y_pred = np.asarray(model.predict([x_test]))
    LOGGER.debug(f'y_pred.shape: {y_pred.shape}')

    test_auc, test_ap = compute_auc(y_pred, y_test.astype('bool'))
    LOGGER.info(f'test_auc: {test_auc}\n'
                f'test_ap: {test_ap}')
    LOGGER.debug(f'Prediction stats: {np.min(y_pred)}, {np.max(y_pred)}, {np.mean(y_pred)}')
    with open(f'metrics/{model_name}_training_metrics.json', 'w') as f:
        json.dump(history.history, f)
    model.save(f'models/{model_name}.h5')
    LOGGER.info('Test AUC is {}. Test AP is {}.'.format(test_auc, test_ap))

# last best model
# def build_attention_efficient_net_2(x_train_upper_bound, efficient_net, adjust_weights=False):
#     # pull the base model from the dictionary and configure it
#     base_model = efficient_net['base_model'](include_top=False,
#                                              weights='imagenet',
#                                              pooling='max')
#     base_model.trainable = adjust_weights
#
#     # input layers
#     I = Input(shape=(PATCH_SIZE, PATCH_SIZE))
#     x = ClipByValue(max_hic_value=x_train_upper_bound)(I) # limit the data to a lower and upper bound (removes the outliers)
#     x = Rescaling(scale=1.0 / x_train_upper_bound)(x)
#     x = Reshape(target_shape=(PATCH_SIZE, PATCH_SIZE, 1))(x)
#     x = GaussianNoise(stddev=.05)(x)
#     x = concatenate(inputs=[x, x, x], axis=-1)
#
#     # efficient net b0
#     x = base_model(x)
#     x = Dense(FLATTENED_PATCH_SIZE)(x)
#     x = tf.reshape(x, (-1, PATCH_SIZE, PATCH_SIZE, 1))
#     x = AttentionGate(inter_channels=256)(x)
#     # x = AttentionGate(inter_channels=64 // 2)(x)
#
#     # prediction layers
#     x = Flatten()(x)
#     x = Reshape((FLATTENED_PATCH_SIZE, 1))(x)
#     x = Dense(1)(x)
#     out = Activation('sigmoid', name='sigmoid')(x)
#
#     # instantiate the model
#     model = Model(inputs=[I], outputs=[out])
#
#     # compile the model
#     model.compile(
#         # loss_weights={'sigmoid': PATCH_SIZE * PATCH_SIZE},
#         loss={'sigmoid': SigmoidFocalCrossEntropy(from_logits=False,
#                                                   alpha=0.98,
#                                                   gamma=2.0,
#                                                   reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)},
#         optimizer=Adam(learning_rate=PolynomialDecay(initial_learning_rate=0.00001,
#                                                      decay_steps=2000 * 20,
#                                                      end_learning_rate=0.00005,
#                                                      power=2.0)),
#         metrics={'sigmoid': [
#                                 BinaryAccuracy(name='binary_accuracy', threshold=0.9),
#                                 Precision(name='precision'),
#                                 Recall(name='recall'),
#                                 AUC(curve="ROC", name='ROC_AUC'),
#                                 AUC(curve="PR", name='PR_AUC')
#                              ]
#                  }
#     )
#
#     # print the network architecture
#     model.summary()
#
#     return model