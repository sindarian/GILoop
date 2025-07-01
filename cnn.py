#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import json
import textwrap
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (Input, Dropout, ReLU, Activation, Conv2D, MaxPooling2D, Reshape,
                                     UpSampling2D, GaussianNoise, Dense, Rescaling)
from tensorflow.keras.layers import concatenate

import tensorflow_addons as tfa

from tensorflow.keras.applications import (EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
                                           EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7)
from tensorflow.keras.layers import Lambda, Resizing
from tensorflow.keras.models import Model

from custom_layers import HiCScale, CombineConcat, ClipByValue
from utils import PATCH_SIZE, get_split_imageset
from metrics import compute_auc



base_models = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4,
               EfficientNetB5, EfficientNetB6, EfficientNetB7]

def prep_data(train_images, val_images, train_y,val_y ):
    # Data preparation (convert to tensors)
    train_images_tensor = tf.convert_to_tensor(train_images, dtype=tf.float32)
    val_images_tensor = tf.convert_to_tensor(val_images, dtype=tf.float32)
    train_x_tensors = [train_images_tensor]
    val_x_tensors = [val_images_tensor]
    flatten_train_y = train_y.reshape((-1, PATCH_SIZE * PATCH_SIZE))[..., np.newaxis]
    flatten_val_y = val_y.reshape((-1, PATCH_SIZE * PATCH_SIZE))[..., np.newaxis]

    return train_x_tensors, val_x_tensors, train_y, val_y, flatten_train_y, flatten_val_y

def predict(model, train_x_tensors, val_x_tensors, test_images):
    train_y_pred = np.asarray(model.predict([train_x_tensors[0]])[1])
    val_y_pred = np.asarray(model.predict([val_x_tensors[0]])[1])
    test_y_pred = np.asarray(model.predict([test_images])[1])

    return train_y_pred, val_y_pred, test_y_pred

def compute_metrics(train_y_pred, train_y, val_y_pred, val_y, test_y_pred, test_y):
    train_auc, train_ap = compute_auc(train_y_pred, train_y.astype('bool'))
    val_auc, val_ap = compute_auc(val_y_pred, val_y.astype('bool'))
    test_auc, test_ap = compute_auc(test_y_pred, test_y.astype('bool'))

    return train_auc, train_ap, val_auc, val_ap, test_auc, test_ap

def print_and_save_metrics(model, run_id, train_auc, train_ap, val_auc, val_ap, test_auc, test_ap):
    print('Train AUC is {}. Train AP is {}.'.format(train_auc, train_ap))
    print('Validation AUC is {}. Validation AP is {}.'.format(val_auc, val_ap))
    print('Test AUC is {}. Test AP is {}.'.format(test_auc, test_ap))

def build_cnn(image_upper_bound, cnn_learning_rate, CNN_METRICS):
    I = Input(shape=(PATCH_SIZE, PATCH_SIZE))
    # x = HiCScale(image_upper_bound)(I)
    x = ClipByValue(image_upper_bound)(I)
    x = Rescaling(1 / image_upper_bound)(x)
    x = Reshape((PATCH_SIZE, PATCH_SIZE, 1))(x)
    x = GaussianNoise(0.05)(x)
    conv1 = Conv2D(32,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    conv1 = Conv2D(32,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(pool1)
    conv2 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv2)
    drop2 = Dropout(0.3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    conv3 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(pool2)
    conv3 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv3)
    drop3 = Dropout(0.3)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    conv4 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(pool3)
    conv4 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(512,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(pool4)
    conv5 = Conv2D(512,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv5)
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2D(256,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_regularizer=tf.keras.regularizers.l2(0.0001))(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(merge6)
    conv6 = Conv2D(256,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv6)

    up7 = Conv2D(128,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_regularizer=tf.keras.regularizers.l2(0.0001))(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(merge7)
    conv7 = Conv2D(128,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv7)
    up8 = Conv2D(64,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_regularizer=tf.keras.regularizers.l2(0.0001))(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(merge8)
    conv8 = Conv2D(64,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv8)
    up9 = Conv2D(32,
                 2,
                 activation='relu',
                 padding='same',
                 kernel_regularizer=tf.keras.regularizers.l2(0.0001))(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(32,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(merge9)
    conv9 = Conv2D(16,
                   3,
                   padding='same',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv9)
    conv10 = conv9

    image_embedding = Reshape((PATCH_SIZE * PATCH_SIZE, -1), name='cnn_embedding')(conv10)
    image_decode = ReLU()(image_embedding)
    image_decode = Dropout(0.3)(image_decode)
    image_decode = Dense(32,
                         activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))(image_decode)
    image_decode = Dropout(0.3)(image_decode)
    image_decode = Dense(16,
                         activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))(image_decode)
    image_decode = Dropout(0.3)(image_decode)
    image_decode = Dense(8,
                         activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))(image_decode)
    image_decode = Dropout(0.3)(image_decode)
    cnn_logits = Dense(1,
                       name='cnn_logits',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(image_decode)
    cnn_sigmoid = Activation('sigmoid',
                             name='cnn_sigmoid')(cnn_logits)

    CNN = Model(inputs=[I], outputs=[cnn_logits, cnn_sigmoid])
    CNN.compile(
        loss={
            'cnn_sigmoid': tfa.losses.SigmoidFocalCrossEntropy(from_logits=False, alpha=0.5, gamma=1.2,
                                                               reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        },
        loss_weights={'cnn_sigmoid': PATCH_SIZE * PATCH_SIZE},
        optimizer=tf.keras.optimizers.Adam(learning_rate=cnn_learning_rate),
        metrics={
            'cnn_sigmoid': CNN_METRICS
        }
    )

    CNN.summary()

    return CNN


def cnn_run(chroms, run_id, seed, dataset_name, epoch=50):
    print(textwrap.dedent(f'''
            #################################################
            #            Building/Training the CNN          #
            #################################################'''))
    dataset_dir = os.path.join('dataset', dataset_name)
    # seed = hash(run_id)
    train_images, train_y, val_images, val_y, test_images, test_y = \
        get_split_imageset(dataset_dir, PATCH_SIZE, seed, chroms)

    image_upper_bound = np.quantile(train_images, 0.996)

    cnn_metrics = [
        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=0.5),
        tf.keras.metrics.AUC(curve="ROC", name='ROC_AUC'),
        tf.keras.metrics.AUC(curve="PR", name='PR_AUC')
    ]

    cnn_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
        0.001,
        2000 * 20,
        end_learning_rate=0.00005,
        power=2.0
    )

    cnn = build_cnn(image_upper_bound, cnn_learning_rate, cnn_metrics)

    # use validation AUC of precision-recall for stopping
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_sigmoid_PR_AUC',
                                                  min_delta=0.0001,
                                                  patience=5,
                                                  verbose=1,
                                                  mode='max')

    train_x_tensors, val_x_tensors, train_y, val_y, flatten_train_y, flatten_val_y = \
        prep_data(train_images, val_images, train_y,val_y )

    # train the CNN
    inputs = [train_x_tensors[0]]
    history = cnn.fit(inputs,
                      y=[flatten_train_y, flatten_train_y],
                      batch_size=8,
                      epochs=epoch,
                      validation_data=([val_x_tensors[0]], [flatten_val_y, flatten_val_y]),
                      callbacks=[early_stop],
                      verbose=2)

    # make predictions
    train_y_pred, val_y_pred, test_y_pred = predict(CNN, train_x_tensors, val_x_tensors, test_images)

    # compute metrics
    train_auc, train_ap, val_auc, val_ap, test_auc, test_ap = \
        compute_metrics(train_y_pred, train_y, val_y_pred, val_y, test_y_pred, test_y)

    # print and save
    print('=' * 30)
    print('*******CNN**********')
    print_and_save_metrics(cnn, run_id, train_auc, train_ap, val_auc, val_ap, test_auc, test_ap)
    with open('metrics/cnn/cnn_training_metrics.json', 'w') as f:
        json.dump(history.history, f)
    with open('metrics/cnn/cnn_computed_metrics.json', 'w') as f:
        json.dump({'train_auc': train_auc,
                        'train_ap':train_ap,
                        'val_auc':val_auc,
                        'val_ap':val_ap,
                        'test_auc':test_auc,
                        'test_ap':test_ap}, f)
    cnn.save('models/cnn.h5')

def efficient_net_run(chroms, run_id, seed, dataset_name, efficient_net_id, adjust_weights=False, epoch=50):
    print(textwrap.dedent(f'''
                #################################################
                #      Building/Training EfficientNet{efficient_net_id}          #
                #################################################'''))

    learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(0.001,
                                                                  2000 * 20,
                                                                  end_learning_rate=0.00005,
                                                                  power=2.0)

    dataset_dir = os.path.join('dataset', dataset_name)

    (train_images, train_y,
     val_images, val_y,
     test_images, test_y) = get_split_imageset(dataset_dir, PATCH_SIZE, seed, chroms)

    (train_x_tensors, val_x_tensors,
     train_y, val_y,
     flatten_train_y, flatten_val_y) = prep_data(train_images, val_images, train_y, val_y)

    image_upper_bound = np.quantile(train_images, 0.996)

    model = build_efficient_net(image_upper_bound,
                                learning_rate,
                                metrics,
                                adjust_weights=adjust_weights,
                                in_shape=in_shape)

    inputs = [train_x_tensors[0]]
    history = model.fit(inputs,
                      y=[flatten_train_y, flatten_train_y],
                      batch_size=8,
                      epochs=epoch,
                      validation_data=([val_x_tensors[0]], [flatten_val_y, flatten_val_y]),
                      verbose=2)

    # make predictions
    train_y_pred, val_y_pred, test_y_pred = predict(model, train_x_tensors, val_x_tensors, test_images)

    # compute metrics
    (train_auc, train_ap,
     val_auc, val_ap,
     test_auc, test_ap) = compute_metrics(train_y_pred, train_y, val_y_pred, val_y, test_y_pred, test_y)

    with open(f'metrics/{model_name}/{model_name}_fit_metrics.json', 'w') as f:
        json.dump(history.history, f)
    with open(f'metrics/{model_name}/{model_name}_computed_metrics.json', 'w') as f:
        json.dump({'train_auc': train_auc,
                        'train_ap':train_ap,
                        'val_auc':val_auc,
                        'val_ap':val_ap,
                        'test_auc':test_auc,
                        'test_ap':test_ap}, f)
    model.save(f'models/{model_name}.h5')

    # print and save
    print('=' * 30)
    print(f'*******{model_name}**********')
    print_and_save_metrics(model, run_id, train_auc, train_ap, val_auc, val_ap, test_auc, test_ap)

def build_efficient_net(image_upper_bound, learning_rate, metrics, adjust_weights=False, in_shape=(224, 224)):
    base_model = EfficientNetB0(include_top=False,
                                weights='imagenet',
                                pooling='max')
    base_model.trainable = adjust_weights

    # Input and preprocessing layers
    I = Input(shape=(PATCH_SIZE, PATCH_SIZE), name='input_image')
    x = ClipByValue(image_upper_bound)(I)
    x = Rescaling(1.0 / image_upper_bound)(x)
    x = Reshape((PATCH_SIZE, PATCH_SIZE, 1))(x)
    x = GaussianNoise(0.05)(x)
    x = concatenate([x, x, x], axis=-1)
    # x = concatenate([I,I,I], axis=-1)
    x = Resizing(in_shape[0], in_shape[1])(x)
    # x = Lambda(preprocess_input, name='efficientnet_preprocess')(x)

    # Encoder layers
    x = base_model(x)
    x = ReLU()(x)
    x = Dense(units=32,
              activation='relu')(x)
    x = Dense(units=16,
              activation='relu')(x)
    x = Dense(units=8,
              activation='relu')(x)
    x = Dense(units=1,
              activation='relu')(x)
    sigmoid = Activation('sigmoid', name='sigmoid')(x)
    efn = Model(inputs=[I], outputs=[sigmoid])

    efn.compile(
        loss={
            'sigmoid': tfa.losses.SigmoidFocalCrossEntropy(from_logits=False,
                                                           alpha=0.5,
                                                           gamma=1.2,
                                                           reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        },
        loss_weights={'sigmoid': PATCH_SIZE * PATCH_SIZE},
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics={'sigmoid': metrics}
    )

    efn.summary()

    return efn
