import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_pr_auc', 
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)

metrics = [keras.metrics.AUC(curve='PR', name='pr_auc')]


def make_model_baseline(output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    
    model = keras.Sequential([
                                keras.layers.Dense(32,
                                                   activation='relu',
                                                   input_shape=(train_features.shape[-1],)
                                                   ),
                                keras.layers.Dense(1,
                                                   activation='sigmoid',
                                                   bias_initializer=output_bias),
                                ])

    model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics)

    return model


def make_model_baseline_batchnorm_drouput(output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    
    model = keras.Sequential([
                                keras.layers.Dense(32,
                                                   activation='relu',
                                                   input_shape=(train_features.shape[-1],)
                                                   ),
                                keras.layers.BatchNormalization(),
                                keras.layers.Dropout(0.7),

                                keras.layers.Dense(1,
                                                   activation='sigmoid',
                                                   bias_initializer=output_bias),
                                ])

    model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics)

    return model