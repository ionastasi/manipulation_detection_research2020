import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import *


BUFFER_SIZE = 100000
EPOCHS = 100
BATCH_SIZE = 2048


def train_imbalanced_model(make_model, early_stopping, initial_weights):
    imbalanced_model = make_model()
    imbalanced_model.load_weights(initial_weights)

    imbalanced_history = imbalanced_model.fit(
            train_features,
            train_labels,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks = [early_stopping],
            validation_data=(val_features, val_labels),
            verbose=0)

    return imbalanced_model, imbalanced_history


def train_weighted_model(make_model, early_stopping, initial_weights):
    weighted_model = make_model()
    weighted_model.load_weights(initial_weights)

    weighted_history = weighted_model.fit(
            train_features,
            train_labels,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks = [early_stopping],
            validation_data=(val_features, val_labels),
            class_weight=class_weight,
            verbose=0) 

    return weighted_model, weighted_history


def train_resampled_model(make_model, early_stopping, initial_weights):
    resampled_model = make_model()
    resampled_model.load_weights(initial_weights)

    output_layer = resampled_model.layers[-1] 
    output_layer.bias.assign([0])

    val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(2) 

    resampled_history = resampled_model.fit(
            resampled_ds,
            steps_per_epoch = 20,
            epochs=10*EPOCHS,
            callbacks = [early_stopping],
            validation_data=(val_ds),
            verbose=0)

    return resampled_model, resampled_history


def run_imbalanced_weigthed_oversampled(make_model, early_stopping):
    # inital weights
    initial_bias = np.log([pos/neg])
    model = make_model(output_bias = initial_bias)
    model.predict(train_features[:10])
    initial_weights = os.path.join(tempfile.mkdtemp(),'initial_weights')
    model.save_weights(initial_weights)
        
    imbalanced_model, _ = train_imbalanced_model(make_model, early_stopping, initial_weights)
    weighted_model, _ = train_weighted_model(make_model, early_stopping, initial_weights)
    resampled_model, _ = train_resampled_model(make_model, early_stopping, initial_weights)

    return imbalanced_model, weighted_model, resampled_model


def __make_ds(features, labels):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))#.cache()
    ds = ds.shuffle(BUFFER_SIZE).repeat()
    return ds


def main(input_data='aggregated_ohlcv_labeled.csv'):

    features_labeled_df = pd.read_csv(input_data)

    train_df, test_df = train_test_split(features_labeled_df, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)

    train_labels = np.array(train_df.pop('Label'))
    bool_train_labels = train_labels != 0
    val_labels = np.array(val_df.pop('Label'))
    test_labels = np.array(test_df.pop('Label'))

    train_features = np.array(train_df)
    val_features = np.array(val_df)
    test_features = np.array(test_df)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)

    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    train_features = np.clip(train_features, -5, 5)
    val_features = np.clip(val_features, -5, 5)
    test_features = np.clip(test_features, -5, 5)

    pos_features = train_features[bool_train_labels]
    neg_features = train_features[~bool_train_labels]

    pos_labels = train_labels[bool_train_labels]
    neg_labels = train_labels[~bool_train_labels]

    pos_ds = __make_ds(pos_features, pos_labels)
    neg_ds = __make_ds(neg_features, neg_labels)

    resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
    resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)

    for features, label in resampled_ds.take(1):
        print(label.numpy().mean())



    weight_for_0 = (1 / neg)*(total)/2.0 
    weight_for_1 = (1 / pos)*(total)/2.0


    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(class_weight)


    _ = run_imbalanced_weigthed_oversampled(make_model_baseline)


