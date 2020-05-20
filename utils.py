import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pr(name, labels, predictions, title, **kwargs):
    # set design stuff
    ax = plt.subplot(111)   
    ax.spines["top"].set_visible(False)   
    ax.spines["right"].set_visible(False)
    plt.grid(True, color="#93a1a1", alpha=0.3)
    ax.set_xlim([-0.5,100.5])
    ax.set_ylim([-0.5,100.5])
    ax.set_aspect('equal')
    ax.set_xlabel('Recall [%]', labelpad=15, fontsize=12, color="#333533")
    ax.set_ylabel('Precision [%]', labelpad=15, fontsize=12, color="#333533")
    ax.set_title("Precision-Recall Curve for {}".format(title), fontsize=18, color="#333533")
    
    # actual plot 
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)
    plt.plot(100*recall, 100*precision, label=name, linewidth=2, **kwargs)


def __draw_stuff(imbalanced_model, weighted_model, resampled_model, title=""):
    train_predictions_baseline = imbalanced_model.predict(train_features, batch_size=BATCH_SIZE)
    test_predictions_baseline = imbalanced_model.predict(test_features, batch_size=BATCH_SIZE)
    train_predictions_weighted = weighted_model.predict(train_features, batch_size=BATCH_SIZE)
    test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)
    train_predictions_resampled = resampled_model.predict(train_features, batch_size=BATCH_SIZE)
    test_predictions_resampled = resampled_model.predict(test_features, batch_size=BATCH_SIZE)


    plot_pr("Train Imbalanced", train_labels, train_predictions_baseline, title, color=colors[0])
    plot_pr("Test Imbalanced", test_labels, test_predictions_baseline, title, color=colors[0], linestyle='--')

    plot_pr("Train Weighted", train_labels, train_predictions_weighted, title, color=colors[1])
    plot_pr("Test Weighted", test_labels, test_predictions_weighted, title, color=colors[1], linestyle='--')

    plot_pr("Train Resampled", train_labels, train_predictions_resampled,  title, color=colors[2])
    plot_pr("Test Resampled", test_labels, test_predictions_resampled,  title, color=colors[2], linestyle='--')
    plt.legend(loc='upper right')
    plt.show()


    print("Imbalanced model:")
    imbalanced_results = imbalanced_model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)
    for name, value in zip(imbalanced_model.metrics_names, imbalanced_results):
        print(name, ': ', value)
    print()

    print("Weighted model:")
    weighted_results = weighted_model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)
    for name, value in zip(weighted_model.metrics_names, weighted_results):
        print(name, ': ', value)
    print()

    print("Resampled model")
    resampled_results = resampled_model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)
    for name, value in zip(resampled_model.metrics_names, resampled_results):
        print(name, ': ', value)
    print()

