import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

DATASET_PATH = "dataset/"
BATCH_SIZE = 32
IMG_HEIGHT = 120
IMG_WIDTH = 120


def load_and_split_data():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds_initial = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # Podział 20% walidacji na 10% walidacji i 10% testów
    val_batches = tf.data.experimental.cardinality(val_ds_initial)
    test_ds = val_ds_initial.take(val_batches // 2)
    val_ds = val_ds_initial.skip(val_batches // 2)

    return train_ds, val_ds, test_ds


def main():
    train_ds, val_ds, test_ds = load_and_split_data()
    print("Dane załadowane")


if __name__ == "__main__":
    main()