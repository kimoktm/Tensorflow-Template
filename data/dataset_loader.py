# ============================================================== #
#                        Dataset Loader                          #
#                                                                #
#                                                                #
# Processing occurs on a single image at a time. A               #
# Batch is then formed from these data to be used for training   #
# or evaluation                                                  #
# ============================================================== #

import numpy as np
import tensorflow as tf


def training_generator():
    """
    Generate dataset for training:
    ----------

    Returns:
        dataset: tf.dataset, dataset holding training data and corresponding labels
    """

    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
    train_data, train_labels = mnist_train
    train_data = tf.cast(train_data, tf.float32)
    train_labels = tf.cast(train_labels, tf.int32)
    mnist_train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_labels))

    return mnist_train_ds


def test_generator():
    """
    Generate dataset for testing:
    ----------

    Returns:
        dataset: tf.dataset, dataset holding testing data and corresponding labels
    """

    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
    test_data, test_labels = mnist_test
    test_data = tf.cast(test_data, tf.float32)
    test_labels = tf.cast(test_labels, tf.int32)
    mnist_test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_labels))

    return mnist_test_ds


def input_fn(dataset, batch_size=None, num_epochs=None, shuffle=True):
    """
    Input function handler for tf.estimator:
    ----------

    Returns:
        features, labels: shuffled batches iterator
    """

    if shuffle:
            dataset = dataset.shuffle(buffer_size=int(5.4 * batch_size))

    # dataset = dataset.prefetch(config["max_buffer"])
    dataset = dataset.repeat(num_epochs)

    if batch_size:
        dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return {"x": batch_features}, batch_labels