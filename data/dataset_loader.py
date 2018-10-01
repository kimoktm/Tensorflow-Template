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
import os

def _parse_function(imgfile, params):
    """
    Reads an image from a file, decodes it into a dense tensor
    ----------

    Returns:
        dataset: tf.dataset, dataset holding training data and corresponding labels
    """
    image_string = tf.read_file(imgfile)
    image_decoded = tf.image.decode_jpeg(image_string)
    image = tf.image.per_image_standardization(image_decoded)

    return image, params


def training_generator(path):
    """
    Generate dataset for training:
    ----------

    Returns:
        dataset: tf.dataset, dataset holding training data and corresponding labels
    """

    imgfiles = tf.gfile.Glob(os.path.join(path, '*.jpg'))
    paramsfiles = tf.gfile.Glob(os.path.join(path, 'params_*.npy'))
    params = np.array([np.load(p) for p in paramsfiles])
    params = np.float32(params)

    dataset = tf.data.Dataset.from_tensor_slices((imgfiles, params))
    dataset = dataset.map(_parse_function)

    return dataset


def test_generator(path):
    """
    Generate dataset for testing:
    ----------

    Returns:
        dataset: tf.dataset, dataset holding testing data and corresponding labels
    """

    return training_generator(path)


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