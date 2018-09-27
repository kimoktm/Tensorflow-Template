# ============================================================== #
#                          Model train                           #
#                                                                #
#                                                                #
# Train model with processed dataset                             #
# ============================================================== #

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import data.dataset_loader as dataset_loader
import model.network as model
from config import config

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=model.model_fn,
      params={"learning_rate": config['learning_rate']},
      model_dir=config['model_dir'])

  train_input_fn = lambda: dataset_loader.input_fn(
    dataset=dataset_loader.training_generator(),
    batch_size=config['batch_size'],
    num_epochs=config['training_epochs'],
    shuffle=True)

  # # Set up logging for predictions
  # # Log the values in the "Softmax" tensor with label "probabilities"
  # tensors_to_log = {"probabilities": "softmax_tensor"}
  # logging_hook = tf.train.LoggingTensorHook(
  #     tensors=tensors_to_log, every_n_iter=10000)
  mnist_classifier.train(input_fn=train_input_fn, steps=2000)

if __name__ == "__main__":
  tf.app.run()