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
  path = "/home/karim/Documents/Development/FacialCapture/face3d/examples/results/faces/training"

  my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 5*60,  # Save checkpoints every 1 minutes.
    keep_checkpoint_max = 5,       # Retain the 10 most recent checkpoints.
  )

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
    model_fn=model.model_fn,
    params={"learning_rate": config['learning_rate']},
    model_dir=config['model_dir'],
    config=my_checkpointing_config)

  train_input_fn = lambda: dataset_loader.input_fn(
    dataset=dataset_loader.training_generator(path),
    batch_size=config['batch_size'],
    num_epochs=config['training_epochs'],
    shuffle=True)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  # tensors_to_log = {"rmse"}
  # logging_hook = tf.train.LoggingTensorHook(
  #   tensors=tensors_to_log, every_n_iter=10)

  mnist_classifier.train(input_fn=train_input_fn,
    steps=config['steps'])

if __name__ == "__main__":
  tf.app.run()