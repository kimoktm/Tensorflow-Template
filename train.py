# ============================================================== #
#                          Model train                           #
#                                                                #
#                                                                #
# Train model with processed dataset                             #
# ============================================================== #

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# disable GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

import data.dataset_loader as dataset_loader
import model.network as model
from config import config

tf.logging.set_verbosity(tf.logging.INFO)

class ValidationHook(tf.train.SessionRunHook):
  def __init__(self, model_fn, params, input_fn, checkpoint_dir,
         every_n_secs=None, every_n_steps=None):
    self._iter_count = 0
    self._estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      params=params,
      model_dir=checkpoint_dir
    )
    self._input_fn = input_fn
    self._timer = tf.train.SecondOrStepTimer(every_n_secs, every_n_steps)
    self._should_trigger = False

  def begin(self):
    self._timer.reset()
    self._iter_count = 0

  def before_run(self, run_context):
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)

  def after_run(self, run_context, run_values):
    if self._should_trigger:
      self._estimator.evaluate(
        self._input_fn
      )
      self._timer.update_last_triggered_step(self._iter_count)
    self._iter_count += 1


def main(_):
  train_path = "../face3d/examples/results/faces/train"
  vald_path = "../face3d/examples/results/faces/validation"

  my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 2*60,  # Save checkpoints every 1 minutes.
    # save_checkpoints_steps= 100,
    keep_checkpoint_max = 2,       # Retain the 10 most recent checkpoints.
  )

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
    model_fn=model.model_fn,
    params={"learning_rate": config['learning_rate']},
    model_dir=config['model_dir'],
    config=my_checkpointing_config)

  train_input_fn = lambda: dataset_loader.input_fn(
    dataset=dataset_loader.training_generator(train_path),
    batch_size=config['batch_size'],
    num_epochs=config['training_epochs'],
    shuffle=True)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  # tensors_to_log = {"rotation_loss"}
  # logging_hook = tf.train.LoggingTensorHook(
  #   tensors=tensors_to_log, every_n_iter=10)

  # mnist_classifier.train(input_fn=train_input_fn,
  #   steps=config['steps'])

  eval_input_fn = lambda: dataset_loader.input_fn(
    dataset=dataset_loader.test_generator(vald_path),
    batch_size=config['batch_size'],
    num_epochs=1,
    shuffle=False)

  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=config['steps'])
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, 
    start_delay_secs= 1,
    throttle_secs= 2*60)
  tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)


if __name__ == "__main__":
  tf.app.run()