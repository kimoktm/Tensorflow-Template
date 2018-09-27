# ============================================================== #
#                          Model eval                            #
#                                                                #
#                                                                #
# Eval model with processed dataset                              #
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
      model_fn=model.model_fn, model_dir=config['model_dir'])

  eval_input_fn = lambda: dataset_loader.input_fn(
    dataset=dataset_loader.test_generator(),
    batch_size=config['batch_size'],
    num_epochs=1,
    shuffle=False)

  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()