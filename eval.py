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
import numpy as np

import data.dataset_loader as dataset_loader
import model.network as model
from config import config

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  path = "../face3d/examples/results/faces/test"
  output = "../face3d/examples/results/faces/output"

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=model.model_fn,
      params={"learning_rate": config['learning_rate']},
      model_dir=config['model_dir'])

  eval_input_fn = lambda: dataset_loader.input_fn(
    dataset=dataset_loader.test_generator(path),
    batch_size=config['batch_size'],
    num_epochs=1,
    shuffle=False)

  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
  predict_results = mnist_classifier.predict(input_fn=eval_input_fn)


  import glob, os
  imgfiles = glob.glob(os.path.join(path, '*.jpg'))
  paramsfiles = [s.replace('.jpg', '.npy') for s in imgfiles]
  paramsfiles = [s.replace('generated', 'predicted') for s in paramsfiles]
  paramsfiles = [s.replace('test', 'output') for s in paramsfiles]


  for i, result in enumerate(predict_results):
    result = np.float32(result)
    np.save(paramsfiles[i], result)


if __name__ == "__main__":
  tf.app.run()