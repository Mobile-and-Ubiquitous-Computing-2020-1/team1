"""
just throughput test
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getpass
import math
import os
import time

from absl import app
from absl import flags

import numpy as np

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from models import InceptionResNetV1
from models import ThawedModel1
from models import ThawedModel2
from models import ThawedModel3
from models import ThawedModel4
from models import ThawedModel5
from models import ThawedModel6

FLAGS = flags.FLAGS

model_config = [
    InceptionResNetV1,
    ThawedModel1,
    ThawedModel2,
    ThawedModel3,
    ThawedModel4,
    ThawedModel5,
    ThawedModel6,
]

input_sizes = [
    (160, 160, 3),
    (38, 38, 64),
    (17, 17, 256),
    (17, 17, 256),
    (8, 8, 896),
    (1792,),
    (512,),
]

flags.DEFINE_integer('batch_size', 90,
                     'training batch size')
flags.DEFINE_integer('test_model', 1,
                     'testing thawed model')

num_classes = 8631
def synthetic_dataset(input_shape):
  while True:
    yield (tf.random.normal(shape=(FLAGS.batch_size, *input_shape)),
           tf.random.uniform(shape=(FLAGS.batch_size,),
                             minval=0, maxval=num_classes,
                             dtype=tf.int32))

def num_params(model):
  shape = 0
  tvar = model.trainable_variables
  for var in tvar:
    shape += np.prod(var.shape)
  print(FLAGS.test_model, shape)
  return shape

def main(args):
  model_idx = FLAGS.test_model

  model = model_config[model_idx]()
  input_shape = input_sizes[model_idx]

  batched_shape = (None, *input_shape)
  model.build(batched_shape)

  num_params(model)
  return

  optimizer = tf.keras.optimizers.Adam(learning_rate=0.005,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=0.1)

  @tf.function(input_signature=(tf.TensorSpec((None, *input_shape), tf.float32),
                                tf.TensorSpec((None,), tf.int32)))
  def train_step(images, labels):
    with tf.GradientTape() as tape:
      logits, _ = model(images, training=True)
      loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, logits, False)
      loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, logits

  train_dataset = iter(synthetic_dataset(input_shape))
  num_images = 0
  step = 0
  start = time.time()
  for images, labels in train_dataset:
    train_loss, train_logits = train_step(images, labels)

    num_images += images.shape[0]
    step += 1

    if step % 100 == 0:
      end = time.time()
      throughput = num_images / (end - start)
      print('Step %d: throughput = %.2f' % (step, throughput))

      num_images = 0
      start = time.time()

if __name__ == "__main__":
  app.run(main)
