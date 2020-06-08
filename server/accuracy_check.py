"""
additional training module for facenet
(kind of transfer learning)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getpass
import glob
import math
import os

from absl import app
from absl import flags

import numpy as np
import torch
import time
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from models import InceptionResNetV1
from models import CenterLoss
from utils.log import fedex_logger as logging
from utils.data_pipeline import create_data_pipeline

FLAGS = flags.FLAGS

# flags definition
flags.DEFINE_integer('image_size', 160,
                     'default image size')
flags.DEFINE_integer('batch_size', 90,
                     'training batch size')
flags.DEFINE_integer('num_epochs', 10,
                     'number of training epochs')
flags.DEFINE_integer('num_classes', 8631,
                     'number of new classes')
flags.DEFINE_float('learning_rate', 0.05,
                   'train learning rate')
flags.DEFINE_integer('log_frequency', 50,
                     'logging frequency')
flags.DEFINE_string('data_dir', '/cmsdata/ssd1/cmslab/vggface2/final',
                    'root of dataset')
flags.DEFINE_string('checkpoint_path', f'/tmp/{getpass.getuser()}/checkpoints',
                    'model checkpoint path')

def prepare_intermediates_data():
  data_path = '/cmsdata/ssd1/cmslab/vggface2/final/intermediates'
  all_files = glob.glob(os.path.join(data_path, '*.tfrecord'))
  description = {
      'value': tf.io.FixedLenFeature([], tf.string),
      'label': tf.io.FixedLenFeature([], tf.string),
  }

  def map_fn(x):
    datas = tf.io.parse_single_example(x, description)
    features = datas['value']
    labels = datas['label']
    features = tf.io.parse_tensor(features, tf.float32)
    labels = tf.io.parse_tensor(labels, tf.int32)
    return features, labels

  dataset = tf.data.TFRecordDataset(all_files)
  dataset = dataset.map(map_fn)
  return dataset

def synthetic_dataset():
  while True:
    yield (tf.random.normal(shape=(FLAGS.batch_size, 512)),
           tf.random.uniform(shape=(FLAGS.batch_size,),
                             minval=0, maxval=FLAGS.num_classes,
                             dtype=tf.int32))

def main(args):
  _, test_dataset, _, num_classes, _, num_test, num_train = \
    create_data_pipeline(FLAGS.data_dir, FLAGS.batch_size,
                         FLAGS.image_size, additional_portion=0.7)

  img_size = (None, FLAGS.image_size, FLAGS.image_size, 3)
  model = InceptionResNetV1(num_classes=num_classes,
                            use_center_loss=False)
  model.build(img_size)

  logging.info('load pretrained model...')
  # for center loss variable
  model.load_weights(os.path.join(FLAGS.checkpoint_path, 'facenet_ckpt'))
  logging.info('loading pretrained model finished!')

  loss_metric = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
  accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(
      name='accuracy', dtype=tf.float32)

  # train
  num_train = num_train * 0.64
  step_per_epoch = math.ceil(num_train / FLAGS.batch_size)

  optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=0.1)
  
  train_dataset = prepare_intermediates_data()

  @tf.function(input_signature=(tf.TensorSpec((None, 512), tf.float32),
                                tf.TensorSpec((None,), tf.int32)))
  def train_step(prelogits, labels):
    with tf.GradientTape() as tape:
      logits, _ = model(prelogits, training=True)
      loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, logits, False)
      loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, model.classifier.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.classifier.trainable_variables))
    return loss, logits

  total_start = time.time()
  global_step = 0
  for epoch in range(FLAGS.num_epochs):
    loss_metric.reset_states()
    accuracy_metric.reset_states()
    num_images = 0
    start = time.time()
    for epoch_step, (images, labels) in enumerate(train_dataset):
      train_loss, train_logits = train_step(images, labels)

      loss_metric.update_state(train_loss)
      accuracy_metric.update_state(labels, train_logits)
      global_step += 1
      num_images += images.shape[0]

      percentage = epoch_step / step_per_epoch * 100
      if percentage >= 100:
        break

      if global_step % FLAGS.log_frequency == 0:
        end = time.time()
        throughput = num_images / (end - start)
        logging.debug('Step %d (%f %% of epoch %d): loss = %f, '
                      'accuracy = %f, learning rate = %.4f '
                      'throughput = %.2f',
                      global_step, (epoch_step / step_per_epoch * 100),
                      epoch + 1,
                      loss_metric.result().numpy(),
                      accuracy_metric.result().numpy() * 100,
                      optimizer._decayed_lr(tf.float32),  # pylint: disable=protected-access
                      throughput)

      if global_step % FLAGS.log_frequency == 0:
        num_images = 0
        start = time.time()
  total_end = time.time()

  @tf.function(input_signature=(tf.TensorSpec(img_size, tf.float32),
                                tf.TensorSpec((None,), tf.int32)))
  def eval_step(images, labels):
    logits, _ = model(images, training=False)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, False)
    loss = tf.reduce_mean(loss)
    return loss, logits

  def eval():
    step_per_epoch = math.ceil(num_test / FLAGS.batch_size)
    for i, (images, labels) in enumerate(test_dataset):
      loss, logits = eval_step(images, labels)
      loss_metric.update_state(loss)
      accuracy_metric.update_state(labels, logits)
      print('Eval %f%%, loss = %f, accuracy = %f' % \
            (i / step_per_epoch * 100,
             loss_metric.result().numpy(),
             accuracy_metric.result().numpy() * 100))
  eval()
  print('total time: %f seconds' % (total_end - total_start))

if __name__ == "__main__":
  app.run(main)
