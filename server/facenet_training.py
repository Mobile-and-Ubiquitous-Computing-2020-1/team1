"""
main training modeul for facenet
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getpass
import math
import os

from absl import app
from absl import flags

import numpy as np
import tensorflow as tf
import torch
import time

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
flags.DEFINE_integer('num_epochs', 300,
                     'number of training epochs')
flags.DEFINE_float('learning_rate', 0.05,
                   'train learning rate')
flags.DEFINE_integer('log_frequency', 50,
                     'logging frequency')
flags.DEFINE_integer('save_frequency', 200,
                     'saving model frequency')
flags.DEFINE_string('data_dir', '/cmsdata/ssd1/cmslab/vggface2/final',
                    'root of dataset')
flags.DEFINE_string('model_dir', f'/tmp/{getpass.getuser()}/checkpoints',
                    'model checkpoint path')
flags.DEFINE_bool('eval', False,
                  'eval mode')
flags.DEFINE_bool('load_pretrained', False,
                  'load pretrained weights')
flags.DEFINE_bool('save_tflite', False,
                  'directly save tflite model')
flags.DEFINE_bool('use_center_loss', False,
                  'toggle center loss')

def load_pretrained(model):
  """load pretrained weight from pretrained pytorch model"""
  # pylint: disable=line-too-long
  pretrained_features = './checkpoints/inception_resnet_v1/vggface2_feature_map.pt'
  pretrained_classifier = './checkpoints/inception_resnet_v1/vggface2_classifier.pt'
  # pylint: enable=line-too-long

  pretrained_weights = []
  pretrained_weights.extend(list(torch.load(pretrained_features).values()))
  pretrained_weights.extend(list(torch.load(pretrained_classifier).values()))

  # num_batch_tracked
  pretrained_weights = list(filter(lambda x: tuple(x.shape) != (),
                                   pretrained_weights))

  weight_iter = iter(pretrained_weights)
  for layer in model.layers:
    if isinstance(layer, CenterLoss):
      # skip; PyTorch module does not have
      continue

    num_weights = len(layer.get_weights())
    pth_weights = []
    for _ in range(num_weights):
      weight = next(weight_iter).numpy()
      if len(weight.shape) == 4:
        # conv kernel
        weight = np.transpose(weight, (2, 3, 1, 0))
      elif len(weight.shape) == 2:
        # dense kernel
        weight = np.transpose(weight)
      pth_weights.append(weight)
    layer.set_weights(pth_weights)

def main(args):
  # dataset preparation
  train_dataset, test_dataset, num_classes, num_train, num_test = \
    create_data_pipeline(FLAGS.data_dir, FLAGS.batch_size,
                         FLAGS.image_size)

  model = InceptionResNetV1(num_classes=num_classes,
                            use_center_loss=FLAGS.use_center_loss)
  img_size = (None, FLAGS.image_size, FLAGS.image_size, 3)
  model.build(img_size)

  if FLAGS.load_pretrained:
    logging.info('load pretrained model...')
    # for center loss variable
    try:
      model.load_weights(os.path.join(FLAGS.model_dir, 'facenet_ckpt'))
    except ValueError:
      logging.debug('pretrained checkpoint does not exists, '
                    'failed to restore center loss variable')
    load_pretrained(model)
    logging.info('loading pretrained model finished!')

  if FLAGS.save_tflite:
    # pylint: disable=protected-access
    model._set_inputs(tf.keras.layers.Input(shape=(160, 160, 3), batch_size=1))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with tf.io.gfile.GFile('./tflite-models/facenet.tflite', 'wb') as f:
      f.write(tflite_model)
    return

  loss_metric = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
  center_loss_metric = tf.keras.metrics.Mean(name='center_loss',
                                             dtype=tf.float32)
  accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(
      name='accuracy', dtype=tf.float32)

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

  if FLAGS.eval:
    eval()
    return

  # train
  step_per_epoch = math.ceil(num_train / FLAGS.batch_size)

  lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      [100 * step_per_epoch, 200 * step_per_epoch],
      [FLAGS.learning_rate, FLAGS.learning_rate * 0.1, FLAGS.learning_rate * 0.01]
  )

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=0.1)

  @tf.function(input_signature=(tf.TensorSpec(img_size, tf.float32),
                                tf.TensorSpec((None,), tf.int32)))
  def train_step(images, labels):
    with tf.GradientTape() as tape:
      logits, prelogits = model(images, training=True)
      loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, logits, False)
      loss = tf.reduce_mean(loss)

      if FLAGS.use_center_loss:
        # recomputation embedding (for export convenience)
        embeddings = model.calculate_embedding(prelogits)
        center_loss = model.calculate_center_loss(embeddings, labels)
      else:
        center_loss = tf.constant(0.0, dtype=tf.float32)
      loss = (center_loss * 0.007) + loss
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, center_loss, logits

  model_dir = FLAGS.model_dir
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  global_step = 0
  for epoch in range(FLAGS.num_epochs):
    loss_metric.reset_states()
    center_loss_metric.reset_states()
    accuracy_metric.reset_states()
    num_images = 0
    start = time.time()
    for epoch_step, (images, labels) in enumerate(train_dataset):
      train_loss, train_center_loss, train_logits = train_step(images, labels)

      loss_metric.update_state(train_loss)
      center_loss_metric.update_state(train_center_loss)
      accuracy_metric.update_state(labels, train_logits)
      global_step += 1
      num_images += images.shape[0]

      if global_step % FLAGS.log_frequency == 0:
        end = time.time()
        throughput = num_images / (end - start)
        logging.debug('Step %d (%f %% of epoch %d): loss = %f, '
                      'center loss = %f, accuracy = %f, learning rate = %.4f '
                      'throughput = %.2f',
                      global_step, (epoch_step / step_per_epoch * 100),
                      epoch + 1,
                      loss_metric.result().numpy(),
                      center_loss_metric.result().numpy(),
                      accuracy_metric.result().numpy() * 100,
                      optimizer._decayed_lr(tf.float32),  # pylint: disable=protected-access
                      throughput)

      if FLAGS.save_frequency > 0 and global_step % FLAGS.save_frequency == 0:
        model.save_weights(os.path.join(model_dir, 'facenet_ckpt'))

      if FLAGS.save_frequency > 0 and global_step % 1000 == 0:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with tf.io.gfile.GFile('./tflite-models/facenet.tflite', 'wb') as f:
          f.write(tflite_model)

      if global_step % FLAGS.log_frequency == 0:
        num_images = 0
        start = time.time()

  # eval and finish
  eval()

if __name__ == "__main__":
  app.run(main)
