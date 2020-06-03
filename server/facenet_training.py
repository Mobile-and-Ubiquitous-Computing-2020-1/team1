"""
main training modeul for facenet
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

from absl import app
from absl import flags

import tensorflow as tf

from models import InceptionResNetV1
from utils.log import fedex_logger as logging
from utils.data_pipeline import create_data_pipeline

FLAGS = flags.FLAGS

# flags definition
flags.DEFINE_integer('image_size', 160,
                     'default image size')
flags.DEFINE_integer('batch_size', 64,
                     'training batch size')
flags.DEFINE_integer('num_epochs', 30,
                     'number of training epochs')
flags.DEFINE_float('learning_rate', 0.1,
                   'train learning rate')
flags.DEFINE_integer('log_frequency', 50,
                     'logging frequency')
flags.DEFINE_integer('save_frequency', 200,
                     'saving model frequency')
flags.DEFINE_string('data_dir', '/cmsdata/ssd1/cmslab/vggface2/processed',
                    'root of dataset')

def main(args):
  # dataset preparation
  train_dataset, test_dataset, num_classes, num_images = \
    create_data_pipeline(FLAGS.data_dir, FLAGS.batch_size,
                         FLAGS.image_size)

  model = InceptionResNetV1(num_classes=num_classes)
  img_size = (None, FLAGS.image_size, FLAGS.image_size, 3)

  optimizer = tf.keras.optimizers.Adam(learning_rate=0.1,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=0.1)

  loss_metric = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
  center_loss_metric = tf.keras.metrics.Mean(name='center_loss',
                                             dtype=tf.float32)
  accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(
      name='accuracy', dtype=tf.float32)

  # train
  @tf.function(input_signature=(tf.TensorSpec(img_size, tf.float32),
                                tf.TensorSpec((None,), tf.int32)))
  def train_step(images, labels):
    with tf.GradientTape() as tape:
      logits, embeddings = model(images, training=True)
      loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, logits, True)
      loss = tf.reduce_mean(loss)
      center_loss = model.calculate_center_loss(embeddings, labels)
      loss = (center_loss * 0.007) + loss
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, center_loss, logits

  global_step = 0
  step_per_epoch = math.ceil(num_images / FLAGS.batch_size)
  for epoch in range(FLAGS.num_epochs):
    loss_metric.reset_states()
    center_loss_metric.reset_states()
    accuracy_metric.reset_states()
    for epoch_step, (images, labels) in enumerate(train_dataset):
      train_loss, train_center_loss, train_logits = train_step(images, labels)

      global_step += 1
      loss_metric.update_state(train_loss)
      center_loss_metric.update_state(train_center_loss)
      accuracy_metric.update_state(labels, train_logits)
      if global_step % FLAGS.log_frequency == 0:
        logging.debug('Step %d (%f %% of epoch %d): loss = %f, '
                      'center loss = %f, accuracy = %f',
                      global_step, (epoch_step / step_per_epoch * 100), epoch + 1,  # pylint: disable=line-too-long
                      loss_metric.result().numpy(),
                      center_loss_metric.result().numpy(),
                      accuracy_metric.result().numpy())
      if global_step % FLAGS.save_frequency == 0:
        model.save_weights('/tmp/ktaebum/detect_model_ckpt')

if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '1'
  app.run(main)
