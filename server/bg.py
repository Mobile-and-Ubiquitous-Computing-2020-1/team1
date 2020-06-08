"""
additional training module for facenet
(kind of transfer learning)
"""

from __future__ import absolute_import, division, print_function

import getpass
import math
import os
import requests

import numpy as np
from absl import app, flags

import tensorflow as tf
from models import InceptionResNetV1
from tensorflow.python.data import Dataset
from utils.log import fedex_logger as logging
from utils.data_pipeline import create_data_pipeline

import const as C

FLAGS = flags.FLAGS

# flags definition
flags.DEFINE_integer('image_size', 160,
                     'default image size')
flags.DEFINE_integer('batch_size', 90,
                     'training batch size')
flags.DEFINE_integer('num_epochs', 15,
                     'number of training epochs')
flags.DEFINE_integer('num_classes', 8631,
                     'number of new classes')
flags.DEFINE_float('learning_rate', 0.05,
                   'train learning rate')
flags.DEFINE_integer('log_frequency', 50,
                     'logging frequency')
flags.DEFINE_integer('save_frequency', 200,
                     'saving model frequency')
flags.DEFINE_string('data_dir', '/cmsdata/ssd1/cmslab/vggface2/final',
                    'root of dataset')
flags.DEFINE_string('checkpoint_path', f'/tmp/{getpass.getuser()}/checkpoints',
                    'model checkpoint path')
flags.DEFINE_string('model_dir', f'/tmp/{getpass.getuser()}/checkpoints/additional',
                    'new model checkpoint path')
flags.DEFINE_bool('save_tflite', False,
                  'directly save tflite model')

def synthetic_dataset():
  target_dtype = tf.float32
  shape_per_tensor = (1, 512)  # test for mobilenet_v1
  tensor_size = np.prod(shape_per_tensor)

  path = os.path.join(C.FEATURE_PATH, "intermediates")
  with open(path, 'rb') as f:
    raw_bytes = f.read()
  bytes_size = tensor_size * target_dtype.size
  tensors = []
  for i in range(0, len(raw_bytes), bytes_size):
    raw_byte = raw_bytes[i:i+bytes_size]
    tensor = tf.io.decode_raw(raw_byte, target_dtype)
    tensor = tf.reshape(tensor, shape_per_tensor)
    tensors.append(tensor)
  tensor = tf.stack(tensors, axis=0)
  tensor = tf.reshape(tensor, (len(tensors), 512))

  additional_data = (tensor,
           tf.random.uniform(shape=(len(tensors),),
                             minval=0, maxval=FLAGS.num_classes,
                             dtype=tf.int32))
  dataset = Dataset.from_tensor_slices(additional_data)
  dataset = dataset.shuffle(1000)
  dataset = dataset.batch(FLAGS.batch_size)
  return dataset

def main(args):
  _, test_dataset, _, _, _, num_test, _ = \
    create_data_pipeline(FLAGS.data_dir, FLAGS.batch_size,
                         FLAGS.image_size)

  num_classes = FLAGS.num_classes
  img_size = (None, FLAGS.image_size, FLAGS.image_size, 3)
  model = InceptionResNetV1(num_classes=num_classes,
                            use_center_loss=False)
  model.build(img_size)

  logging.info('load pretrained model...')
  # for center loss variable
  model.load_weights(os.path.join(FLAGS.checkpoint_path, 'facenet_ckpt'))
  logging.info('loading pretrained model finished!')

  classifier = model.classifier

  loss_metric = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
  accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(
      name='accuracy', dtype=tf.float32)

  # train
  num_train = 300000  # temp
  step_per_epoch = math.ceil(num_train / FLAGS.batch_size)

  lr_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      [10 * step_per_epoch, 50 * step_per_epoch],
      [FLAGS.learning_rate, FLAGS.learning_rate * 0.1, FLAGS.learning_rate * 0.01]
  )

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler,
                                       beta_1=0.9,
                                       beta_2=0.999,
                                       epsilon=0.1)

  @tf.function(input_signature=(tf.TensorSpec((None, 512), tf.float32),
                                tf.TensorSpec((None,), tf.int32)))
  def train_step(images, labels):
    with tf.GradientTape() as tape:
      logits, _ = model(images, training=True)
      loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, logits, False)
      loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, classifier.trainable_variables)
    optimizer.apply_gradients(zip(grads, classifier.trainable_variables))
    return loss, logits

  @tf.function(input_signature=(tf.TensorSpec(img_size, tf.float32),
                                tf.TensorSpec((None,), tf.int32)))
  def eval_step(images, labels):
    logits, _ = model(images, training=False)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, False)
    loss = tf.reduce_mean(loss)
    return loss, logits

  def eval():
    accuracy_metric.reset_states()
    step_per_epoch = math.ceil(num_test / FLAGS.batch_size)
    for i, (images, labels) in enumerate(test_dataset):
      loss, logits = eval_step(images, labels)
      loss_metric.update_state(loss)
      accuracy_metric.update_state(labels, logits)
    return accuracy_metric.result().numpy() * 100

  model_dir = FLAGS.model_dir
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  global_step = 0
  train_dataset = iter(synthetic_dataset())
  for epoch in range(FLAGS.num_epochs):
    print("epoch", epoch)
    loss_metric.reset_states()
    accuracy_metric.reset_states()
    for epoch_step, (images, labels) in enumerate(train_dataset):
      train_loss, train_logits = train_step(images, labels)

      loss_metric.update_state(train_loss)
      accuracy_metric.update_state(labels, train_logits)
      global_step += 1

      if FLAGS.save_frequency > 0 and global_step % 1000 == 0:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        model_path = os.getenv("MODEL_PATH")
        output_path = os.path.join(C.MODEL_PATH, model_path)
        with tf.io.gfile.GFile(output_path, 'wb') as f:
          f.write(tflite_model)
  # eval and finish
  acc = eval()
  print(acc)

  trial_id = int(os.getenv("TRIAL_ID", 0))

  requests.post("http://127.0.0.1:8000/model/train", json=dict(acc=acc, trial_id=trial_id))




if __name__ == "__main__":
  app.run(main)
