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

TARGET_DIR = '/cmsdata/ssd1/cmslab/vggface2/final/intermediates'

def serialize(images, labels):
  def serialize_each(x):
    x = tf.io.serialize_tensor(x)
    return x.numpy()

  x = tf.train.BytesList(value=[serialize_each(images)])
  y = tf.train.BytesList(value=[serialize_each(labels)])
  x = tf.train.Feature(bytes_list=x)
  y = tf.train.Feature(bytes_list=y)
  feature = {
    'value': x, 'label': y
  }

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()

def main(args):
  # dataset preparation
  _, test_dataset, train_dataset, num_classes, _, num_test, num_train = \
    create_data_pipeline(FLAGS.data_dir, FLAGS.batch_size,
                         FLAGS.image_size, additional_portion=0.7)

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

  @tf.function(input_signature=(tf.TensorSpec(img_size, tf.float32),))
  def extract_features(images):
    return model.feature_extract(images)

  for i, (images, labels) in enumerate(train_dataset):
    writer = tf.io.TFRecordWriter(
        os.path.join(TARGET_DIR, 'data_%05d.tfrecord' % i))
    features = extract_features(images)
    example = serialize(features, labels)
    writer.write(example)
    writer.close()
    print(i, 'finished')

if __name__ == "__main__":
  app.run(main)
