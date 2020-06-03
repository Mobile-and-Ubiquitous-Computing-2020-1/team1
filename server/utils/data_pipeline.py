"""
Data pipeline module for VGGFace2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os

import tensorflow as tf
from tensorflow.python.data import Dataset

from utils.log import fedex_logger as logging

def create_data_pipeline(data_dir,
                         batch_size=32,
                         img_size=182,
                         target='train'):
  """data pipeline for vggface2"""
  train_ratio = 0.8  # portion of train dataset
  classes = glob.glob(os.path.join(data_dir, target + '/*'))

  logging.debug('creating data pipeline, num classes %d', len(classes))

  class2labels = {c.split('/')[-1]: i for i, c in enumerate(classes)}
  num_classes = len(classes)
  num_images = 0

  train_datas = []
  train_labels = []

  test_datas = []
  test_labels = []

  for folder in classes:
    class_name = folder.split('/')[-1]
    all_images = glob.glob(os.path.join(folder, '*.jpg'))
    num_images += len(all_images)

    split_idx = int(len(all_images) * train_ratio)

    train_data = all_images[:split_idx]
    test_data = all_images[split_idx:]
    train_datas.extend(train_data)
    train_labels.extend([class2labels[class_name]] * len(train_data))
    test_datas.extend(test_data)
    test_labels.extend([class2labels[class_name]] * len(test_data))

  def process_dataset(dataset, is_training=False):
    def make_img_tensor(filename):
      file_contents = tf.io.read_file(filename)
      image = tf.image.decode_image(file_contents, 3)
      image = tf.image.random_flip_left_right(image)
      image = tf.image.resize_with_crop_or_pad(image, img_size, img_size)
      image = (tf.cast(image, tf.float32) - 127.5) / 128.0
      return image

    dataset = dataset.map(
        lambda filename, label: (make_img_tensor(filename), label))

    if is_training:
      dataset = dataset.shuffle(num_classes)

    dataset = dataset.batch(batch_size)
    return dataset

  train_dataset = Dataset.from_tensor_slices((train_datas, train_labels))
  test_dataset = Dataset.from_tensor_slices((test_datas, test_labels))

  train_dataset = process_dataset(train_dataset, True)
  test_dataset = process_dataset(test_dataset, False)
  return train_dataset, test_dataset, num_classes, num_images
