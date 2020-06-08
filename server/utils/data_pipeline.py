"""
Data pipeline module for VGGFace2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import random

import tensorflow as tf
from tensorflow.python.data import Dataset

from utils.log import fedex_logger as logging

def create_data_pipeline(data_dir,
                         batch_size=32,
                         img_size=182,
                         target='train',
                         additional_portion=0.5):
  """data pipeline for vggface2"""
  train_ratio = 0.8  # portion of train dataset
  classes = glob.glob(os.path.join(data_dir, target + '/*'))

  logging.debug('creating data pipeline, num classes %d', len(classes))

  shorten_classes = list(map(lambda x: x.split('/')[-1], classes))
  class2labels = {c: i for i, c in enumerate(
      sorted(shorten_classes, key=lambda x: int(x[1:])))}
  num_classes = len(classes)
  num_train = 0
  num_test = 0
  num_additional = 0

  train_datas = []
  train_labels = []

  test_datas = []
  test_labels = []

  additional_datas = []
  additional_labels = []

  for folder in classes:
    class_name = folder.split('/')[-1]
    all_images = glob.glob(os.path.join(folder, '*.jpg'))

    split_idx = int(len(all_images) * train_ratio)

    train_data = all_images[:split_idx]
    test_data = all_images[split_idx:]

    split_idx = int(len(test_data) * additional_portion)

    additional_data = test_data[:split_idx]
    test_data = test_data[split_idx:]

    num_train += len(train_data)
    num_test += len(test_data)
    num_additional += len(additional_data)
    train_datas.extend(train_data)
    train_labels.extend([class2labels[class_name]] * len(train_data))
    test_datas.extend(test_data)
    test_labels.extend([class2labels[class_name]] * len(test_data))
    additional_datas.extend(additional_data)
    additional_labels.extend([class2labels[class_name]] * len(additional_data))

  def process_dataset(dataset, is_training=False):
    def make_img_tensor(filename):
      file_contents = tf.io.read_file(filename)
      image = tf.image.decode_image(file_contents, 3)
      if is_training:
        image = tf.image.random_flip_left_right(image)
      image = tf.image.resize_with_crop_or_pad(image, img_size, img_size)
      image = (tf.cast(image, tf.float32) - 127.5) / 128.0
      return image

    dataset = dataset.map(
        lambda filename, label: (make_img_tensor(filename), label))

    dataset = dataset.batch(batch_size)
    return dataset

  # pre-shuffle
  def pre_shuffle(images, labels):
    dataset = list(map(lambda x: (x[0], x[1]), zip(images, labels)))
    random.shuffle(dataset)
    images = []
    labels = []
    for i, l in dataset:
      images.append(i)
      labels.append(l)
    return images, labels

  train_datas, train_labels = pre_shuffle(train_datas, train_labels)
  additional_datas, additional_labels = pre_shuffle(additional_datas,
                                                    additional_labels)

  logging.debug('num train data: %d, num test data: %d, num additional data %d',
                len(train_datas), len(test_datas), len(additional_datas))

  train_dataset = Dataset.from_tensor_slices((train_datas, train_labels))
  test_dataset = Dataset.from_tensor_slices((test_datas, test_labels))
  additional_dataset = Dataset.from_tensor_slices((additional_datas,
                                                   additional_labels))

  train_dataset = process_dataset(train_dataset, True)
  test_dataset = process_dataset(test_dataset, False)
  additional_dataset = process_dataset(additional_dataset, True)
  return train_dataset, test_dataset, additional_dataset, \
    num_classes, num_train, num_test, num_additional
