"""
Model code for FaceNet

updated version (compatible with TF 2.x) of
https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from tensorflow.python import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

class CenterLoss(layers.Layer):
  """center loss calculation (this is stateful)"""
  def __init__(self, num_classes, embed_dim=512):
    super(CenterLoss, self).__init__()
    self.num_classes = num_classes
    self.embed_dim = embed_dim

  def build(self, input_shape):
    self.center_var = self.add_weight('center_var',
                                      shape=(self.num_classes, self.embed_dim),
                                      dtype=tf.float32,
                                      initializer=initializers.zeros,
                                      trainable=False)
    self.built = True

  def call(self, features, labels):
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(self.center_var, labels)
    diff = (1 - 0.95) * (centers_batch - features)
    with tf.control_dependencies(
        [self.center_var.scatter_nd_sub(labels, diff)]):
      loss = tf.reduce_mean(tf.square(features - centers_batch))
      return loss

class BaseConvBlock(keras.Model):
  """Base Convolution Module"""
  def __init__(self,
               output_channels,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               weight_decay=5e-4,
               kernel_initializer=initializers.glorot_uniform,
               batch_norm_decay=0.995,
               batch_norm_epsilon=0.001,
               name=None):
    super(BaseConvBlock, self).__init__()

    self.conv = layers.Conv2D(filters=output_channels,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding=padding,
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=regularizers.l2(weight_decay) \
                              if weight_decay > 0 else None,
                              use_bias=False,
                              name=name)
    self.norm = layers.BatchNormalization(axis=-1,
                                          momentum=batch_norm_decay,
                                          epsilon=batch_norm_epsilon,
                                          name=name)
    self.activation = layers.ReLU()

  def call(self, x, training=False):
    x = self.conv(x)
    x = self.norm(x, training=training)
    x = self.activation(x)
    return x

class Block35(keras.Model):
  """Block35 module"""
  def __init__(self, filters, scale=1.0, activation_fn=tf.nn.relu):
    super(Block35, self).__init__()
    # branch 0
    self.tower_conv = BaseConvBlock(32, (1, 1),
                                    padding='same',
                                    name='Conv2d_1x1')

    # branch 1
    self.tower_conv1_0 = BaseConvBlock(32, (1, 1),
                                       padding='same',
                                       name='Conv2d_0a_1x1')
    self.tower_conv1_1 = BaseConvBlock(32, (3, 3),
                                       padding='same',
                                       name='Conv2d_0b_3x3')

    # branch 2
    self.tower_conv2_0 = BaseConvBlock(32, (1, 1),
                                       padding='same',
                                       name='Conv2d_0a_1x1')
    self.tower_conv2_1 = BaseConvBlock(32, (3, 3),
                                       padding='same',
                                       name='Conv2d_0b_3x3')
    self.tower_conv2_2 = BaseConvBlock(32, (3, 3),
                                       padding='same',
                                       name='Conv2d_0c_3x3')

    # filters 256
    self.up_conv = layers.Conv2D(filters, (1, 1), name='Conv2d_1x1')

    self.scale = scale
    self.activation_fn = activation_fn

  def call(self, x, training=False):
    inputs = x
    branch1 = self.tower_conv(x, training=training)

    branch2 = self.tower_conv1_0(x, training=training)
    branch2 = self.tower_conv1_1(branch2, training=training)

    branch3 = self.tower_conv2_0(x, training=training)
    branch3 = self.tower_conv2_1(branch3, training=training)
    branch3 = self.tower_conv2_2(branch3, training=training)

    mixed = tf.concat([branch1, branch2, branch3], axis=3)  # 32 * 3 == 96
    x = self.up_conv(mixed, training=training)  # 96 => 256

    x = inputs + self.scale * x

    if self.activation_fn is not None:
      x = self.activation_fn(x)

    return x

class Block17(keras.Model):
  def __init__(self, filters, scale=1.0, activation_fn=tf.nn.relu):
    super(Block17, self).__init__()
    # branch 0
    self.tower_conv = BaseConvBlock(128, (1, 1),
                                    padding='same',
                                    name='Conv2d_1x1')

    # branch 1
    self.tower_conv1_0 = BaseConvBlock(128, (1, 1),
                                       padding='same',
                                       name='Conv2d_0a_1x1')
    self.tower_conv1_1 = BaseConvBlock(128, (1, 7),
                                       padding='same',
                                       name='Conv2d_0b_1x7')
    self.tower_conv1_2 = BaseConvBlock(128, (7, 1),
                                       padding='same',
                                       name='Conv2d_0c_7x1')

    self.up_conv = layers.Conv2D(filters, (1, 1), name='Conv2d_1x1')

    self.scale = scale
    self.activation_fn = activation_fn

  def call(self, x, training=False):
    inputs = x
    branch1 = self.tower_conv(x, training=training)

    branch2 = self.tower_conv1_0(x, training=training)
    branch2 = self.tower_conv1_1(branch2, training=training)
    branch2 = self.tower_conv1_2(branch2, training=training)

    mixed = tf.concat([branch1, branch2], axis=3)
    x = self.up_conv(mixed, training=training)

    x = inputs + self.scale * x

    if self.activation_fn is not None:
      x = self.activation_fn(x)

    return x

class Block8(keras.Model):
  def __init__(self, filters, scale=1.0, activation_fn=tf.nn.relu):
    super(Block8, self).__init__()
    # branch 0
    self.tower_conv = BaseConvBlock(192, (1, 1),
                                    padding='same',
                                    name='Conv2d_1x1')

    # branch 1
    self.tower_conv1_0 = BaseConvBlock(192, (1, 1),
                                       padding='same',
                                       name='Conv2d_0a_1x1')
    self.tower_conv1_1 = BaseConvBlock(192, (1, 3),
                                       padding='same',
                                       name='Conv2d_0b_1x3')
    self.tower_conv1_2 = BaseConvBlock(192, (3, 1),
                                       padding='same',
                                       name='Conv2d_0c_3x1')

    self.up_conv = layers.Conv2D(filters, (1, 1), name='Conv2d_1x1')

    self.scale = scale
    self.activation_fn = activation_fn

  def call(self, x, training=False):
    inputs = x
    branch1 = self.tower_conv(x, training=training)

    branch2 = self.tower_conv1_0(x, training=training)
    branch2 = self.tower_conv1_1(branch2, training=training)
    branch2 = self.tower_conv1_2(branch2, training=training)

    mixed = tf.concat([branch1, branch2], axis=3)
    x = self.up_conv(mixed, training=training)

    x = inputs + self.scale * x

    if self.activation_fn is not None:
      x = self.activation_fn(x)

    return x

class ReductionA(keras.Model):
  def __init__(self, k, l, m, n):
    super(ReductionA, self).__init__()
    # branch 0
    self.tower_conv = BaseConvBlock(n, (3, 3), (2, 2),
                                    padding='valid',
                                    name='Conv2d_1a_3x3')

    # branch 1
    self.tower_conv1_0 = BaseConvBlock(k, (1, 1),
                                       padding='same',
                                       name='Conv2d_0a_1x1')
    self.tower_conv1_1 = BaseConvBlock(l, (3, 3),
                                       padding='same',
                                       name='Conv2d_0b_3x3')
    self.tower_conv1_2 = BaseConvBlock(m, (3, 3), (2, 2),
                                       padding='valid',
                                       name='Conv2d_1a_3x3')

    # branch 2
    self.tower_pool = layers.MaxPooling2D((3, 3), (2, 2), padding='valid',
                                          name='MaxPool_1a_3x3')
  def call(self, x, training=False):
    branch0 = self.tower_conv(x, training=training)  # n

    branch1 = self.tower_conv1_0(x, training=training)
    branch1 = self.tower_conv1_1(branch1, training=training)
    branch1 = self.tower_conv1_2(branch1, training=training)  # l

    branch2 = self.tower_pool(x)  # x

    x = tf.concat([branch0, branch1, branch2], axis=3)  # n + l + x
    return x

class ReductionB(keras.Model):
  def __init__(self):
    super(ReductionB, self).__init__()
    # branch 0
    self.tower_conv = BaseConvBlock(256, (1, 1),
                                    padding='same',
                                    name='Conv2d_0a_1x1')
    self.tower_conv_1 = BaseConvBlock(384, (3, 3),
                                      strides=(2, 2),
                                      padding='valid',
                                      name='Conv2d_1a_3x3')

    # branch 1
    self.tower_conv1 = BaseConvBlock(256, (1, 1),
                                     padding='same',
                                     name='Conv2d_0a_1x1')
    self.tower_conv1_1 = BaseConvBlock(256, (3, 3),
                                       strides=(2, 2),
                                       padding='valid',
                                       name='Conv2d_1a_3x3')

    # branch 2
    self.tower_conv2 = BaseConvBlock(256, (1, 1),
                                     padding='same',
                                     name='Conv2d_0a_1x1')
    self.tower_conv2_1 = BaseConvBlock(256, (3, 3),
                                       padding='same',
                                       name='Conv2d_0b_3x3')
    self.tower_conv2_2 = BaseConvBlock(256, (3, 3),
                                       strides=(2, 2),
                                       padding='valid',
                                       name='Conv2d_1a_3x3')

    self.tower_pool = layers.MaxPooling2D((3, 3), strides=(2, 2),
                                          padding='valid',
                                          name='MaxPool_1a_3x3')

  def call(self, x, training=False):
    inputs = x

    branch0 = self.tower_conv(x, training=training)
    branch0 = self.tower_conv_1(branch0, training=training)

    branch1 = self.tower_conv1(x, training=training)
    branch1 = self.tower_conv1_1(branch1, training=training)

    branch2 = self.tower_conv2(x, training=training)
    branch2 = self.tower_conv2_1(branch2, training=training)
    branch2 = self.tower_conv2_2(branch2, training=training)

    branch3 = self.tower_pool(x)

    x = tf.concat([branch0, branch1, branch2, branch3], axis=3)
    return x

class InceptionResNetV1(keras.Model):
  def __init__(self,
               dropout_keep_prob=0.4,
               bottleneck_layer_size=512,
               use_center_loss=False,
               num_classes=8631):
    super(InceptionResNetV1, self).__init__()

    self.conv1 = BaseConvBlock(32, (3, 3),
                               strides=(2, 2),
                               padding='valid',
                               name='Conv2d_1a_3x3')
    self.conv2 = BaseConvBlock(32, (3, 3),
                               padding='valid',
                               name='Conv2d_2a_3x3')
    self.conv3 = BaseConvBlock(64, (3, 3),
                               padding='same',
                               name='Conv2d_2b_3x3')

    self.pool = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid',
                                    name='MaxPool_3a_3x3')
    self.conv4 = BaseConvBlock(80, (1, 1),
                               padding='valid',
                               name='Conv2d_3b_1x1')
    self.conv5 = BaseConvBlock(192, (3, 3),
                               padding='valid',
                               name='Conv2d_4a_3x3')
    self.conv6 = BaseConvBlock(256, (3, 3),
                               strides=(2, 2),
                               padding='valid',
                               name='Conv2d_4b_3x3')

    self.block35 = [Block35(256, scale=0.17) for _ in range(5)]

    self.reduction_a = ReductionA(192, 192, 256, 384)  # 256 + 256 + 384

    self.block17 = [Block17(256 + 256 + 384, scale=0.10) for _ in range(10)]

    self.reduction_b = ReductionB()

    self.block8 = [Block8(1792, scale=0.20, activation_fn=tf.nn.relu \
                          if i < 5 else None) for i in range(6)]

    self.avg_pool = layers.GlobalAveragePooling2D(name='AvgPool_1a_global')

    self.flatten = layers.Flatten()
    self.dropout = layers.Dropout(1 - dropout_keep_prob)

    self.embedding = layers.Dense(bottleneck_layer_size, name='Bottleneck',
                                  use_bias=False)
    self.last_bn = layers.BatchNormalization()
    # pylint: disable=line-too-long
    self.classifier = layers.Dense(num_classes,
                                   kernel_initializer=initializers.glorot_uniform,
                                   kernel_regularizer=regularizers.l2(5e-4),
                                   name='Logits')
    self.activation = layers.Activation('softmax')

    self.use_center_loss = use_center_loss
    if use_center_loss:
      self.center_loss = CenterLoss(num_classes, 512)

  def build(self, input_shape):
    if self.use_center_loss:
      self.center_loss.build(input_shape)
    super(InceptionResNetV1, self).build(input_shape)

  def calculate_embedding(self, prelogits):
    # https://github.com/tamerthamoqa/facenet-pytorch-vggface2/blob/master/models/resnet.py
    x = tf.nn.l2_normalize(prelogits, axis=1, epsilon=1e-10)
    x = x * 10.
    return x

  def calculate_center_loss(self, features, labels):
    assert self.use_center_loss
    return self.center_loss(features, labels)

  def call(self, x, training=False):
    if len(x.shape) == 4:
      x = self.conv1(x, training=training)
      x = self.conv2(x, training=training)
      x = self.conv3(x, training=training)
      x = self.pool(x, training=training)
      x = self.conv4(x, training=training)
      x = self.conv5(x, training=training)
      x = self.conv6(x, training=training)
      for block in self.block35:
        x = block(x, training=training)
      x = self.reduction_a(x, training=training)
      for block in self.block17:
        x = block(x, training=training)
      x = self.reduction_b(x, training=training)
      for block in self.block8:
        x = block(x, training=training)
      x = self.avg_pool(x)
      x = self.flatten(x)
      x = self.dropout(x, training=training)
      prelogits = self.embedding(x)
      prelogits = self.last_bn(prelogits, training=training)
    else:
      assert len(x.shape) == 2
      prelogits = x
    x = self.calculate_embedding(prelogits)
    x = self.classifier(x)
    x = self.activation(x)
    return x, prelogits

  def feature_extract(self, x):
    """just feature extraction without training flag"""
    x = self.conv1(x, training=False)
    x = self.conv2(x, training=False)
    x = self.conv3(x, training=False)
    x = self.pool(x, training=False)
    x = self.conv4(x, training=False)
    x = self.conv5(x, training=False)
    x = self.conv6(x, training=False)
    for block in self.block35:
      x = block(x, training=False)
    x = self.reduction_a(x, training=False)
    for block in self.block17:
      x = block(x, training=False)
    x = self.reduction_b(x, training=False)
    for block in self.block8:
      x = block(x, training=False)
    x = self.avg_pool(x)
    x = self.flatten(x)
    x = self.dropout(x, training=False)
    prelogits = self.embedding(x)
    prelogits = self.last_bn(prelogits, training=False)
    return prelogits

class ThawedModel1(keras.Model):
  """
  assume only first three layers are fixed
  input size should be 38 x 38 x 64
  """
  def __init__(self,
               dropout_keep_prob=0.4,
               bottleneck_layer_size=512,
               use_center_loss=False,
               num_classes=8631):
    super(ThawedModel1, self).__init__()

    self.conv4 = BaseConvBlock(80, (1, 1),
                               padding='valid',
                               name='Conv2d_3b_1x1')
    self.conv5 = BaseConvBlock(192, (3, 3),
                               padding='valid',
                               name='Conv2d_4a_3x3')
    self.conv6 = BaseConvBlock(256, (3, 3),
                               strides=(2, 2),
                               padding='valid',
                               name='Conv2d_4b_3x3')

    self.block35 = [Block35(256, scale=0.17) for _ in range(5)]

    self.reduction_a = ReductionA(192, 192, 256, 384)  # 256 + 256 + 384

    self.block17 = [Block17(256 + 256 + 384, scale=0.10) for _ in range(10)]

    self.reduction_b = ReductionB()

    self.block8 = [Block8(1792, scale=0.20, activation_fn=tf.nn.relu \
                          if i < 5 else None) for i in range(6)]

    self.avg_pool = layers.GlobalAveragePooling2D(name='AvgPool_1a_global')

    self.flatten = layers.Flatten()
    self.dropout = layers.Dropout(1 - dropout_keep_prob)

    self.embedding = layers.Dense(bottleneck_layer_size, name='Bottleneck',
                                  use_bias=False)
    self.last_bn = layers.BatchNormalization()
    # pylint: disable=line-too-long
    self.classifier = layers.Dense(num_classes,
                                   kernel_initializer=initializers.glorot_uniform,
                                   kernel_regularizer=regularizers.l2(5e-4),
                                   name='Logits')
    self.activation = layers.Activation('softmax')

    self.use_center_loss = use_center_loss
    if use_center_loss:
      self.center_loss = CenterLoss(num_classes, 512)

  def build(self, input_shape):
    if self.use_center_loss:
      self.center_loss.build(input_shape)
    super(ThawedModel1, self).build(input_shape)

  def calculate_embedding(self, prelogits):
    # https://github.com/tamerthamoqa/facenet-pytorch-vggface2/blob/master/models/resnet.py
    x = tf.nn.l2_normalize(prelogits, axis=1, epsilon=1e-10)
    x = x * 10.
    return x

  def calculate_center_loss(self, features, labels):
    assert self.use_center_loss
    return self.center_loss(features, labels)

  def call(self, x, training=False):
    x = self.conv4(x, training=training)
    x = self.conv5(x, training=training)
    x = self.conv6(x, training=training)
    for block in self.block35:
      x = block(x, training=training)
    x = self.reduction_a(x, training=training)
    for block in self.block17:
      x = block(x, training=training)
    x = self.reduction_b(x, training=training)
    for block in self.block8:
      x = block(x, training=training)
    x = self.avg_pool(x)
    x = self.flatten(x)
    x = self.dropout(x, training=training)
    prelogits = self.embedding(x)
    prelogits = self.last_bn(prelogits, training=training)
    x = self.calculate_embedding(prelogits)
    x = self.classifier(x)
    x = self.activation(x)
    return x, prelogits

class ThawedModel2(keras.Model):
  """
  input size should be 17 x 17 x 256
  """
  def __init__(self,
               dropout_keep_prob=0.4,
               bottleneck_layer_size=512,
               use_center_loss=False,
               num_classes=8631):
    super(ThawedModel2, self).__init__()

    self.block35 = [Block35(256, scale=0.17) for _ in range(5)]

    self.reduction_a = ReductionA(192, 192, 256, 384)  # 256 + 256 + 384

    self.block17 = [Block17(256 + 256 + 384, scale=0.10) for _ in range(10)]

    self.reduction_b = ReductionB()

    self.block8 = [Block8(1792, scale=0.20, activation_fn=tf.nn.relu \
                          if i < 5 else None) for i in range(6)]

    self.avg_pool = layers.GlobalAveragePooling2D(name='AvgPool_1a_global')

    self.flatten = layers.Flatten()
    self.dropout = layers.Dropout(1 - dropout_keep_prob)

    self.embedding = layers.Dense(bottleneck_layer_size, name='Bottleneck',
                                  use_bias=False)
    self.last_bn = layers.BatchNormalization()
    # pylint: disable=line-too-long
    self.classifier = layers.Dense(num_classes,
                                   kernel_initializer=initializers.glorot_uniform,
                                   kernel_regularizer=regularizers.l2(5e-4),
                                   name='Logits')
    self.activation = layers.Activation('softmax')

    self.use_center_loss = use_center_loss
    if use_center_loss:
      self.center_loss = CenterLoss(num_classes, 512)

  def build(self, input_shape):
    if self.use_center_loss:
      self.center_loss.build(input_shape)
    super(ThawedModel2, self).build(input_shape)

  def calculate_embedding(self, prelogits):
    # https://github.com/tamerthamoqa/facenet-pytorch-vggface2/blob/master/models/resnet.py
    x = tf.nn.l2_normalize(prelogits, axis=1, epsilon=1e-10)
    x = x * 10.
    return x

  def calculate_center_loss(self, features, labels):
    assert self.use_center_loss
    return self.center_loss(features, labels)

  def call(self, x, training=False):
    for block in self.block35:
      x = block(x, training=training)
    x = self.reduction_a(x, training=training)
    for block in self.block17:
      x = block(x, training=training)
    x = self.reduction_b(x, training=training)
    for block in self.block8:
      x = block(x, training=training)
    x = self.avg_pool(x)
    x = self.flatten(x)
    x = self.dropout(x, training=training)
    prelogits = self.embedding(x)
    prelogits = self.last_bn(prelogits, training=training)
    x = self.calculate_embedding(prelogits)
    x = self.classifier(x)
    x = self.activation(x)
    return x, prelogits

class ThawedModel3(keras.Model):
  """
  input size should be 17 x 17 x 256
  """
  def __init__(self,
               dropout_keep_prob=0.4,
               bottleneck_layer_size=512,
               use_center_loss=False,
               num_classes=8631):
    super(ThawedModel3, self).__init__()

    self.reduction_a = ReductionA(192, 192, 256, 384)  # 256 + 256 + 384

    self.block17 = [Block17(256 + 256 + 384, scale=0.10) for _ in range(10)]

    self.reduction_b = ReductionB()

    self.block8 = [Block8(1792, scale=0.20, activation_fn=tf.nn.relu \
                          if i < 5 else None) for i in range(6)]

    self.avg_pool = layers.GlobalAveragePooling2D(name='AvgPool_1a_global')

    self.flatten = layers.Flatten()
    self.dropout = layers.Dropout(1 - dropout_keep_prob)

    self.embedding = layers.Dense(bottleneck_layer_size, name='Bottleneck',
                                  use_bias=False)
    self.last_bn = layers.BatchNormalization()
    # pylint: disable=line-too-long
    self.classifier = layers.Dense(num_classes,
                                   kernel_initializer=initializers.glorot_uniform,
                                   kernel_regularizer=regularizers.l2(5e-4),
                                   name='Logits')
    self.activation = layers.Activation('softmax')

    self.use_center_loss = use_center_loss
    if use_center_loss:
      self.center_loss = CenterLoss(num_classes, 512)

  def build(self, input_shape):
    if self.use_center_loss:
      self.center_loss.build(input_shape)
    super(ThawedModel3, self).build(input_shape)

  def calculate_embedding(self, prelogits):
    # https://github.com/tamerthamoqa/facenet-pytorch-vggface2/blob/master/models/resnet.py
    x = tf.nn.l2_normalize(prelogits, axis=1, epsilon=1e-10)
    x = x * 10.
    return x

  def calculate_center_loss(self, features, labels):
    assert self.use_center_loss
    return self.center_loss(features, labels)

  def call(self, x, training=False):
    x = self.reduction_a(x, training=training)
    for block in self.block17:
      x = block(x, training=training)
    x = self.reduction_b(x, training=training)
    for block in self.block8:
      x = block(x, training=training)
    x = self.avg_pool(x)
    x = self.flatten(x)
    x = self.dropout(x, training=training)
    prelogits = self.embedding(x)
    prelogits = self.last_bn(prelogits, training=training)
    x = self.calculate_embedding(prelogits)
    x = self.classifier(x)
    x = self.activation(x)
    return x, prelogits

class ThawedModel4(keras.Model):
  """
  input size should be 8 x 8 x 896
  """
  def __init__(self,
               dropout_keep_prob=0.4,
               bottleneck_layer_size=512,
               use_center_loss=False,
               num_classes=8631):
    super(ThawedModel4, self).__init__()

    self.reduction_b = ReductionB()

    self.block8 = [Block8(1792, scale=0.20, activation_fn=tf.nn.relu \
                          if i < 5 else None) for i in range(6)]

    self.avg_pool = layers.GlobalAveragePooling2D(name='AvgPool_1a_global')

    self.flatten = layers.Flatten()
    self.dropout = layers.Dropout(1 - dropout_keep_prob)

    self.embedding = layers.Dense(bottleneck_layer_size, name='Bottleneck',
                                  use_bias=False)
    self.last_bn = layers.BatchNormalization()
    # pylint: disable=line-too-long
    self.classifier = layers.Dense(num_classes,
                                   kernel_initializer=initializers.glorot_uniform,
                                   kernel_regularizer=regularizers.l2(5e-4),
                                   name='Logits')
    self.activation = layers.Activation('softmax')

    self.use_center_loss = use_center_loss
    if use_center_loss:
      self.center_loss = CenterLoss(num_classes, 512)

  def build(self, input_shape):
    if self.use_center_loss:
      self.center_loss.build(input_shape)
    super(ThawedModel4, self).build(input_shape)

  def calculate_embedding(self, prelogits):
    # https://github.com/tamerthamoqa/facenet-pytorch-vggface2/blob/master/models/resnet.py
    x = tf.nn.l2_normalize(prelogits, axis=1, epsilon=1e-10)
    x = x * 10.
    return x

  def calculate_center_loss(self, features, labels):
    assert self.use_center_loss
    return self.center_loss(features, labels)

  def call(self, x, training=False):
    x = self.reduction_b(x, training=training)
    for block in self.block8:
      x = block(x, training=training)
    x = self.avg_pool(x)
    x = self.flatten(x)
    x = self.dropout(x, training=training)
    prelogits = self.embedding(x)
    prelogits = self.last_bn(prelogits, training=training)
    x = self.calculate_embedding(prelogits)
    x = self.classifier(x)
    x = self.activation(x)
    return x, prelogits

class ThawedModel5(keras.Model):
  """
  input size should be 1792
  """
  def __init__(self,
               dropout_keep_prob=0.4,
               bottleneck_layer_size=512,
               use_center_loss=False,
               num_classes=8631):
    super(ThawedModel5, self).__init__()

    self.embedding = layers.Dense(bottleneck_layer_size, name='Bottleneck',
                                  use_bias=False)
    self.last_bn = layers.BatchNormalization()
    # pylint: disable=line-too-long
    self.classifier = layers.Dense(num_classes,
                                   kernel_initializer=initializers.glorot_uniform,
                                   kernel_regularizer=regularizers.l2(5e-4),
                                   name='Logits')
    self.activation = layers.Activation('softmax')

    self.use_center_loss = use_center_loss
    if use_center_loss:
      self.center_loss = CenterLoss(num_classes, 512)

  def build(self, input_shape):
    if self.use_center_loss:
      self.center_loss.build(input_shape)
    super(ThawedModel5, self).build(input_shape)

  def calculate_embedding(self, prelogits):
    # https://github.com/tamerthamoqa/facenet-pytorch-vggface2/blob/master/models/resnet.py
    x = tf.nn.l2_normalize(prelogits, axis=1, epsilon=1e-10)
    x = x * 10.
    return x

  def calculate_center_loss(self, features, labels):
    assert self.use_center_loss
    return self.center_loss(features, labels)

  def call(self, x, training=False):
    prelogits = self.embedding(x)
    prelogits = self.last_bn(prelogits, training=training)
    x = self.calculate_embedding(prelogits)
    x = self.classifier(x)
    x = self.activation(x)
    return x, prelogits

class ThawedModel6(keras.Model):
  """
  input size should be 512
  """
  def __init__(self,
               dropout_keep_prob=0.4,
               bottleneck_layer_size=512,
               use_center_loss=False,
               num_classes=8631):
    super(ThawedModel6, self).__init__()

    self.classifier = layers.Dense(num_classes,
                                   kernel_initializer=initializers.glorot_uniform,
                                   kernel_regularizer=regularizers.l2(5e-4),
                                   name='Logits')
    self.activation = layers.Activation('softmax')

    self.use_center_loss = use_center_loss
    if use_center_loss:
      self.center_loss = CenterLoss(num_classes, 512)

  def build(self, input_shape):
    if self.use_center_loss:
      self.center_loss.build(input_shape)
    super(ThawedModel6, self).build(input_shape)

  def calculate_embedding(self, prelogits):
    # https://github.com/tamerthamoqa/facenet-pytorch-vggface2/blob/master/models/resnet.py
    x = tf.nn.l2_normalize(prelogits, axis=1, epsilon=1e-10)
    x = x * 10.
    return x

  def calculate_center_loss(self, features, labels):
    assert self.use_center_loss
    return self.center_loss(features, labels)

  def call(self, x, training=False):
    prelogits = x
    x = self.calculate_embedding(prelogits)
    x = self.classifier(x)
    x = self.activation(x)
    return x, prelogits
