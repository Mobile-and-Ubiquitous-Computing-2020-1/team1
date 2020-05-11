"""
DataLoader utils
"""

import numpy as np

import tensorflow as tf

def load_intermediates_feature_data(data_path,
                                    tensor_shape,
                                    target_dtype=tf.float32):
  tensor_size = np.prod(tensor_shape)
  with open(data_path, 'rb') as f:
    raw_bytes = f.read()
    bytes_size = tensor_size * target_dtype.size

    tensors = []

    for i in range(0, len(raw_bytes), bytes_size):
      raw_byte = raw_bytes[i:i + bytes_size]
      tensor = tf.io.decode_raw(raw_byte, target_dtype)
      tensor = tf.reshape(tensor, shape_per_tensor)
      tensors.append(tensor)

    return tf.data.Dataset.from_tensor_slices(tensors)
