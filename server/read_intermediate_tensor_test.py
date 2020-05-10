import numpy as np

import tensorflow as tf

def main():
  target_dtype = tf.float32
  shape_per_tensor = (28, 28, 256)  # test for mobilenet_v1
  tensor_size = np.prod(shape_per_tensor)
  with open('./intermediate-features/intermediates', 'rb') as f:
    raw_bytes = f.read()
    bytes_size = tensor_size * target_dtype.size
    tensors = []
    for i in range(0, len(raw_bytes), bytes_size):
      raw_byte = raw_bytes[i:i+bytes_size]
      tensor = tf.io.decode_raw(raw_byte, target_dtype)
      tensor = tf.reshape(tensor, shape_per_tensor)
      tensors.append(tensor)

    tensor = tf.stack(tensors, axis=0)
    print(tensor)
    print(tensor.shape)
    print(type(tensor))

if __name__ == "__main__":
  main()
