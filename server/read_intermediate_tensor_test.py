import tensorflow as tf

def main():
  with open('./intermediate-features/intermediates', 'rb') as f:
    raw_bytes = f.read()
    tensor = tf.io.decode_raw(raw_bytes, tf.float32)
    print(tensor)
    print(tensor.shape)
    print(type(tensor))

if __name__ == "__main__":
  main()
