"""
convert prepared resnet model into tflite model
run this python script in server root
"""

import os

import tensorflow as tf

from tensorflow import lite as tf_lite

from models import resnet50

CHECKPOINT_DIR = './checkpoints/resnet50'
TFLITE_MODEL_DIR = '../client/app/src/main/assets'

def main() -> None:

  checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)

  model = resnet50(1001)
  model.load_weights(checkpoint)

  converter = tf_lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  if not os.path.exists(TFLITE_MODEL_DIR):
    os.mkdir(TFLITE_MODEL_DIR)

  with open(os.path.join(TFLITE_MODEL_DIR, 'resnet50.tflite'), 'wb') as f:
    f.write(tflite_model)

if __name__ == "__main__":
  main()
