"""
convert prepared resnet model into tflite model
run this python script in server root
"""

import os
import argparse

import tensorflow as tf

from tensorflow import lite as tf_lite

from models import resnet50, mobilenet

TFLITE_MODEL_DIR = '../client/app/src/main/assets'

def convert_resnet() -> None:
  CHECKPOINT_DIR = './checkpoints/resnet50'

  checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)

  model = resnet50(1001)
  model.load_weights(checkpoint)

  converter = tf_lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  if not os.path.exists(TFLITE_MODEL_DIR):
    os.mkdir(TFLITE_MODEL_DIR)

  with open(os.path.join(TFLITE_MODEL_DIR, 'resnet50.tflite'), 'wb') as f:
    f.write(tflite_model)

def convert_mobilenet() -> None:
  # Create tf model
  model = mobilenet()

  # Save tf model
  SAVED_MODEL_DIR = '/tmp/mobilenet/'
  tf.saved_model.save(model, SAVED_MODEL_DIR)

  # Convert to tflite
  converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
  tflite_model = converter.convert()

  # Save tflite model
  with open(os.path.join(TFLITE_MODEL_DIR, 'mobilenet_v1.tflite'), 'wb') as f:
    f.write(tflite_model)

def main(args) -> None:
    if args.model == 'all':
        convert_resnet()
        convert_mobilenet()
    elif args.model == 'resnet50':
        convert_resnet()
    elif args.model == 'mobilenet':
        convert_mobilenet()
    else:
        raise ValueError("Not supported model: {}".format(args.model))

if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('-m', '--model', default='all',
                      help='model to convert')
  args = parser.parse_args()
  main(args)
