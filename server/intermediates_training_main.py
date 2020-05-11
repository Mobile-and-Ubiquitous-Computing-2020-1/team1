"""
scripts for additional training using intermediates features
"""

import os

from models import mobilenet
from models import mobilenet_thawed
from utils.dataload import load_intermediates_feature_data

import tensorflow as tf

MODEL_PATH = './checkpoints/mobilenet_v1_1.0_224/mobilenet_1_0_224_tf.h5'

def main():
  if not os.path.exists(MODEL_PATH):
    os.system('./scripts/prepare_mobilenetv1.sh')

  thawed_model = mobilenet_thawed(input_shape=(14, 14, 512))
  thawed_model.load_weights(MODEL_PATH, by_name=True)  # just load by name
  thawed_model.summary()


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '1'
  main()
