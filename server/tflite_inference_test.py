import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image

import tensorflow as tf
from models import resnet50
from tensorflow import lite as tf_lite

CHECKPOINT_DIR = './checkpoints/resnet50'
TF_LITE_MODEL = './tflite-models/resnet50.tflite'

def run_tflite(interpreter: tf_lite.Interpreter, inputs: np.ndarray) \
    -> Tuple[np.ndarray, np.ndarray]:
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]['index'], inputs)
  interpreter.invoke()

  softmax_result = interpreter.get_tensor(output_details[0]['index'])
  intermediate_result = interpreter.get_tensor(output_details[1]['index'])

  return softmax_result, intermediate_result

def run_keras(model: tf.keras.Model, inputs: np.ndarray) \
    -> Tuple[np.ndarray, np.ndarray]:

  softmax_result, intermediate_result = model(inputs, training=False)

  return softmax_result.numpy(), intermediate_result.numpy()

def main():
  # load tensorflow keras model
  image_files = glob.glob('./assets/imagenet-val-samples/*')
  checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)

  model = resnet50(1001)
  model.load_weights(checkpoint)

  # load tensorflow lite model
  interpreter = tf_lite.Interpreter(model_path=TF_LITE_MODEL)
  interpreter.allocate_tensors()

  for image_file in image_files:
    image = Image.open(image_file)
    image = image.resize((224, 224))
    image_data = np.asarray(image, dtype=np.float32)
    image_data = np.expand_dims(image_data, axis=0)

    tf_outputs = run_keras(model, image_data)
    tflite_outputs = run_tflite(interpreter, image_data)

    print('Diff 1: %.6f, Diff 2: %.6f' % \
          (np.mean(tf_outputs[0] - tflite_outputs[0]),
           np.mean(tf_outputs[1] - tflite_outputs[1])))

if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  main()
