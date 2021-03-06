"""
preprocess dataset by using MTCNN
"""
from __future__ import absolute_import, division, print_function

import argparse
import glob
import os

import cv2

from mtcnn import MTCNN
from utils.log import fedex_logger as logger


def _make_dataset(input_dir, output_dir, image_size, margin, split='train'):
  """process each image and save result image
  it saves cropped face region and bounding box information of face

  @param input_dir: root of input directory
  @param output_dir: root of output directory
  @param image_size: resize image size
  @param margin: margin of cropped image
  @param split: train / test

  return None
  """
  input_dir = os.path.join(input_dir, split)

  output_root = os.path.join(output_dir, split)
  if not os.path.exists(output_root):
    os.makedirs(output_root)

  class_folders = glob.glob(os.path.join(input_dir, '*'))
  detector = MTCNN()

  for class_folder in class_folders:
    target_output_dir = os.path.join(output_root, class_folder.split('/')[-1])
    if not os.path.exists(target_output_dir):
      os.makedirs(target_output_dir)

    target_files = glob.glob(os.path.join(class_folder, '*'))
    logger.debug('processing %s...', class_folder)
    for file in target_files:
      img = cv2.imread(file)
      detect_result = detector.detect_faces(img)

      if not detect_result:
        logger.warning('WARNING: failed to detect face in file %s, skip', file)
        continue

      x0, y0, width, height = detect_result[0]['box']
      x1, y1 = x0 + width, y0 + height

      x0 = max(x0 - margin // 2, 0)
      y0 = max(y0 - margin // 2, 0)
      x1 = min(x1 + margin // 2, img.shape[1])
      y1 = min(y1 + margin // 2, img.shape[0])

      face_img = img[y0:y1, x0:x1, :]
      face_img = cv2.resize(face_img, dsize=(image_size, image_size),
                            interpolation=cv2.INTER_LINEAR)

      filename = file.split('/')[-1]
      img_name = filename.split('.')[0]
      cv2.imwrite(os.path.join(target_output_dir, filename),
                  face_img)
      with open(os.path.join(target_output_dir, img_name + '.txt'), 'w') as f:
        f.write('%d %d %d %d\n' % (x0, y0, x1, y1))
    logger.debug('processing %s finished!', class_folder)

def make_dataset_cached(input_dir, output_dir, image_size, split='train'):
  input_dir = os.path.join(input_dir, split)

  output_root = os.path.join(output_dir, split)
  if not os.path.exists(output_root):
    os.makedirs(output_root)

  class_folders = glob.glob(os.path.join(input_dir, '*'))
  for class_folder in class_folders:
    target_output_dir = os.path.join(output_root, class_folder.split('/')[-1])
    if not os.path.exists(target_output_dir):
      os.makedirs(target_output_dir)

    target_files = glob.glob(os.path.join(class_folder, '*'))
    logger.debug('processing %s...', class_folder)
    for file in target_files:
      filename = file.split('/')[-1][:-4]
      class_name = class_folder.split('/')[-1]
      cached_file = '/cmsdata/ssd1/cmslab/vggface2/processed/train/%s/%s.txt' \
        % (class_name, filename)
      if not os.path.exists(cached_file):
        continue
      with open(cached_file, 'r') as f:
        x0, y0, x1, y1 = map(int, f.read().strip().split(' '))
      img = cv2.imread(file)
      face_img = img[y0:y1, x0:x1, :]
      face_img = cv2.resize(face_img, dsize=(image_size, image_size),
                            interpolation=cv2.INTER_LINEAR)
      cv2.imwrite(os.path.join(target_output_dir, filename + '.jpg'),
                  face_img)
    logger.debug('processing %s finished!', class_folder)

def main(args):
  input_dir = args.input_dir
  output_dir = args.output_dir
  image_size = args.image_size
  margin = args.margin
  split = args.split

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # _make_dataset(input_dir, output_dir, image_size, margin, split)
  make_dataset_cached(input_dir, output_dir, image_size)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('input_dir', type=str,
                      help='unaligned images directory')
  parser.add_argument('output_dir', type=str,
                      help='output images directory')
  parser.add_argument('--split', type=str,
                      default='train')
  parser.add_argument('--image_size', type=int,
                      default=182)
  parser.add_argument('--margin', type=int,
                      default=44)

  main(parser.parse_args())
