"""
Download pretrained facenet (InceptionResNetV1) from
https://drive.google.com/uc?export=download&id=1cWLH_hPns8kSfMz9kKl9PsG5aNV2VSMn
https://drive.google.com/uc?export=download&id=1mAie3nzZeno9UIzFXvmVZrDG3kwML46X
"""
import os
from requests.adapters import HTTPAdapter
import requests

def main():
  # pylint: disable=line-too-long
  feature_path = 'https://drive.google.com/uc?export=download&id=1cWLH_hPns8kSfMz9kKl9PsG5aNV2VSMn'
  classifier_path = 'https://drive.google.com/uc?export=download&id=1mAie3nzZeno9UIzFXvmVZrDG3kwML46X'
  # pylint: enable=line-too-long
  download_path = './checkpoints/inception_resnet_v1'

  if not os.path.exists(download_path):
    os.makedirs(download_path)

  identifiers = ['feature_map', 'classifier']

  for i, path in enumerate([feature_path, classifier_path]):
    cached_file = os.path.join(download_path,
                               '{}_{}.pt'.format('vggface2', identifiers[i]))
    if not os.path.exists(cached_file):
      print('Downloading parameters ({}/2)'.format(i+1))
      s = requests.Session()
      s.mount('https://', HTTPAdapter(max_retries=10))
      r = s.get(path, allow_redirects=True)
      with open(cached_file, 'wb') as f:
        f.write(r.content)

if __name__ == "__main__":
  main()
