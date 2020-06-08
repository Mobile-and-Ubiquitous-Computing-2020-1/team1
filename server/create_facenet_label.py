"""
create faceID => real name label
"""
import glob

import numpy as np


def main():
  # Error
  # csv_data = np.genfromtxt('./assets/identity_meta.csv', delimiter=',')

  classes = glob.glob('/cmsdata/ssd1/cmslab/vggface2/final/train/*')
  classes = list(map(lambda x: x.split('/')[-1], classes))

  classes = sorted(classes, key=lambda x: int(x[1:]))

  num_classes = len(classes)

  name2label = {name: label for label, name in enumerate(classes)}

  label2name = {}
  ident2name = {}
  with open('./assets/identity_meta.csv', 'r') as f:
    for line in f:
      ident, real_name, _, _, _ = line.strip().split(' ')
      ident = ident[:-1]
      real_name = real_name[1:-2]
      if ident in name2label:
        label2name[name2label[ident]] = real_name
        ident2name[ident] = real_name

  with open('./assets/face_label.txt', 'w') as f:
    for i in range(num_classes):
      f.write('%s\n' % label2name[i])

if __name__ == "__main__":
  main()
