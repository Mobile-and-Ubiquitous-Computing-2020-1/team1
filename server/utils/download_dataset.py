"""
Download script of VGGFace2

Must sign up to
https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/data_infor.html
"""
import argparse
import getpass
import os
import sys

import requests

LOGIN_URL = "http://zeus.robots.ox.ac.uk/vgg_face2/login/"
FILE_URLS = [
    "http://zeus.robots.ox.ac.uk/vgg_face2/get_file?fname=vggface2_train.tar.gz",
    "http://zeus.robots.ox.ac.uk/vgg_face2/get_file?fname=vggface2_test.tar.gz",
]

def main(args):
  data_dir = args.data_dir
  print('Please enter your VGG Face 2 credentials:')
  user_string = input('    User: ')
  password_string = getpass.getpass(prompt='    Password: ')

  payload = {
      'username': user_string,
      'password': password_string
  }

  session = requests.session()
  r = session.get(LOGIN_URL)

  if 'csrftoken' in session.cookies:
    csrftoken = session.cookies['csrftoken']
  elif 'csrf' in session.cookies:
    csrftoken = session.cookies['csrf']
  else:
    raise ValueError("Unable to locate CSRF token.")

  payload['csrfmiddlewaretoken'] = csrftoken

  r = session.post(LOGIN_URL, data=payload)

  for FILE_URL in FILE_URLS:
    filename = FILE_URL.split('=')[-1]

    with open(os.path.join(data_dir, filename), "wb") as f:
      print(f"Downloading file: `{filename}`")
      r = session.get(FILE_URL, data=payload, stream=True)
      bytes_written = 0
      for data in r.iter_content(chunk_size=4096):
        f.write(data)
        bytes_written += len(data)
        MiB = bytes_written / (1024 * 1024)
        sys.stdout.write(f"\r{MiB:0.2f} MiB downloaded...")
        sys.stdout.flush()

  print("\nDone.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('data_dir', type=str,
                      help='data directory prefix (where to store)')
  main(parser.parse_args())
