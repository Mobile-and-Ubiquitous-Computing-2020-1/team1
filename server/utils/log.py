"""
Simple logger for FeDex
"""
import logging as _logging
import os

from distutils import sysconfig

_path_prefix = sysconfig.get_python_lib()

class _PathShortenFormatter(_logging.Formatter):
  """modified formatter that omit site-package path prefix"""
  def format(self, record):
    if hasattr(record, 'pathname'):
      pathname = record.pathname
      if _path_prefix in pathname:
        pathname = pathname[len(_path_prefix) + 1:]
        record.pathname = pathname

    return super(_PathShortenFormatter, self).format(record)

formatter = _PathShortenFormatter(
    fmt='%(asctime)s.%(msecs)06d: %(name)s %(levelname).1s ' \
        '@%(thread)d %(pathname)s:%(lineno)d %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
stream_handler = _logging.StreamHandler()
stream_handler.setFormatter(formatter)

fedex_logger = _logging.getLogger('FeDex')
try:
  fedex_logger.setLevel(os.environ["FEDEX_LOG_LEVEL"])
except KeyError:
  fedex_logger.setLevel('DEBUG')
fedex_logger.addHandler(stream_handler)

_logging.basicConfig(filename="log.txt")
