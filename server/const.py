"""FedEx constants."""

import os

# total resource
MAX_GPUS = 6

# urls
PING_URL = "/ping"
PUSH_URL = "feature/push"
PULL_URL = "model/best/pull"
INFO_URL = "model/best/info"

UPDATE_URL = "model/train"


# paths
FEATURE_PATH = "intermediate-features"
os.makedirs(FEATURE_PATH, exist_ok=True)

MODEL_PATH = "tflite-models"

# command
BACKGROUND_JOB = "bg.py"
