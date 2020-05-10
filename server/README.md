# FedEx Server

## TFLite Convert Example (ResNet50)
  1. download checkpoints  
  `./scripts/prepare_resnet50.sh`
  2. run python script  
  `python convert_tflite.py`
  3. output model `resnet50.tflite` will be saved at tflite-models directory