# FedEx Server

## TFLite Convert Example (ResNet50)

1. download checkpoints
   `./scripts/prepare_resnet50.sh`
2. run python script
   `python convert_tflite.py`
3. output model `resnet50.tflite` and `mobilenet_v1.tflite` will be saved at `$PROJECT_ROOT/client/app/src/main/assets`

## Read intermediates Tensor Example

1. run `./scripts/make_intermediates.sh`
2. At the application click `inference` button
3. run `./scripts/get_intermediates.sh`
4. run `python read_intermediate_tensor_test.py`
