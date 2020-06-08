# FedEx Server

## FaceNet Preparation

1. download checkpoints
   `python ./scripts/prepare_facenet.py`
2. main script is `facenet_training.py`
    - supports eval / training / tflite export
3. you can create label txt file via `python create_facenet_label.py`
4. for data preparation use
    - `utils/download_dataset.py` for downloads whole dataset (about 40GB)
    - `preprocess_image.py` for preprocess datasets (use `_make_dataset` instead of `make_dataset_cached`)

## TFLite Convert Example (ResNet50)

1. download checkpoints
   `./scripts/prepare_resnet50.sh`
2. run python script
   `python convert_tflite.py`
3. output model `resnet50.tflite` and `mobilenet_v1.tflite` will be saved at `$PROJECT_ROOT/client/app/src/main/assets`

## Read intermediates Tensor Example

1. At the application click `inference` button
2. run `./scripts/get_intermediates.sh`
3. run `python read_intermediate_tensor_test.py`
