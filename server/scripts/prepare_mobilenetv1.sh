# run it in $PROJECT_HOME/server

if [ ! -d "checkpoints" ]
then
  mkdir checkpoints
fi

cd checkpoints

MOBILENET_NAME=mobilenet_v1_1.0_224

if [ ! -d "$MOBILENET_NAME" ]
then
  mkdir $MOBILENET_NAME
fi

cd $MOBILENET_NAME

if [ ! -f "mobilenet_1_0_224_tf.h5" ]
then
  wget https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_1_0_224_tf.h5
fi
