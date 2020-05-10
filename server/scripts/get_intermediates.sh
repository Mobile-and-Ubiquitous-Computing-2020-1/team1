if [ ! -d "intermediate-features" ]
then
  mkdir intermediate-features
fi

adb pull /data/local/tmp/intermediates intermediate-features

