#!/usr/bin/env bash

if [ ! -d "intermediate-features" ]
then
  mkdir intermediate-features
fi

adb shell "run-as com.example.client cat /data/data/com.example.client/files/intermediates > /data/local/tmp/intermediates"
adb pull /data/local/tmp/intermediates intermediate-features
