# Client-side application

## Prerequisite

* `app/build.gradle:39`: `apply from: download.gradle` is commented out. Uncomment it for the first time you're running the code (to download pretrained model) and then comment it afterwards.

## Interface

* Spinner: contains list of demo images
* ImageView: displays selected image
* Button: triggers inference
* TextView: Displays result (possibly necessary log)

## MainActivity

1. `onCreate` initializes UI components as well as classifier via `createClassifier`.
2. When the button is triggered, `runInference` is called.
3. Results are displayed as soon as they're ready.

## Classifier

Most of the code is borrowed from official example of object classification in tensorflow-lite examples repository [here](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android).

* Class `Classifier` handles initialization, recognition, pre/postprocessing (and virtually everything)
* Resnet-specific values are extended in `ClassifierResNet`

## Misc.

* To add other demo images, add original image in `res/drawable` and modify `res/values/spinner.xml` accordingly.
* We use `tensorflow-lite:2.1.0` and `tensorflow-lite-gpu:2.1.0` at the moment, but it is possible to use legacy version.
* If there are memory issues while building, modify `org.gradle.jvmargs` in `gradle.properties` file to fit your environment.

## Screenshot

![ui](https://raw.githubusercontent.com/Mobile-and-Ubiquitous-Computing-2020-1/team1/client_backbone/client/ui.jpg)
