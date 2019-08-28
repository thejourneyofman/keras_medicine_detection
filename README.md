# Build a powerful image classification model using small dataset.

## What's Fine Tuning?
When constructing a model from scratch such as CNN, it's necessary to collect a large amount of sample images, and deep learning takes time. A method of transfer learning aims to build models in less time by using already learned models.

## Concept
In this implementation, given that full connected layeys of VGG16 are removed and a new FC layer is added, we only learn the 15th and its subsequent layers and all weights up to 14th layer will not be updated. This makes it possible to build a highly accurate model by inheriting the important features' extraction of VGG16 with a small data set as well as a shorter learning time. In other words, fine tuning performs efficiently by removing those deep layers and just reusing the shallow layer.

## Dataset and image augmentation
There are 905 train images in 10 classes and 23 test videos for object detection and classifications. When using CNN, there are cases where the accuracy can be improved by augmenting the training data using ImageDataGenerator in Keras' preprocessing.image module.

## Results
Validation_accuracy finally reached about 80%. Test of real object detection in videos is also highly performed at both accuracies and single/multiple detections.
Thanks to fine tuning, only hundreds of training images can achieve this accuracy in a short time.

## Demo
![demo](https://github.com/thejourneyofman/keras_medicine_detection/blob/master/images/medicine_detection.gif)

