# Build a powerful image classification model using small dataset.

## Demo
![demo](https://github.com/thejourneyofman/keras_medicine_detection/blob/master/images/medicine_detection.gif)

## What's Fine Tuning?
When constructing a model from scratch such as CNN, it's necessary to collect a large amount of sample images, and deep learning takes time. A method of transfer learning also known as fine tuning aims to build models in less time by using already learned models.

## Concept
In this implementation, given that full connected layeys of VGG16 are removed and a new FC layer is added, we only learn the 15th and its subsequent layers and all weights up to 14th layer will not be updated. This makes it possible to build a highly accurate model by inheriting the important features' extraction of VGG16 with a small data set as well as a shorter learning time. In other words, fine tuning performs efficiently by removing those deep layers while just reusing the shallow layers.

## Dataset and image augmentation
There are 905 train images in 10 classes and 23 test videos for object detection and classifications. When using CNN, there are cases where the accuracy can be improved by augmenting the training data using ImageDataGenerator in Keras' preprocessing.image module.

## Results
Validation accuracy finally reached above 95%. Test of real object detection in videos is also highly performed at both accuracies and single/multiple detections.
Thanks to fine tuning, hundreds of training images can achieve this accuracy in a short time.

## Bonus
Remove the hand and shadows in the motion video improved the accuracies of detection.

## Bonus to Bonus
Is there an efficient way to predict an object by "Shadow" as for early indication then go to a more accurate one? Imagine a moonnight you see a long shadow approaching and hold your fingers crossed...
