# Arnold-BarkNet

Created a CNN to classify tree species based on images of the bark. It uses transfer learning through using the ResNet-50 architecture with ImageNet weights.
It was trained on over 1000 images for two species of trees which will soon increase as more images are collected. It achieves an accuracy of 93% on the validation
set. 

## Data Pipeline

Data_Labeller -> Dataset_Maker -> Image_Preprocessing

Data_Labeller automatically labels all of the images based on a text file which has the class name and the number of consecutive pictures of that species.

Dataset_Maker then splits the images into k-folds for training making sure each fold has an equal distribution of each tree.

Image_Preprocessing then performs random shifts across the vertical axis and random 224x224 pixel crops of the 2000x912 pixel image.

## BarkNet

BarkNet uses Nadam as the optimizer and the ResNet-50 Architecture with ImageNet weights. As usual with transfer learning the CNN feature maps are frozen, and
the dense layer is trained. I chose to train it over 15 epochs due to negligible increase with continuing epochs. The accuracy of the model for each epoch and
for each fold can be found in the results_graphs folder.

## Packages Needed
* NumPy == 1.19.2
* Pandas == 1.1.3
* TensorFlow == 2.3.0
* Python == 3.0 or above
* Matplotlib == 3.2.2

## Acknowledgements

Tree Species Identification from Bark Images Using Convolutional Neural Networks: https://arxiv.org/abs/1803.00949


