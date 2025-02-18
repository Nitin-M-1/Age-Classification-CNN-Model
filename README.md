# Gender and Age Classification with CNNs

This project focuses on classifying gender and age from facial images using Convolutional Neural Networks (CNNs). The model builds upon previous architectures and enhances them through network depth adjustments and gender-specific age classification.

## Dataset
The project uses the [Adience Benchmark](http://www.openu.ac.il/home/hassner/Adience/data.html) dataset, which contains 26,580 images of 2,284 unique subjects. These images are labeled with gender and age ranges. The dataset was preprocessed to only include frontal face images, reducing the size to 17,523 images.

## Preprocessing
- **Cropping**: Randomly crops images to 227x227 pixels.
- **Mirroring**: Randomly mirrors images during training.
- **Centering**: For prediction, the network expects a cropped 227x227 image centered around the face.

## Model Overview

### Gender Classification:
1. 3x7x7 filter with 96 feature maps, followed by ReLU, Max-Pool, and LRN.
2. 96x28x28 filter with 256 feature maps, followed by ReLU, Max-Pool, and LRN.
3. 256x3x3 filter with stride 1 and padding 1, followed by ReLU and Max-Pool.
4. Fully connected layer with 512 neurons, followed by ReLU and Dropout (0.5).
5. Another fully connected layer with 512 neurons, followed by ReLU and Dropout (0.5).
6. Final layer maps to 2 gender classes: Male or Female.

### Age Classification (Gender-based):
- **Model used**: The same architecture as gender classification but with modifications:
  1. Increased dropout in the second fully connected layer to 0.7.
  2. Added weighted losses.
  3. Output layer maps to 8 age classes based on gender.

The models are trained using **Stochastic Gradient Descent** with 4-fold cross-validation.

## Running the Model

### 1. Gender Classification:
Execute the following to train and test the gender classification model:
```bash
python gender/train_n_test_gender.py
