# Handwritten Character Recognition using CNN

## Overview

This repository contains a Convolutional Neural Network (CNN) deep learning pipeline developed for Task 3: Handwritten Character Recognition, part of the CodeAlpha Machine Learning Internship.

The objective is to accurately identify and classify handwritten digits (0–9) from 28x28 pixel grayscale images using spatial feature extraction and deep learning.

---

## Dataset Details

The model uses the classic MNIST Handwritten Digit Dataset:

- **Training Set:** 60,000 grayscale images (28x28 pixels)
- **Test Set:** 10,000 grayscale images (28x28 pixels)
- **Target Classes:** Digits `0` through `9` (10 classes)

---

## Model Architecture & Features

The pipeline reshapes raw pixel values into 3D tensors (`28x28x1`) normalized to the range `[0, 1]`, then passes them through the following layers:

1. **Conv2D (32 filters, 3x3 kernel, ReLU):** Captures low-level spatial features such as edges and curves
2. **MaxPooling2D (2x2):** Downsamples spatial dimensions while preserving key features
3. **Conv2D (64 filters, 3x3 kernel, ReLU):** Extracts higher-level structural patterns
4. **MaxPooling2D (2x2):** Further reduces spatial complexity
5. **Flatten:** Unrolls feature maps into a 1D vector
6. **Dense (128 units, ReLU):** Fully connected feature integration
7. **Dropout (0.4):** Prevents co-adaptation and reduces overfitting
8. **Dense Output (10 units, Softmax):** Outputs a class probability distribution

The model is compiled with the Adam optimizer and categorical cross-entropy loss, trained for 10 epochs with a batch size of 64 and a 10% validation split.

---

## Performance Summary

- **Test Accuracy:** 99.21%
- **Test Loss:** 0.0254
- **Precision / Recall / F1-Score:** 0.99 across all digit classes

---

## Project Structure

```text
CodeAlpha_Handwritten_Character_Recognition/
├── handwritten_recognition.ipynb   # Main Jupyter Notebook
└── README.md                       # Project documentation
```

The MNIST dataset is downloaded automatically by Keras on first run, so no local dataset files are required.

---

## Installation & Running the Code

### Prerequisites

Ensure TensorFlow and the core evaluation dependencies are installed:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

### Execution

Open `handwritten_recognition.ipynb` in VS Code or Jupyter and run the cells top to bottom.

The notebook loads and preprocesses MNIST, builds and trains the CNN, then prints the evaluation report alongside a confusion matrix plot.

---

## Author & Acknowledgments

- **Developer:** Wireko Fosu Eric
- **Program:** Machine Learning Internship at CodeAlpha
