This repository contains a deep learning pipeline to train a Convolutional Neural Network (CNN) on the MNIST dataset for handwritten digit recognition, along with a real-time webcam demo to predict digits from user input. The setup includes PyTorch-based model training, custom dataset loading, and OpenCV-based visualization and interaction.

Features
Custom PyTorch CNN for MNIST, implemented from scratch.

Custom Dataset Loader for MNIST IDX files (no torchvision dependency for data).

Training and Validation Loop with accuracy/loss logging and early stopping.

Webcam MNIST Demo: Draw or show a digit to your webcam and the model predicts it in real-time using OpenCV

1. Model Definition
model-2.py constructs a basic CNN:

2 convolutional layers, each followed by ReLU and max-pooling

2 fully connected layers for digit classification (output: 10 classes)

Uses dropout for regularization

2. Training
train-2.py:

Provides a CustomMNISTDataset class to load MNIST IDX data directly

Splits data into train/validation/test sets

Trains the model with Adam optimizer and cross-entropy loss

Implements early stopping based on validation loss

Saves the best performing model as bestmodel.pth

3. Real-Time Inference
trackbar.py:

Loads bestmodel.pth and initializes the webcam

Processes each webcam frame: resizes, converts to grayscale and binary, normalizes, then predicts using the trained CNN

Displays the predicted digit on the video stream in real-time
