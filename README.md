# Image-Recognition
Classify Dog or Cat image 
This repository contains the Jupyter Notebook for a 5-day bootcamp on image recognition. The project walks through the fundamentals of building Convolutional Neural Networks (CNNs), applying techniques like data augmentation and transfer learning, and evaluating model performance on various datasets.

Project Overview
This bootcamp is structured as a day-by-day learning path to build and train increasingly complex image recognition models. It starts with a basic CNN for digit recognition, moves to a more robust model for object recognition, and culminates in using transfer learning for a binary classification task.

Day 1: Setup and Data Exploration
Environment Setup: Installed the Kaggle library and configured the API for dataset access.

Dataset Loading: Loaded the MNIST (handwritten digits) and CIFAR-10 (objects) datasets using TensorFlow's built-in datasets module.

Data Preprocessing:

Normalized pixel values to a range of 0 to 1.

Reshaped the MNIST data to include a channel dimension, making it suitable for CNNs.

Data Visualization: Plotted several images from the MNIST training set to visualize the handwritten digits and their corresponding labels.

Day 2: Building a Basic CNN for MNIST
Model Architecture: Constructed a Sequential CNN model with:

Two Conv2D layers (with 32 and 64 filters).

MaxPooling2D layers for down-sampling.

A Flatten layer to prepare the data for the dense layers.

Two Dense layers, with a final softmax activation for 10-class classification.

Training: The model was trained on the MNIST dataset for 5 epochs, achieving a test accuracy of approximately 99.03%.

Evaluation: Plotted the training and validation accuracy over epochs to monitor performance.

Regularization: Introduced the concept of regularization by adding a Dropout layer to the model architecture to prevent overfitting.

Day 3: Data Augmentation and CIFAR-10 Classification
Data Augmentation: Used ImageDataGenerator to create augmented versions of the CIFAR-10 images (rotation, shifts, flips) to make the model more robust.

Model Enhancement: Built a deeper CNN for the more complex CIFAR-10 dataset, incorporating:

Three Conv2D layers.

BatchNormalization for stabilizing learning.

Dropout for regularization.

Training and Evaluation: Trained the model for 10 epochs using the augmented data generator. The model's performance was evaluated using a classification report and a confusion matrix, achieving an accuracy of around 69%.

Day 4: Transfer Learning for Cats vs. Dogs Classification
Dataset: Downloaded the "Cat and Dog" dataset from Kaggle.

Data Pipeline: Set up ImageDataGenerator to load images from directories, rescale them, and split them into training and validation sets.

Transfer Learning:

Utilized the pre-trained MobileNetV2 model with weights from ImageNet.

Froze the base model's layers to retain learned features.

Added custom top layers: GlobalAveragePooling2D and Dense layers for binary classification (cat or dog).

Fine-Tuning: After an initial training phase, the base model was unfrozen and fine-tuned with a very low learning rate (1e-5) for 3 additional epochs.

Model Persistence: The final trained model was saved to a file (mobilenet_cats_dogs.h5).

Evaluation: An ROC curve was plotted to evaluate the model's performance, achieving an AUC of 0.76.

Day 5: Model Deployment and Performance Comparison
Prediction: Loaded the saved model and used it to predict the class of a new, uploaded image (dog.jpg), correctly identifying it as "Dog".

Performance Summary: A bar chart was generated to compare the final validation accuracies of the models built for all three datasets:

MNIST: ~98%

CIFAR-10: ~75%

Cats vs. Dogs: ~90% (Note: This likely reflects an optimistic accuracy value used for the chart, as the training history showed a validation accuracy closer to 76% after fine-tuning).

Datasets Used
MNIST: A dataset of 70,000 grayscale images of handwritten digits (0-9).

CIFAR-10: A dataset of 60,000 color images across 10 different object classes.

Cat and Dog: A dataset of color images for binary classification of cats and dogs, sourced from Kaggle.

Technologies and Libraries
TensorFlow & Keras: For building, training, and evaluating deep learning models.

Kaggle API: For downloading datasets.

Scikit-learn: For generating evaluation metrics like confusion matrix, classification report, and ROC curve.

Matplotlib & Seaborn: For data visualization, including plotting training history, images, and performance charts.

NumPy: For numerical operations and data manipulation.

How to Run This Project
Environment: This notebook is designed to run in a Google Colab environment.

Kaggle API Key: To download the "Cat and Dog" dataset (Day 4), you will need a kaggle.json API key. Upload this file when prompted in the initial setup cells.

Execution: Run the notebook cells sequentially from top to bottom.

Prediction Image: On Day 5, you can upload your own image of a cat or dog to test the final model's prediction capabilities.
