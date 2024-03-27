"""
This script demonstrates multiclass classification using a convolutional neural network (CNN) in Keras.
The script performs the following steps:
1. Imports necessary libraries for data processing, model building, and evaluation.
2. Defines the paths to the directories containing the images.
3. Defines the image dimensions, batch size, number of epochs, target size, classes, and binary classes.
4. Creates an ImageDataGenerator object for data augmentation and normalization.
5. Defines the training and testing data generators using flow_from_directory() method.
6. Builds and trains a binary classification model using a CNN architecture.
7. Builds and trains a multiclass classification model using a CNN architecture.
8. Evaluates the binary and multiclass models on the validation set.
9. Plots the training and validation accuracy and loss curves for both models.
10. Generates confusion matrices for binary and multiclass classification.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define the paths to the directories containing the images
original_dir = r"c:\Users\Saifk\Desktop\Original"
segmented_dir = r"c:\Users\Saifk\Desktop\Segmented"

# Define the image dimensions
img_width, img_height = 150, 150

# Define the batch size
batch_size = 32

# Define the number of epochs
epochs = 10 # Increased number of epochs

# Define the target size of the images
target_size = (img_width, img_height)

# Define the classes
classes = ['Benign', 'Early', 'Pre', 'Pro']

# Define the binary classes
binary_classes = ['Benign', 'Pro']

# ImageDataGenerator is a class in Keras that helps in data augmentation by generating 
# tensor image data with real-time data augmentation. The data will be looped over in batches.
# Here, we are rescaling the images by 1./255 to normalize the pixel values (which are 0-255 for RGB images).
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# flow_from_directory() is a method that allows you to read images from a big numpy array and folders containing images.
# It automates the process of getting images ready for model training.
# Here, we are defining the training set. The images will be read from the 'original_dir' directory,
# resized to the 'target_size', and fed into the model in batches of size 'batch_size'.
# The 'classes' parameter is set to 'binary_classes', indicating that we are dealing with a binary classification problem.
# The 'class_mode' is set to 'binary', which means that the labels will be binary labels.
train_generator = train_datagen.flow_from_directory(
    original_dir,
    target_size=target_size,
    batch_size=batch_size,
    classes=binary_classes,
    class_mode='binary')

# Here, we are defining the testing set in a similar way to the training set.
# The only differences are that the images are read from the 'segmented_dir' directory,
# and the 'classes' parameter is set to 'classes', indicating that we are dealing with a multiclass classification problem.
# The 'class_mode' is set to 'categorical', which means that the labels will be one-hot encoded.
test_generator = test_datagen.flow_from_directory(
    segmented_dir,
    target_size=target_size,
    batch_size=batch_size,
    classes=classes,
    class_mode='categorical')

# Define the binary classification model
# Maxpooling 2D is used to define the maximum value in each window of a feature map produced by a convolutional layer.
# conv2D is used to extract features from the input image by sliding a convolution filter over the input image.
# we add the flatten layer to convert the final feature maps into a one single 1D vector or array.
# in the end it is applied into the Dense layer which is a fully connected 512 neurons 
# sigmoid is used to output values between 0 and 1 
# the 0 and 1 values are then used on benign and malignant images
binary_model = Sequential()
binary_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
binary_model.add(MaxPooling2D((2, 2)))
binary_model.add(Conv2D(64, (3, 3), activation='relu'))
binary_model.add(MaxPooling2D((2, 2)))
binary_model.add(Conv2D(128, (3, 3), activation='relu'))
binary_model.add(MaxPooling2D((2, 2)))
binary_model.add(Conv2D(128, (3, 3), activation='relu'))
binary_model.add(MaxPooling2D((2, 2)))
binary_model.add(Flatten())
binary_model.add(Dense(512, activation='relu'))
binary_model.add(Dense(1, activation='sigmoid'))
# Compile the binary classification model
binary_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])

# Train the binary classification model
binary_model.fit(train_generator, epochs=epochs, validation_data=train_generator)  # Added validation_data argument

# Define the multiclass classification model

# 
# we use max-pooling to reduce the dimensionality of the feature maps
# the softmax function allows us to interpret the outputs as probabilities
# these probalities are used to classify the input images into 1 of the 4 classes
# categorical cross entropy is used to calculate the loss between the predicted and actual class
#
multiclass_model = Sequential()
multiclass_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
multiclass_model.add(MaxPooling2D((2, 2)))
multiclass_model.add(Conv2D(64, (3, 3), activation='relu'))
multiclass_model.add(MaxPooling2D((2, 2)))
multiclass_model.add(Conv2D(128, (3, 3), activation='relu'))
multiclass_model.add(MaxPooling2D((2, 2)))
multiclass_model.add(Conv2D(128, (3, 3), activation='relu'))
multiclass_model.add(MaxPooling2D((2, 2)))
multiclass_model.add(Flatten())
multiclass_model.add(Dense(512, activation='relu'))
multiclass_model.add(Dense(len(classes), activation='softmax'))
# Compile the multiclass classification model
multiclass_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])

# Train the multiclass classification model
multiclass_model.fit(test_generator, epochs=epochs, validation_data=test_generator)  # Added validation_data argument

# Evaluate the binary classification model on the validation set
binary_loss, binary_accuracy = binary_model.evaluate_generator(train_generator)
print(f'Binary Classification Model - Loss: {binary_loss:.2f}, Accuracy: {binary_accuracy:.2f}')

# Evaluate the multiclass classification model on the validation set
multiclass_loss, multiclass_accuracy = multiclass_model.evaluate_generator(test_generator)
print(f'Multiclass Classification Model - Loss: {multiclass_loss:.2f}, Accuracy: {multiclass_accuracy:.2f}')

# Train the models and store the history
history_binary = binary_model.fit(train_generator, epochs=epochs, validation_data=train_generator)  # Added validation_data argument
history_multiclass = multiclass_model.fit(test_generator, epochs=epochs, validation_data=test_generator)  # Added validation_data argument

# Plot training & validation accuracy values for binary classification model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_binary.history['accuracy'])
plt.plot(history_binary.history['val_accuracy'])  # Added this line for validation accuracy
plt.title('Binary Classification Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values for binary classification model
plt.subplot(1, 2, 2)
plt.plot(history_binary.history['loss'])
plt.plot(history_binary.history['val_loss'])  # Added this line for validation loss
plt.title('Binary Classification Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation accuracy values for multiclass classification model
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history_multiclass.history['accuracy'])
plt.plot(history_multiclass.history['val_accuracy'])  # Added this line for validation accuracy
plt.title('Multiclass Classification Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values for multiclass classification model
plt.subplot(1, 2, 2)
plt.plot(history_multiclass.history['loss'])
plt.plot(history_multiclass.history['val_loss'])  # Added this line for validation loss
plt.title('Multiclass Classification Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

binary_predictions = binary_model.predict_generator(train_generator)
binary_predictions = (binary_predictions > 0.5).astype(int)
binary_true_labels = train_generator.classes

binary_cm = confusion_matrix(binary_true_labels, binary_predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(binary_cm, annot=True, fmt='d', cmap='Blues', xticklabels=binary_classes, yticklabels=binary_classes)
plt.title('Binary Classification Confusion Matrix')
plt.ylabel('Predicted')
plt.xlabel('True')
plt.show()

multiclass_predictions = multiclass_model.predict_generator(test_generator)
multiclass_predictions = np.argmax(multiclass_predictions, axis=1)
multiclass_true_labels = test_generator.classes


multiclass_cm = confusion_matrix(multiclass_true_labels, multiclass_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(multiclass_cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Multiclass Classification Confusion Matrix')
plt.ylabel('Predicted')
plt.xlabel('True')
plt.show()


