# -*- coding: utf-8 -*-
"""CNNModel.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RKWdB-EKl5x0mU7btI71jUSq8UzDXWHG
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

root_directory = "/content/drive/MyDrive/Projects/Dataset"
subdirectories = ["Dataset_Real", "Dataset_Fake"]
target_size = (224, 224)

data = []
labels = []

for label, subdirectory in enumerate(subdirectories):
    subdirectory_path = os.path.join(root_directory, subdirectory)

    if os.path.exists(subdirectory_path):
        folder_list = os.listdir(subdirectory_path)
        for folder_name in folder_list:
            folder_path = os.path.join(subdirectory_path, folder_name)
            if os.path.isdir(folder_path):
                image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]


                for image_filename in image_files:
                    image_path = os.path.join(folder_path, image_filename)
                    image = cv2.imread(image_path)

                    image = cv2.resize(image, target_size)

                    if image.shape[2] != 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    image = image.astype(np.float32) / 255.0

                    data.append(image)
                    labels.append(label)

    else:
        print(f"Subdirectory '{subdirectory}' does not exist in the root directory.")

# Convert data and labels lists to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Perform train-test split (adjust test_size as needed)
X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,  # You can adjust the test size as needed
    random_state=42  # Seed for reproducibility
)

# Print the shapes of the split datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
# Define directories
train_directory ="/content/drive/MyDrive/Projects/Dataset"

# Image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Create ImageDataGenerator for training data with data splitting
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

# Load and augment the training data
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

# Load and augment the validation data
validation_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    subset='validation'
)

# Build the CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,

)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
# Define directories
train_directory ="/content/drive/MyDrive/Projects/Dataset"

# Image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Create ImageDataGenerator for training data with data splitting
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

# Load and augment the training data
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

# Load and augment the validation data
validation_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    subset='validation'
)

# Build the CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,

)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
# Define directories
train_directory ="/content/drive/MyDrive/Projects/Dataset"

# Image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Create ImageDataGenerator for training data with data splitting
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

# Load and augment the training data
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    subset='training'
)

# Load and augment the validation data
validation_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    subset='validation'
)

# Build the CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

learning_rate = 3e-4
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,

)

from sklearn.metrics import classification_report

# Make predictions on validation data
validation_generator.reset()  # Reset generator to start from the beginning
predictions = model.predict(validation_generator, steps=validation_generator.samples // batch_size + 1)
predicted_classes = (predictions > 0.5).astype('int32')  # Binary classification threshold

# Get true labels
true_classes = validation_generator.classes

# Classification Report
report = classification_report(true_classes, predicted_classes, target_names=['Dataset_fake', 'Dataset_Real'])
print(report)

