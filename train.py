import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Parameters
img_height = 128
img_width = 128
batch_size = 32
num_classes = 2

# Data Preparation
def load_data(data_dir):
    images = []
    labels = []
    for label, subdir in enumerate(os.listdir(data_dir)):
        subdir_path = os.path.join(data_dir, subdir)
        for img_file in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, img_file)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load dataset
data_dir = "path/to/skin_cancer_dataset"
images, labels = load_data(data_dir)

# Normalize the data
images = images / 255.0

# Split the data
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Model Definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model Training
train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
history = model.fit(train_generator, epochs=20, validation_data=(X_val, y_val))

# Evaluation
y_pred = np.argmax(model.predict(X_val), axis=1)
print("Classification Report:")
print(classification_report(y_val, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# Save Model
model.save("skin_cancer_detection_model.h5")