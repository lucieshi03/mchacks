import os
import cv2
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Image Preprocessing & Dataset Preparation
def preprocess_images(data_dir, img_size=(64, 64)):
    images = []
    labels = []
    
    for label in ["good", "bad"]:
        folder_path = os.path.join(data_dir, f"{label}_postures") # data_dir = current directory
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):
                # Read image
                img = cv2.imread(os.path.join(folder_path, filename))
                img = cv2.resize(img, img_size)  # Resize to uniform size
                img = img / 255.0  # Normalize the image
                
                images.append(img)
                labels.append(True if label == "bad" else False)  # Label 0 for "bad", 1 for "good"
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Prepare the data
data_dir = "."  # Current directory
images, labels = preprocess_images(data_dir)

# Convert labels to integers for compatibility with TensorFlow
labels = labels.astype(int)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
checkpoint = ModelCheckpoint("posture_model.h5", monitor="val_loss", save_best_only=True, mode="min")

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16, callbacks=[checkpoint])

# Save the trained model
model.save("posture_model.h5")
