# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 23:02:42 2023

@author: diogo
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import math

# Function to load and sort images from a folder based on the "xxx" part of filenames
def load_and_sort_images_from_folder(folder_path):
    loaded_images = []
    for filename in sorted(os.listdir(folder_path), key=lambda x: int(x.split("_")[1].split(".")[0])):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                loaded_images.append(img)
                #print(f"Loaded: {filename}")
    return loaded_images

def create_shape_mask_from_grayscale_image(grayscale_image, threshold_value=128):
    # Threshold the grayscale image to create a binary mask
    _, mask = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Invert the mask
    inverted_mask = cv2.bitwise_not(mask)

    return inverted_mask

def apply_mask_to_image(image, mask):
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

###################################################################################################### 
#                                           Set Variables                                            #
######################################################################################################

# Specify the paths to the folders containing your images
path_placement = "placement_256"
path_routing = "routing_256_0"

# Specify the number of images to display after testing
num_samples_to_display = 10

###################################################################################################### 
#                                             Get Images                                             #
######################################################################################################
# Load and sort images from both folders and store them in their respective lists
images_placement = load_and_sort_images_from_folder(path_placement)
images_routing = load_and_sort_images_from_folder(path_routing)

###################################################################################################### 
#                                          Model Training                                            #
######################################################################################################

print('Splitting')
X_train, X_test, y_train, y_test = train_test_split(images_placement, images_routing, test_size=0.3)

# Define a simple CNN model
print('Training Model')
model = models.Sequential()
model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(256, 256, 1)))  # Input shape is (256, 256, 1) for grayscale
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(128, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256*256, activation='relu'))  # Adjust the output size for grayscale
model.add(layers.Reshape((256, 256, 1)))  # Reshape to the desired output shape (256, 256, 1)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32)

###################################################################################################### 
#                                           Model Testing                                            #
######################################################################################################

# Evaluate the model on the test set
loss = model.evaluate(np.array(X_test), np.array(y_test))
print(f'Test Loss: {loss}')

# Save the trained model
model.save('shape_cnn_model.h5')

results = model.predict(np.array(X_test))

###################################################################################################### 
#                                 Print Some Images For Comparison                                   #
######################################################################################################

# Display a few results alongside the ground truth images
for i in range(num_samples_to_display):
    input_image = X_test[i].squeeze()  # Remove the single channel dimension for display
    output_image = results[i].squeeze()
    ground_truth_image = y_test[i].squeeze()
    
    mask = create_shape_mask_from_grayscale_image(input_image)
    
    output_image_w_mask = apply_mask_to_image(output_image, mask)
    #ground_truth_image_w_mask = apply_mask_to_image(ground_truth_image, mask)

    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plt.imshow(input_image, cmap='gray')
    plt.title('Input Image')

    plt.subplot(132)
    plt.imshow(output_image, cmap='gray')
    plt.title('Model Output')

    #plt.subplot(132)
    #plt.imshow(output_image_w_mask, cmap='gray')
    #plt.title('Model Output + Mask')
    
    plt.subplot(133)
    plt.imshow(ground_truth_image, cmap='gray')
    plt.title('Ground Truth')

    plt.show()