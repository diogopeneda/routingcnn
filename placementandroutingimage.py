# -*- coding: utf-8 -*-
"""
@author: diogo
"""

import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

# Function to create synthetic images
def create_image(scale, coordinates_1, sizes_1, coordinates_2, sizes_2, noise_stddev, padding=10, colour_1 = (0, 0, 255), colour_2 = (255, 0, 0)):

    image = np.zeros((scale, scale, 3), dtype=np.uint8)

    for i in range(len(coordinates_1) + len(coordinates_2)):
        
        if i < len(coordinates_1):
            x_topright = int(coordinates_1[i][0] * (scale - 2 * padding))  # Scale and convert to integer
            y_topright = int(coordinates_1[i][1] * (scale - 2 * padding))  # Scale and convert to integer
            width = int(sizes_1[i][0] * (scale - 2 * padding))          # Scale and convert to integer
            height = int(sizes_1[i][1] * (scale - 2 * padding))         # Scale and convert to integer

            # Calculate the top-left and bottom-right coordinates with padding
            x1 = x_topright + padding
            y1 = y_topright + padding
            x2 = x_topright + padding + width
            y2 = y_topright + padding + height
            
            cv2.rectangle(image, (x1, y1), (x2, y2), colour_1, -1)
        
        else:
            x_topright = int(coordinates_2[i-len(coordinates_1)][0] * (scale - 2 * padding))  # Scale and convert to integer
            y_topright = int(coordinates_2[i-len(coordinates_1)][1] * (scale - 2 * padding))  # Scale and convert to integer
            width = int(sizes_2[i-len(coordinates_1)][0] * (scale - 2 * padding))          # Scale and convert to integer
            height = int(sizes_2[i-len(coordinates_1)][1] * (scale - 2 * padding))         # Scale and convert to integer

            # Calculate the top-left and bottom-right coordinates with padding
            x1 = x_topright + padding
            y1 = y_topright + padding
            x2 = x_topright + padding + width
            y2 = y_topright + padding + height
            
            cv2.rectangle(image, (x1, y1), (x2, y2), colour_2, -1)
            
            
    # Create a binary mask of the shapes
    mask = image.copy()
    mask[mask > 0] = 1

    # Dilate the mask to add some padding
    kernel = np.ones((padding, padding), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel)

    # Generate noise and apply it to the areas covered by the dilated mask
    noise = np.random.normal(0, noise_stddev, image.shape).astype(np.uint8)
    noisy_image = image.copy()
    noisy_image[dilated_mask == 1] += noise[dilated_mask == 1]


    return noisy_image


def zoom_to_square(size, image, padding=10):
    # Find the coordinates of the non-black (non-zero) pixels in the image
    image_flat = np.mean(image, axis = 2)
    nonzero_coords = np.column_stack(np.where(image_flat > 0))

    if nonzero_coords.size > 0:
        # Calculate the minimum and maximum coordinates of the non-black pixels
        x_min, y_min = np.min(nonzero_coords, axis=0)
        x_max, y_max = np.max(nonzero_coords, axis=0)

        # Calculate the size of the square to fit around the region
        side_height = x_max - x_min
        side_length = y_max - y_min

        # Calculate the top-left and bottom-right coordinates for the square
        x1 = x_min
        y1 = y_min
        x2 = x_min + side_length
        y2 = y_min + side_length

        # Crop the region defined by the square
        cropped_image = image[x1:x2, y1:y2]

        # Resize the cropped image to the original size
        resized_image = cv2.resize(cropped_image, (size-(2*padding), size-(2*padding)))

        # Pad the final image with the specified padding
        if padding > 0:
            padded_image = cv2.copyMakeBorder(resized_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            return padded_image
        
        return resized_image

    return image

def center_to_square(size, image, padding=10):
    # Find the coordinates of the non-black (non-zero) pixels in the image
    image_flat = np.mean(image, axis = 2)
    nonzero_coords = np.column_stack(np.where(image_flat > 0))

    if nonzero_coords.size > 0:
        # Calculate the minimum and maximum coordinates of the non-black pixels
        x_min, y_min = np.min(nonzero_coords, axis=0)
        x_max, y_max = np.max(nonzero_coords, axis=0)

        # Calculate the size of the square to fit around the region
        side_height = x_max - x_min
        side_length = y_max - y_min

        # Calculate the top-left and bottom-right coordinates for the square
        x1 = x_min
        y1 = y_min
        x2 = x_min + side_height
        y2 = y_min + side_length

        # Crop the region defined by the square
        cropped_image = image[x1:x2, y1:y2]

        # Resize the cropped image to the original size
        #resized_image = cv2.resize(cropped_image, (1024-(2*padding), 1024-(2*padding)))

        # Pad the final image with the specified padding
        #if padding > 0:
        #    padded_image = cv2.copyMakeBorder(resized_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        #    return padded_image
        
        #return resized_image
        
        border_height = size-side_height-(2*padding)
        top_border = 0
        bottom_border = 0
        if border_height%2 == 0:
            top_border = int(border_height/2)
            bottom_border = int(border_height/2)
        else:
            top_border = (border_height//2)
            bottom_border = (border_height//2)+1
            
        border_length = size-side_length-(2*padding)
        left_border = 0
        right_border = 0
        if border_length%2 == 0:
            left_border = int(border_length/2)
            right_border = int(border_length/2)
        else:
            left_border = (border_length//2)
            right_border = (border_length//2)+1
            
        centered_image = cv2.copyMakeBorder(cropped_image, top_border, bottom_border, left_border, right_border, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        if padding > 0:
            padded_image = cv2.copyMakeBorder(centered_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            return padded_image
        
        return centered_image

    return image

def get_shapes(file, i):
    
    shape = pd.read_csv(file, nrows=1, usecols=[0, 1, 2, 3, 4])
    shape_line = pd.read_csv(file, nrows=1, header=None, skiprows=(i))

    n_columns = shape.columns.size
    n_rows = math.floor(shape_line.columns.size/n_columns)

    for current_row in list(range(0, n_rows)):
        row_data = np.array([(shape_line.iloc[0, n_columns*current_row], shape_line.iloc[0, n_columns*current_row+1], shape_line.iloc[0, n_columns*current_row+2], shape_line.iloc[0, n_columns*current_row+3], shape_line.iloc[0, n_columns*current_row+4])])    
        df_row_data = pd.DataFrame(row_data, columns=shape.columns)
        shape = pd.concat([shape, df_row_data], axis = 0)

    shape = shape.reset_index(drop=True)

    device_placement = pd.DataFrame(shape, copy=True)
    device_placement = device_placement.reset_index(drop=True)
    deleted_device = 0

    terminal_placement = pd.DataFrame(shape, copy=True)
    terminal_placement = terminal_placement.reset_index(drop=True)
    deleted_terminal = 0

    for current_row in list(range(0, math.floor(shape_line.columns.size/shape.columns.size))):
        width = float(shape['width'].iloc[current_row])
        heigth = float(shape['heigth'].iloc[current_row])
        if width == 0.0 and heigth == 0.0:
            device_placement.drop(device_placement.index[current_row-deleted_device], axis=0, inplace=True)
            deleted_device = deleted_device + 1
        else:
            terminal_placement.drop(terminal_placement.index[current_row-deleted_terminal], axis=0, inplace=True)
            deleted_terminal = deleted_terminal + 1

    device_coords = np.delete(device_placement, [0, 3, 4], 1)
    device_coords_float = device_coords.astype(float)
    device_coords_float = np.multiply(device_coords_float, 10000)
    device_sizes = np.delete(device_placement, [0, 1, 2], 1)
    device_sizes_float = device_sizes.astype(float)
    device_sizes_float = np.multiply(device_sizes_float, 10000)
    
    return device_coords_float, device_sizes_float

def get_routing(file, i):
    route = pd.read_csv(file, nrows=1, usecols=[0, 1, 2, 3, 4])
    route_line = pd.read_csv(file, nrows=1, header=None, skiprows=(i))

    n_columns = route.columns.size
    n_rows = math.floor(route_line.columns.size/n_columns)

    for current_row in list(range(0, n_rows)):
        row_data = np.array([(route_line.iloc[0, n_columns*current_row], route_line.iloc[0, n_columns*current_row+1], route_line.iloc[0, n_columns*current_row+2], route_line.iloc[0, n_columns*current_row+3], route_line.iloc[0, n_columns*current_row+4])])
        df_row_data = pd.DataFrame(row_data, columns=route.columns)
        route = pd.concat([route, df_row_data], axis = 0)
    
    route_coords = np.delete(route, [0, 3, 4], 1)
    route_coords_float =  route_coords.astype(float)
    route_coords_float = np.multiply( route_coords_float, 10000)
    route_sizes = np.delete(route, [0, 1, 2], 1)
    route_sizes_float =  route_sizes.astype(float)
    route_sizes_float = np.multiply( route_sizes_float, 10000)
    
    return route_coords_float, route_sizes_float

def append_vectors(vector1, vector2):
    # Check if both vectors have the same number of vectors
    if len(vector1) != len(vector2):
        print("Error: Both vectors must have the same number of vectors.")
        return None

    # Initialize an empty result vector of vectors
    result = []

    for v1, v2 in zip(vector1, vector2):
        if len(v1) != len(v2):
            print("Error: Vectors must have the same length for appending.")
            return None
        appended_vector = [x1 + x2 for x1, x2 in zip(v1, v2)]
        result.append(appended_vector)

    return result


###################################################################################################### 
#                                           Set Variables                                            #
######################################################################################################

#Number of samples
total_runs_with_zero = 419 

#Files Input
shapes_file = "routed-singlestageampPTG305-shapes0.csv"
routing_file = "routed-singlestageampPTG305-routing0.csv"

#Folders Output (Where images will be placed)
placement_and_routing_folder = "placement_and_route_256"
placement_folder = "placement_256"
routing_folder = "routing_256_0"

#Image Sizes
original_size = 1024 #Used for first image generation
final_size = 256 #Used for final image, after resize and centering

###################################################################################################### 
#                                           Get Values                                               #
######################################################################################################

print('Getting Devices')
device_float = [get_shapes(shapes_file, val) for val in range(1, total_runs_with_zero)]
print('Got Devices')
print('Getting Routing')
route_float = [get_routing(routing_file, val) for val in range(1, total_runs_with_zero)]
print('Got Routing')

###################################################################################################### 
#                                 Create Placement + Routing Images                                  #
######################################################################################################

#Get both Routing and Placement Images
os.makedirs(placement_and_routing_folder, exist_ok=True)

for i in range(total_runs_with_zero-1):

    image = create_image(original_size, device_float[i][0], device_float[i][1], route_float[i][0], route_float[i][1] , 0)
    
    image_zoomed = zoom_to_square(final_size, image, 10)
    
    image_fixed = center_to_square(final_size, image_zoomed, 10)

    filename = f"placement_and_route_{i:03d}.jpg"  # Format the filename as "image_xxx.jpg"
    output_path = os.path.join("placement_and_route_256", filename)
    # Save the image using OpenCV
    cv2.imwrite(output_path, image_fixed)
    
    #plt.imshow(cv2.cvtColor(image_fixed, cv2.COLOR_BGR2RGB))
    #plt.title('Image to Display')
    #plt.axis('off')
    #plt.show()
    
###################################################################################################### 
#                                      Create Placement Images                                       #
######################################################################################################

#Get Placement Images
os.makedirs(placement_folder, exist_ok=True)

for i in range(total_runs_with_zero-1):

    image = create_image(original_size, route_float[i][0], route_float[i][1], device_float[i][0], device_float[i][1] , 0, colour_1 = (0, 0, 0), colour_2 = (255, 255, 255))
    
    image_zoomed = zoom_to_square(final_size, image, 20)
    
    image_fixed = center_to_square(final_size, image_zoomed, 20)

    filename = f"placement_{i:03d}.jpg"  # Format the filename as "image_xxx.jpg"
    output_path = os.path.join("placement_256", filename)
    # Save the image using OpenCV
    cv2.imwrite(output_path, image_fixed)
    
    #plt.imshow(cv2.cvtColor(image_fixed, cv2.COLOR_BGR2RGB))
    #plt.title('Image to Display')
    #plt.axis('off')
    #plt.show()

###################################################################################################### 
#                                       Create Routing Images                                        #
######################################################################################################

#Get Routing Images
os.makedirs(routing_folder, exist_ok=True)

for i in range(total_runs_with_zero-1):
    
    image = create_image(original_size, device_float[i][0], device_float[i][1], route_float[i][0], route_float[i][1] , 0, colour_1 = (0, 0, 0), colour_2 = (255, 255, 255))
    
    image_zoomed = zoom_to_square(final_size, image, 20)
    
    image_fixed = center_to_square(final_size, image_zoomed, 20)

    filename = f"route_{i:03d}.jpg"  # Format the filename as "image_xxx.jpg"
    output_path = os.path.join("routing_256_0", filename)
    # Save the image using OpenCV
    cv2.imwrite(output_path, image_fixed)
    
    #plt.imshow(cv2.cvtColor(image_fixed, cv2.COLOR_BGR2RGB))
    #plt.title('Image to Display')
    #plt.axis('off')
    #plt.show()

###################################################################################################### 
#                                               Extra                                                #
######################################################################################################

#pos = device_float[0][0].append(route_float[0][0])
#size = device_float[0][1].append(route_float[0][1])

#pos = np.concatenate((device_float[0][0], route_float[0][0]))
#size = np.concatenate((device_float[0][1], route_float[0][1]))

#combine = np.concatenate((pos, size), axis = 1)
#test = tuple(zip(pos, size))
#combined_result = append_vectors(device, route)

#print('Creating Image')
#image = create_image(pos, size, 0, colour = 111)
#image = [create_image(coords, size, 0, colour = 111) for coords, size in tuple(zip(pos, size))]


#print('Treating Images')
#image_fixed = [zoom_to_square(img, 15) for img in image]

#plt.imshow(cv2.cvtColor(image_fixed, cv2.COLOR_BGR2RGB))
#plt.title('Image to Display')
#plt.axis('off')
#plt.show()




























