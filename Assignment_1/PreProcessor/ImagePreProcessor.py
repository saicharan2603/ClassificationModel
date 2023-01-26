'''
This file contains the functions for preprocessing the images
The images are preprocessed by applying the following operations:
    1. Enhancement
    2. Posterization
    3. Random Rotation
    4. Contrast and Horizontal Flip

This also contains the function for printing the images
and other useful functions for the four major operations

The input to the functions is the image array - Row Majored Array
The output of the functions is the preprocessed image array - Row Majored Array
'''

import numpy as np
from matplotlib import pyplot as plt

def enhance(image):
    # finding the min and max from the array of image values
    min = np.min(image)
    max = np.max(image)

    # applying the enhancement operation and returning the np.array
    return ((image - min) * (255 / (max - min))).astype('uint8')

def posterize(image):
    
    # choosing the min and max values from the image
    min_value, max_value = np.min(image), np.max(image)

    image = np.array(image)
    # Calculate the range
    range = max_value - min_value
    # Get the divider
    divider = 255/range
    # Get the level of colors
    levels = image / divider
    # Apply the color palette
    posterized_image = levels + min_value
    # Make sure the final image has pixel values in the range of [0, 255]
    posterized_image = np.clip(posterized_image, 0, 255).astype(np.uint8)

    return posterized_image

def random_rotate(image, x = 32, y = 32, channels = 3):
    # randomly select theta with a step size of 45
    return rotate(image, np.random.choice(np.arange(-np.pi, np.pi + np.pi/8, np.pi/4)), x, y, channels)

def rotate(image, theta, x = 32, y = 32, channels = 3):
    # defining the transformation matrix for rotating an image 
    Transformation_matrix = np.array([[np.cos(theta), -np.sin(theta), -(x//2)*np.cos(theta)+(y//2)*np.sin(theta)+(x//2)],
                                      [np.sin(theta),  np.cos(theta), -(y//2)*np.cos(theta)-(x//2)*np.sin(theta)+(y//2)]])
    
    # initializing the rotated image array
    rotated_image = np.zeros((x*y*channels))

    for i in range(x):
        for j in range(y):
            # finding the new coordinates of the pixel
            X = np.matmul(Transformation_matrix, np.array([[i], [j], [1]]))
            p = int(X[0])
            q = int(X[1])

            # checking if the new coordinates are within the image
            # if they are, then we map the pixel to the new array
            if (p <= (x-1) and p >= 0) and (q <= (y-1) and q >= 0):
                for k in range(3):
                    # performing the mapping of the pixels to new array
                    rotated_image[p*y + q + k*x*y] = image[i*y + j + k*x*y]

    return rotated_image.astype('uint8')

def contrast(image):
    # randomly choosing the alpha value
    alpha = np.random.uniform(0.5, 2)

    # applying the contrast
    contrast_image = (alpha * (image - 128)) + 128

    # making sure that the range of values is between 0 and 255
    return np.clip(contrast_image, 0, 255).astype(np.uint8)


def horizontal_flip(image, x = 32, y = 32, channels = 3):
    # flipping the image horizontally
    for i in range(0, x*y, y):
        for j in range(y//2):
            for k in range(channels):
                # swapping the pixels of 1st and last column
                image[i+j + k*x*y], image[i+abs(j-x+1) + k*x*y] = image[i+abs(j-x+1) + k*x*y], image[i+j + k*x*y]

    return image


def contrast_and_flip(image, x = 32, y = 32, channels = 3):
    # changing the contrast of an image
    contrast_image = contrast(image)

    # randomly flipping the image
    flip = np.random.choice([True, False], p=[0.5, 0.5])
    if flip:
        return horizontal_flip(contrast_image, x, y, channels)

    return contrast_image


def show_image(image, x = 32, y = 32, channels = 3):
    """
    reshapes the row majored image array into x*y*z image and displays it using matplotlib
    """
    reshaped = np.zeros((32,32,3)).astype('uint8')

    # reshape the row majored array into 32*32*3 image
    n = 0
    for k in range(channels):
        for i in range(x):
            for j in range(y):
                reshaped[i][j][k] = image[n]
                n = n+1

    # displaying the image using matplotlib
    plt.imshow(reshaped)
    plt.show()
