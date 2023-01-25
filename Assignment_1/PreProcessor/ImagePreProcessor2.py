import numpy as np
from matplotlib import pyplot as plt


def enhance(image):
    # finding the min and max from the array of image values
    min = np.min(image)
    max = np.max(image)

    # applying the enhancement operation and returning the np.array
    return ((image - min) * (255 / (max - min))).astype('uint8')

def posterize(image, min_value, max_value):
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Calculate the range
    range = max_value - min_value
    # Get the divider
    divider = 255/range

    # Get the level of colors
    levels = (image_array - min_value) * divider
    # Apply the color palette
    posterized_image = levels + min_value
    # Make sure the final image has pixel values in the range of [0, 255]
    posterized_image = np.clip(posterized_image, 0, 255).astype(np.uint8)

    return posterized_image

def random_rotate(image):

    print('shape:', image.shape)

    # randomly select theta with a step size of 45
    theta = np.random.choice(np.arange(-180, 181, 45))

    Transformation_matrix = np.array([[np.cos(theta), -np.sin(theta), 16*np.cos(theta)-16*np.sin(theta)-16],
                             [np.sin(theta), np.cos(theta), 16*np.cos(theta)+16*np.sin(theta)-16]])
    
    rotated_image = np.zeros((3072,1))

    n = 0
    for i in range(32):

        for j in range(32):
            X = np.matmul(Transformation_matrix, np.array([[i], [j], [1]]))
            p = int(X[0])
            q = int(X[1])

            if (p <= 31 and p >= 0) and (q<=31 and q >= 0):
                print("Pixel entered")
                
                for k in range(3):
                    rotated_image[p*32 + q + k*1024] = image[i*32 + j + k*1024]

    return rotated_image.astype('uint8')

def contrast(image):
    # randomly choosing the alpha value
    alpha = np.random.uniform(0.5, 2)

    # applying the contrast
    contrast_image = (alpha * (image - 128)) + 128

    return np.clip(contrast_image, 0, 255).astype(np.uint8)

def horizontal_flip(image):
    for i in range(0, 1024, 32):
        for j in range(16):
            for k in range(3):
                image[i+j + k*1024], image[i+abs(j-31) + k*1024] = image[i+abs(j-31) + k*1024], image[i+j + k*1024]

    return image


def contrast_and_flip(image):
    # changing the contrast of an image
    contrast_image = contrast(image)

    # randomly flipping the image
    flip = np.random.choice([True, False], p=[0.5, 0.5])
    if flip:
        return horizontal_flip(contrast_image)

    return contrast_image


def show_image(image):
    reshaped = np.zeros((32,32,3)).astype('uint8')

    # reshape the row majored array into 32*32*3 image
    n = 0
    for k in range(3):
        for i in range(32):
            for j in range(32):
                reshaped[i][j][k] = image[n]
                n = n+1

    plt.imshow(reshaped)
    plt.show()

def show_image_rehape(image):
    reshaped = image.reshape((32, 32, 3))

    plt.imshow(reshaped)
    plt.show()