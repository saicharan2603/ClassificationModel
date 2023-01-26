# importing sys
import sys
 
# add the path of the Assignment_1 folder to the sys.path
sys.path.append('Assignment_1')

import numpy as np
from Assignment_1.PreProcessor import ImagePreProcessor

def print_mat(array):
    array = np.array(array)
    array = array.reshape(3,32,32)

    for i in range(32):
        for j in range(32):
            print(f'{array[0][i][j]:4}', end = " ")
        print()

    print()


# test random_rotate function
if __name__ == "__main__":
    image = np.arange(3072)
    print_mat(image)

    print_mat(ImagePreProcessor.rotate(image, np.pi/4))
    print_mat(ImagePreProcessor.rotate(image, np.pi/2))
    print_mat(ImagePreProcessor.rotate(image, 3*np.pi/4))
    print_mat(ImagePreProcessor.rotate(image, np.pi))
    print_mat(ImagePreProcessor.rotate(image, -np.pi/4))
    print_mat(ImagePreProcessor.rotate(image, -np.pi/2))
    print_mat(ImagePreProcessor.rotate(image, -3*np.pi/4))
    print_mat(ImagePreProcessor.rotate(image, -np.pi))
