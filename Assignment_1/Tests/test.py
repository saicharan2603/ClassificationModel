# importing sys
import sys

# add the path of the Assignment_1 folder to the sys.path
sys.path.append('Assignment_1')


from PreProcessor import ImagePreProcessor
from Utils.GetData import get_train_data

import numpy as np
from matplotlib import pyplot as plt

dict = get_train_data()

image = dict[b'data'][1]
ImagePreProcessor.show_image(image)

ImagePreProcessor.show_image(ImagePreProcessor.rotate(image, np.pi/4))
ImagePreProcessor.show_image(ImagePreProcessor.rotate(image, np.pi/2))
ImagePreProcessor.show_image(ImagePreProcessor.rotate(image, 3*np.pi/4))
ImagePreProcessor.show_image(ImagePreProcessor.rotate(image, np.pi))
ImagePreProcessor.show_image(ImagePreProcessor.rotate(image, -np.pi/4))
ImagePreProcessor.show_image(ImagePreProcessor.rotate(image, -np.pi/2))
ImagePreProcessor.show_image(ImagePreProcessor.rotate(image, -3*np.pi/4))
ImagePreProcessor.show_image(ImagePreProcessor.rotate(image, -np.pi))