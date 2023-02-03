# importing sys
import sys
from PIL import Image

# add the path of the Assignment_1 folder to the sys.path
sys.path.append('Assignment_1')


from PreProcessor import ImagePreProcessor
from PreProcessor import FeatureVectorGenerator

from PreProcessor.ImagePreProcessor import show_image
from matplotlib import pyplot as plt
from Assignment_1.Utils.Data_Utils import get_train_data

import numpy as np
from matplotlib import pyplot as plt

dict = get_train_data()

image = dict[b'data'][1]
# ImagePreProcessor.show_image(image)

# ImagePreProcessor.show_image(ImagePreProcessor.rotate(image, np.pi/4))
# ImagePreProcessor.show_image(ImagePreProcessor.rotate(image, np.pi/2))
# ImagePreProcessor.show_image(ImagePreProcessor.rotate(image, 3*np.pi/4))
# ImagePreProcessor.show_image(ImagePreProcessor.rotate(image, np.pi))
# ImagePreProcessor.show_image(ImagePreProcessor.rotate(image, -np.pi/4))
# ImagePreProcessor.show_image(ImagePreProcessor.rotate(image, -np.pi/2))
# ImagePreProcessor.show_image(ImagePreProcessor.rotate(image, -3*np.pi/4))
# ImagePreProcessor.show_image(ImagePreProcessor.rotate(image, -np.pi))
im = image.reshape(3, 32, 32)
im = im.reshape(3* 32* 32)


im = FeatureVectorGenerator.resize(image)
print(im.shape)
print(im.dtype)
print(im[0][0][0])
im = im.reshape((3*224*224))
print(im.dtype)
print(im.shape)
print(im[0])
show_image(im, 224, 224, 3)