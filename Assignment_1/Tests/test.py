
# importing sys
import sys
 
# add the path of the Assignment_1 folder to the sys.path
sys.path.insert(0, 'D:/_STORAGE/03_EDUCATION/IIT Kanpur - Solid Mechanics and Design/OneDrive - IIT Kanpur/SEM 2/CS776A - Deep learning for Computer vision/CS776/Assignment_1')

from PreProcessor import ImagePreProcessor
from PreProcessor.unpickle import unpickle


import numpy as np
from matplotlib import pyplot as plt


labels = unpickle("Assignment_1\\Data\\batches.meta")
dict = unpickle("Assignment_1\\Data\\data_batch_1")

print(type(dict[b'data'][0]))
print(len(dict))

print(labels[b'label_names'])
print(dict.keys())
print(len(dict[b'labels']))
print(len(dict[b'data']))
print(len(dict[b'data'][0]))
print("_______________________________")
print("_______________________________")

image = dict[b'data'][1]
min = np.min(image)
max = np.max(image)
print("max:", max, "min:", min)
ImagePreProcessor.show_image(image)

ImagePreProcessor.show_image(ImagePreProcessor.random_rotate(image))

print(image)
enhanced = ImagePreProcessor.enhance(image)

print(enhanced)
min = np.min(enhanced)
max = np.max(enhanced)
print("max:", max, "min:", min)

print(enhanced.dtype)

ImagePreProcessor.show_image(ImagePreProcessor.contrast_and_flip(image))





