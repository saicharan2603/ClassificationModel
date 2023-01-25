from unpickle import unpickle
import numpy as np
from matplotlib import pyplot as plt


def reshape(image):
    reshaped = np.zeros((32,32,3)).astype('uint8')
    n = 0
    for k in range(3):
        for i in range(32):
            for j in range(32):
                reshaped[i][j][k] = image[n]
                n = n+1

    return reshaped

labels = unpickle("Data\\input\\cifar-10-batches-py\\batches.meta")
dict = unpickle("Data\\input\\cifar-10-batches-py\\data_batch_1")

print(type(dict[b'data'][0]))
print(len(dict))

print(labels[b'label_names'])
print(dict.keys())
print(len(dict[b'labels']))
print(len(dict[b'data']))
print(len(dict[b'data'][0]))
print("_______________________________")
print("_______________________________")

import ImagePreProcessor

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





