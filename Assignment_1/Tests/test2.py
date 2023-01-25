# from unpickle import unpickle
import numpy as np
# from matplotlib import pyplot as plt


# labels = unpickle("Data\\input\\cifar-10-batches-py\\batches.meta")
# dict = unpickle("Data\\input\\cifar-10-batches-py\\data_batch_1")

# # print(type(dict[b'data'][0]))

# print(labels[b'label_names'])
# print(dict.keys())
# # print(len(dict[b'labels']))
# # print(len(dict[b'data']))
# # print(len(dict[b'data'][0]))

# import ImagePreProcessor

# image = dict[b'data'][1]
# reshapemohan=image.reshape(3072,1)
# print (image)
# print(type(image))
# print(type(dict[b'data']))
# print(reshapemohan.shape)
# # min = np.min(image)
# # max = np.max(image)
# # print("max:", max, "min:", min)
# # ImagePreProcessor.show_image(image)

# # ImagePreProcessor.show_image(ImagePreProcessor.random_rotate(image))

# # print(image)
# # enhanced = ImagePreProcessor.enhance(image)

# # print(enhanced)
# # min = np.min(enhanced)
# # max = np.max(enhanced)
# # print("max:", max, "min:", min)

# # print(enhanced.dtype)

# # ImagePreProcessor.show_image(ImagePreProcessor.contrast_and_flip(image))

mohandict={"assign":np.array([1,2,3])}
mohandict["assign"] = np.append(mohandict["assign"], np.array([2,3,4]))
print(mohandict)



