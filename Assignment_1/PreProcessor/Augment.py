from PreProcessor import ImagePreProcessor
import numpy as np

def random_pre_process(image):
    rand = np.random.randint(0, 4)

    if rand == 0:
        return ImagePreProcessor.enhance(image)
    elif rand == 1:
        return ImagePreProcessor.posterize(image)
    elif rand == 2:
        return ImagePreProcessor.random_rotate(image)
    elif rand == 3:
        return ImagePreProcessor.contrast_and_flip(image)


def get_Augmented_Data(data: dict):

    data2 = data.copy()
    size = len(data[b'data'])

    for i in range(size): 
        data2[b'data'][i] = random_pre_process(data2[b'data'][i])
    
    data2[b'data'] = np.vstack((data[b'data'], data2[b'data']))
    data2[b'labels'] = data[b'labels'] + data2[b'labels']

    return data2