'''
This file contains functions to print information about the data
'''

def print_labels_info(labels):
    print("Keys:")
    print('labels:',labels.keys())
    print("_______________________________")
    print("labels:")
    print('label_names:',labels[b'label_names'])
    print('num_cases_per_batch:',labels[b'num_cases_per_batch'])
    print('num_vis:',labels[b'num_vis'])
    print("_______________________________")
    
def print_dict_info(dict):
    print("Keys:")
    print('dictionary:',dict.keys())
    print("_______________________________")
    print("type of data in dict:")
    print(type(dict[b'data']))
    print("shape:", dict[b'data'].shape)
    print("_______________________________")
    print("type of data[0] in dict:")
    print(type(dict[b'data'][0]))
    print("shape:", dict[b'data'][0].shape)
    print("_______________________________")
    print("type of labels in dict:")
    print(type(dict[b'labels']))
    print("size:", len(dict[b'labels']))
    print('First 10 labels',dict[b'labels'][0:10])
    print("_______________________________")
    print("type of batch_label in dict:")
    print(type(dict[b'batch_label']))
    print('batch_label',dict[b'batch_label'])
    print("_______________________________")
    print("type of filenames in dict:")
    print(type(dict[b'filenames'][0]))
    print("size:", len(dict[b'filenames']))
    print("_______________________________")