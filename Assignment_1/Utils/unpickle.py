def unpickle(file):
    # Source code to unpickle the data from
    # https://www.cs.toronto.edu/~kriz/cifar.html
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict