"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py which is GPL licensed.

This code reads the mnist dataset and creates appropriate training sets for each experiment.
"""
import os
import struct
import numpy as np

def get_mnist_train_per_class(examples_per_class, examples_skip):
    n_instances = 10*examples_per_class
    added = [0 for _ in range(10)]
    skipped = [0 for _ in range(10)]
    images = np.zeros((n_instances,28*28), dtype=np.float32)
    labels = np.zeros(n_instances, dtype=int)
    img_id = 0
    for label, pixels in read("training"):
        if skipped[label] < (examples_skip * examples_per_class):
            # we haven't skipped enough examples yet
            skipped[label] += 1
            continue
        if examples_per_class <= added[label]:
            # we already have enough examples of this class
            continue
        images[img_id,:] = pixels.flatten().astype(np.float32)
        labels[img_id] = label
        img_id += 1
        added[label] += 1
        if img_id == n_instances:
            break
    return images, labels


def get_mnist_numpy(dataset, n_instances):
    images = np.zeros((n_instances,28*28), dtype=np.float32)
    labels = np.zeros(n_instances, dtype=int)
    img_id = 0
    for label, pixels in read(dataset):
        images[img_id,:] = pixels.flatten().astype(np.float32)
        labels[img_id] = label
        img_id += 1
        if img_id == n_instances:
            break
    return images, labels

def get_mnist_train_numpy(n_instances = 60000):
    return get_mnist_numpy("training", n_instances)

def get_mnist_test_numpy(n_instances = 10000):
    return get_mnist_numpy("testing", n_instances)

def read(dataset = "training"):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = '../mnist/train-images.idx3-ubyte'
        fname_lbl = '../mnist/train-labels.idx1-ubyte'
    elif dataset is "testing":
        fname_img = '../mnist/t10k-images.idx3-ubyte'
        fname_lbl = '../mnist/t10k-labels.idx1-ubyte'
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)
        img = img.astype(int)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)
