import os
import csv
import numpy as np

def get_sentiment140_train_per_class(examples_per_class, examples_skip):
    """
    Get a balanced training set with a specified number of examples per class,
    skipping a certain number of examples for each class.
    
    Args:
    examples_per_class (int): Number of examples to include for each class
    examples_skip (int): Number of examples to skip for each class before including
    
    Returns:
    tuple: (texts, labels) where texts is a numpy array of tweet texts and
           labels is a numpy array of sentiment labels (0 or 1)
    """
    n_instances = 2 * examples_per_class  # 2 classes: negative (0) and positive (1)
    added = [0, 0]  # Track added examples for each class
    skipped = [0, 0]  # Track skipped examples for each class
    texts = np.zeros(n_instances, dtype=object)
    labels = np.zeros(n_instances, dtype=int)
    text_id = 0
    
    for label, text in read("training"):
        class_idx = 0 if label == 0 else 1  # Convert 0,4 labels to 0,1
        if skipped[class_idx] < (examples_skip * examples_per_class):
            skipped[class_idx] += 1
            continue
        if examples_per_class <= added[class_idx]:
            continue
        texts[text_id] = text
        labels[text_id] = class_idx
        text_id += 1
        added[class_idx] += 1
        if text_id == n_instances:
            break
    
    return texts, labels

def get_sentiment140_numpy(dataset, n_instances):
    """
    Get a specified number of examples from the dataset.
    
    Args:
    dataset (str): 'training' or 'testing'
    n_instances (int): Number of examples to retrieve
    
    Returns:
    tuple: (texts, labels) where texts is a numpy array of tweet texts and
           labels is a numpy array of sentiment labels (0 or 1)
    """
    texts = np.zeros(n_instances, dtype=object)
    labels = np.zeros(n_instances, dtype=int)
    text_id = 0
    
    for label, text in read(dataset):
        texts[text_id] = text
        # Convert 0,2,4 labels to -1,0,1
        if label == 0:
            labels[text_id] = -1
        elif label == 2:
            labels[text_id] = 0
        else:
            labels[text_id] = 1
        text_id += 1
        if text_id == n_instances:
            break
    
    return texts, labels

def get_sentiment140_train_numpy(n_instances=1600000):
    """
    Get training data from Sentiment140.
    
    Args:
    n_instances (int): Number of training examples to retrieve (max 1,600,000)
    
    Returns:
    tuple: (texts, labels) for training data
    """
    return get_sentiment140_numpy("training", n_instances)

def get_sentiment140_test_numpy(n_instances=498):
    """
    Get test data from Sentiment140.
    
    Args:
    n_instances (int): Number of test examples to retrieve (max 498)
    
    Returns:
    tuple: (texts, labels) for test data
    """
    return get_sentiment140_numpy("testing", n_instances)

def read(dataset="training"):
    """
    Python function for importing the Sentiment140 dataset. It returns an iterator
    of 2-tuples with the first element being the label (0 or 4) and the second
    element being the tweet text.
    
    Args:
    dataset (str): 'training' or 'testing'
    
    Yields:
    tuple: (label, text) for each example
    """
    if dataset == "training":
        fname = '../sentiment140/train.csv'
    elif dataset == "testing":
        fname = '../sentiment140/test.csv'
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    with open(fname, 'r', encoding='latin-1') as f:
        reader = csv.reader(f)
        for row in reader:
            # Sentiment140 format: 
            # 0 - sentiment (0 = negative, 2 = neutral, 4 = positive)
            # 5 - text of the tweet
            label = int(row[0])
            text = row[5]
            yield (label, text)