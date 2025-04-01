import csv
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenize_sst3(texts, max_words=10000, max_len=50):
    """
    Tokenize each tweet for use in the TNN.
    """
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences

def get_sst3_train_per_class(examples_per_class, examples_skip):
    """
    Get a balanced training set with a specified number of examples per class,
    skipping a certain number of examples for each class.
    
    Args:
    examples_per_class (int): Number of examples to include for each class
    examples_skip (int): Number of examples to skip for each class before including
    
    Returns:
    tuple: (texts, labels) where texts is a numpy array of tweet texts and
           labels is a numpy array of sentiment labels (-1 or 0 or 1)
    """
    n_instances = 3 * examples_per_class  # 3 classes: -1, 0, 1
    added = [0, 0, 0]  # [-1, 0, 1]
    skipped = [0, 0, 0]
    raw_texts = []
    labels = []
    
    for label, text in read("training"):
        class_idx = label + 1  # Map -1->0, 0->1, 1->2
        if skipped[class_idx] < examples_skip * examples_per_class:
            skipped[class_idx] += 1
            continue
        if added[class_idx] < examples_per_class:
            raw_texts.append(text)
            labels.append(label)
            added[class_idx] += 1
        if sum(added) == n_instances:
            break
    
    tokenized_texts = tokenize_sst3(raw_texts)
    labels_array = np.array(labels)
    print("Tokenized texts shape:", tokenized_texts.shape)
    print("Labels:", labels_array)
    return tokenized_texts, labels_array

def get_sst3_numpy(dataset, n_instances):
    """
    Get a specified number of examples from the dataset.
    
    Args:
    dataset (str): 'training' or 'testing'
    n_instances (int): Number of examples to retrieve
    
    Returns:
    tuple: (texts, labels) where texts is a numpy array of tweet texts and
           labels is a numpy array of sentiment labels (-1, 0 or 1)
    """
    raw_texts = []
    labels = np.zeros(n_instances, dtype=int)
    text_id = 0
    for label, text in read(dataset):
        raw_texts.append(text)
        if label == 0:
            labels[text_id] = -1
        elif label == 1:
            labels[text_id] = 0
        else:
            labels[text_id] = 1
        text_id += 1
        if text_id == n_instances:
            break
    
    tokenized_texts = tokenize_sst3(raw_texts)  # (n_instances, 50)
    return tokenized_texts, labels

def get_sst3_train_numpy(n_instances=8000):
    """
    Get training data from SST3.
    
    Args:
    n_instances (int): Number of training examples to retrieve (max 8000)
    
    Returns:
    tuple: (texts, labels) for training data
    """
    return get_sst3_numpy("training", n_instances)

def get_sst3_test_numpy(n_instances=2000):
    """
    Get test data from SST3.
    
    Args:
    n_instances (int): Number of test examples to retrieve (max 2000)
    
    Returns:
    tuple: (texts, labels) for test data
    """
    return get_sst3_numpy("testing", n_instances)

def read(dataset="training"):
    """
    Python function for importing the SST3 dataset. It returns an iterator
    of 2-tuples with the first element being the label (0 or 4) and the second
    element being the tweet text.
    
    Args:
    dataset (str): 'training' or 'testing'
    
    Yields:
    tuple: (label, text) for each example
    """
    if dataset == "training":
        fname = '../sst3/train.csv'
    elif dataset == "testing":
        fname = '../sst3/test.csv'
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    with open(fname, 'r', encoding='latin-1') as f:
        reader = csv.reader(f)
        row_count = 0
        for row in reader:
            row_count += 1
            label = int(row[0])
            text = row[1]
            yield (label, text)
        print(f"Total rows processed: {row_count}")