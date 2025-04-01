import os
import csv
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenize_sentiment140(texts, max_words=10000, max_len=50):
    """
    Tokenize each tweet for use in the TNN.
    """
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    print("Shape of sequence", padded_sequences.shape)
    return padded_sequences

def get_sentiment140_train_per_class(examples_per_class, examples_skip):
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
    n_instances = 3 * examples_per_class  # 3 classes: negative (-1), neutral (0) and positive (1)
    added = [0, 0, 0]  # Track added examples for each class
    skipped = [0, 0, 0]  # Track skipped examples for each class
    raw_texts = []
    labels = []
    
    for label, text in read("training"):
        # Convert label from (0,2,4) to (-1,0,1)
        if label == 0:
            class_idx = 0  # -1 class
        elif label == 2:
            class_idx = 1  # 0 class
        else:  # label == 4
            class_idx = 2  # 1 class

        if skipped[class_idx] < examples_skip * examples_per_class:
            skipped[class_idx] += 1
            continue

        if added[class_idx] < examples_per_class:
            raw_texts.append(text)
            labels.append(class_idx - 1)  # Convert to -1,0,1
            added[class_idx] += 1
            print(f"Added class {class_idx}: {text[:50]}... (label: {class_idx - 1})")
        
        # Track if we've added enough instances
        if sum(added) == n_instances:
            print("Number added ------------")
            print(n_instances)
            print(added)
            break

    if sum(added) < n_instances:
        print(f"Warning: Only collected {sum(added)} samples instead of {n_instances}. Added per class:", added)

    # Tokenize text
    print("Raw texts collected:", len(raw_texts))
    tokenized_texts = tokenize_sentiment140(raw_texts)
    labels_array = np.array(labels)
    print("Tokenized texts shape:", tokenized_texts.shape)
    print("Labels:", labels_array)
    return tokenized_texts, labels_array

def get_sentiment140_numpy(dataset, n_instances):
    """
    Get a specified number of examples from the dataset.
    
    Args:
    dataset (str): 'training' or 'testing'
    n_instances (int): Number of examples to retrieve
    
    Returns:
    tuple: (texts, labels) where texts is a numpy array of tweet texts and
           labels is a numpy array of sentiment labels (-1, 0 or 1)
    """
    texts = np.zeros(n_instances, dtype=object)
    labels = np.zeros(n_instances, dtype=int)
    text_id = 0
    
    for label, text in read(dataset):
        texts[text_id] = tokenize_sentiment140(text)
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
        row_count = 0
        for row in reader:
            row_count += 1
            label = int(row[0])
            text = row[5]
            yield (label, text)
        print(f"Total rows processed: {row_count}")