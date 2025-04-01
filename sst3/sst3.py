import csv
import os
from datasets import load_dataset

def download_sst_from_hf():
    """Download SST-5 dataset from Hugging Face."""
    print("Loading SST dataset from Hugging Face...")
    dataset = load_dataset("stanfordnlp/sst", "default")  # Sentiment scores (0.0-1.0)
    return dataset

def convert_to_sst3(dataset, output_file="sst3.csv"):
    """Convert SST-5 to SST-3 using sentiment scores."""
    print(f"Converting to SST-3 and saving to {output_file}")
    label_counts = {-1: 0, 0: 0, 1: 0}
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["split", "label", "sentence"])
        
        # Process train, validation, and test splits
        for split_name, split_data in dataset.items():
            split_id = {"train": 1, "validation": 3, "test": 2}[split_name]  # 1=train, 2=test, 3=dev
            for example in split_data:
                score = example["label"]  # Float from 0.0 to 1.0
                sentence = example["sentence"]
                if score < 0.4:
                    label = -1  # Negative
                elif 0.4 <= score <= 0.6:
                    label = 0   # Neutral
                else:
                    label = 1   # Positive
                writer.writerow([split_id, label, sentence])
                label_counts[label] += 1
    
    print("Label counts:", label_counts)

def split_sst3_into_files(input_file="sst3.csv", output_dir="sst3"):
    """Split SST-3 into separate train and test CSV files."""
    train_samples = []
    test_samples = []
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            split, label, sentence = row
            split = int(split)
            if split == 1:  # Train
                train_samples.append([label, sentence])
            elif split == 2:  # Test
                test_samples.append([label, sentence])
            # Dev (split=3) ignored or can be added to train/test
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(output_dir, "train.csv")
    print(f"Writing {len(train_samples)} samples to {train_file}")
    with open(train_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(train_samples)
    
    test_file = os.path.join(output_dir, "test.csv")
    print(f"Writing {len(test_samples)} samples to {test_file}")
    with open(test_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(test_samples)

if __name__ == "__main__":
    # Install datasets library if not already installed
    try:
        import datasets
    except ImportError:
        print("Installing datasets library...")
        os.system("pip install datasets")
    
    # Download SST-5 from Hugging Face
    dataset = download_sst_from_hf()
    
    # Convert to SST-3
    convert_to_sst3(dataset)
    
    # Split into train.csv and test.csv
    split_sst3_into_files()