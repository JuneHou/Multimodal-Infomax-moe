import os
import sys
import pickle
import numpy as np
from collections import defaultdict

def load_pickle(path):
    """Loads a pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)

class Config:
    """Configuration class for dataset paths."""
    def __init__(self, sdk_dir, dataset_dir):
        self.sdk_dir = sdk_dir
        self.dataset_dir = dataset_dir
        self.n_class = 1

        if not os.path.exists(self.dataset_dir):
            print(f"Error: Dataset directory {self.dataset_dir} does not exist. Please provide the correct path.")
            exit(1)

class MOSI:
    """Class to load MOSI dataset from pickle files."""
    def __init__(self, config, data_path):
        self.load_dataset(config, data_path)

    def load_dataset(self, config, data_path):
        """Loads train, validation, and test datasets from pickle files."""
        self.train = load_pickle(f"{data_path}/train.pkl")
        self.dev = load_pickle(f"{data_path}/dev.pkl")
        self.test = load_pickle(f"{data_path}/test.pkl")
        self.word2id = None  # Placeholder if needed

    def get_data(self, mode):
        """Returns the dataset split based on mode (train, valid, test)."""
        if mode == "train":
            return self.train, self.word2id, None
        elif mode == "dev":
            return self.dev, self.word2id, None
        elif mode == "test":
            return self.test, self.word2id, None
        else:
            print("Error: Mode should be one of ['train', 'dev', 'test']")
            exit(1)

class MOSEI(MOSI):
    """MOSEI dataset, inheriting from MOSI."""
    pass

def load_dataset(config, hp, mode):
    """
    Loads MOSI or MOSEI dataset based on `hp.dataset` value and returns (data, word2id, None).
    """
    dataset = hp.dataset.upper()
    data_dir = hp.dataset_path
    if "new_weights" not in data_dir:
        data_path = f"{data_dir}/{dataset}"
    else:
        data_path = data_dir
    if hp.dataset == "mosi" or "mosei":
        dataset = MOSI(config, data_path)
    else:
        print("Error: Dataset not properly defined. Choose 'mosi' or 'mosei'.")
        exit(1)

    return dataset.get_data(mode)
