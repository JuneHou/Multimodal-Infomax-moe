import os
import sys
from create_dataset import *

# Configuration class to hold directory paths
class Config:
    def __init__(self, sdk_dir, dataset_dir):
        self.sdk_dir = '/data/wang/junh/githubs/CMU-MultimodalSDK'
        self.dataset_dir = '/data/wang/junh/githubs/Multimodal-Infomax/datasets/MOSEI'
        self.n_class = 7

# Check if the necessary directories exist and create them if not
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Main function to setup and process data
def main():
    # Path where the SDK and datasets are located
    sdk_path = '/data/wang/junh/githubs/CMU-MultimodalSDK'
    dataset_path = '/data/wang/junh/githubs/Multimodal-Infomax/datasets/MOSEI'
    
    # Ensure that dataset directory exists
    ensure_dir(dataset_path)
    
    # Create a configuration object
    config = Config(sdk_dir=sdk_path, dataset_dir=dataset_path)

    # dataset = MOSI(config)
    dataset = MOSEI(config)

    # Since the constructor is supposed to handle everything,
    # there is no need to call anything else.

if __name__ == "__main__":
    main()
