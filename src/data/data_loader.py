# src/data/data_loader.py
import gzip
import json
import os
import shutil

import requests
from tqdm import tqdm


def extract_gz_file(input_file, output_file):
    """Extracts a .gz file to the specified output file."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist!")

    with gzip.open(input_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"File extracted to {output_file}")
    print("Data extracted successfully.")


def download_data(url: str, save_path: str):
    """Downloads the dataset from a URL and saves it to the specified path."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading Data")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    print("Data downloaded successfully.")

def preprocess_data(input_path: str, output_path: str):
    """Converts JSONL data to TSV format with selected fields."""
    # First, count the total number of lines for progress bar initialization
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Now process the file line by line with a progress bar
    with open(input_path, 'r', encoding='utf-8') as f, open(output_path, 'w', encoding='utf-8') as w:
        progress_bar = tqdm(total=total_lines, desc="Processing Data")

        for line in f:
            data = json.loads(line)
            rating = data.get('rating')
            review = data.get('text')
            w.write(f"{rating}\t{review}\n")
            progress_bar.update(1)

        progress_bar.close()
    print(f"Preprocessing completed. JSONL file ({input_path}) has been converted to TSV format: {output_path}")


def split_data(dataset_path: str, train_path: str, test_path: str, train_size: int = 1000):
    """Splits data into training and testing sets."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(train_path, 'w', encoding='utf-8') as train_file, open(test_path, 'w', encoding='utf-8') as test_file:
        train_file.writelines(lines[:train_size])
        test_file.writelines(lines[train_size:])
