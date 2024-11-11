# main.py

import argparse
import torch
from torch.utils.data import DataLoader

from data.data_loader import download_data, split_data, preprocess_data
from data.data_preprocessing import filter_and_save_dataset
from data.dataset import ReviewDataset
from src.features.feature_extraction import review2vector, get_markers
from src.train.train import train_model, evaluate_model, initialize_model
from utils.util import predict_and_print, predict_regression, predict_and_print_regression, predict_binary, \
    predict_and_print_binary, predict_categorical, predict_and_print_categorical


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a review classification model.")
    parser.add_argument(
        '--model_type',
        type=str,
        default='binary_classification',  # Default model type set here
        choices=['regression', 'binary_classification', 'multi_class_classification'],
        help="Type of model to train. Choose from: regression, binary_classification, multi_class_classification."
    )
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for DataLoader.")
    parser.add_argument('--n_hidden', type=int, default=10, help="Number of hidden units in the model.")
    parser.add_argument('--data_path', type=str, default='data/amazon-reviews.jsonl', help="Path to the dataset file.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Step 1: Download and preprocess data
    url = 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz'
    raw_data_path = 'data/amazon-reviews.jsonl.gz'
    raw_data_path_jsonl = 'data/Video_Games.jsonl'
    processed_data_path = 'data/amazon-reviews.tsv'
    # download_data(url, raw_data_path)
    # preprocess_data(raw_data_path_jsonl, processed_data_path)
    filtered_data_path = 'data/dataset.tsv'
    # filter_and_save_dataset(processed_data_path, filtered_data_path)
    # Split the data into train and test sets
    # split_data(filtered_data_path, 'data/train.tsv', 'data/test.tsv')

    # Step 2: Load data and create DataLoaders
    train_dataset = ReviewDataset('data/train.tsv', review2vector)
    test_dataset = ReviewDataset('data/test.tsv', review2vector)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Step 3: Initialize model and optimizer
    n_features = len(get_markers())
    model = initialize_model(args.model_type, n_features, args.n_hidden)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0)

    # Define adaptation function based on model type
    if args.model_type == "binary_classification":
        adapt_fn = lambda x, y : (x, (y > 2.5).float()) # Binary thresholding
    elif args.model_type == "multi_class_classification":
        adapt_fn = lambda x, y: (x, (y - 1).long())  # Adjust labels for multi-class
    else:
        adapt_fn = lambda x, y: (x, y.float())  # Regression

    # Assuming `model` is defined somewhere in the script
    loss_type = "categorical"  # Options: "regression", "binary", "categorical"

    # Choose the appropriate prediction function based on the loss type
    if loss_type == "regression":
        predict = predict_regression
        predict_and_print = predict_and_print_regression
    elif loss_type == "binary_classification":
        predict = predict_binary
        predict_and_print = predict_and_print_binary
    elif loss_type == "multi_class_classification":
        predict = predict_categorical
        predict_and_print = predict_and_print_categorical
    else:
        raise ValueError("Invalid loss_type. Choose 'regression', 'binary', or 'categorical'.")


    # Step 4: Train the model
    print("Starting training...")
    train_model(model, train_loader, optimizer, adapt_fn, epochs=args.epochs)

    # Example usage
    # 4-star review
    predict_and_print('very good product. excellent quality', 4, model)

    # 1-star review
    predict_and_print("It echoes so badly with my voice. Don't waste your time. I'm heavily considering returning it.",
                      1, model)

    # Step 5: Evaluate the model
    print("Evaluating on test data...")
    evaluate_model(model, test_loader, adapt_fn)


if __name__ == "__main__":
    main()
