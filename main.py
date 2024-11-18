import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.data.data_loader import download_data, preprocess_data, split_data, extract_gz_file
from src.data.data_preprocessing import filter_and_save_dataset
from src.data.dataset import ReviewDataset
from src.features.feature_extraction import review2vector, get_markers
from src.train.train import train_model, evaluate_model, initialize_model
from src.utils.util import predict_and_print_regression, predict_and_print_binary, predict_and_print_categorical


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a review classification model.")
    parser.add_argument(
        '--action',
        type=str,
        required=True,
        help="Comma-separated list of actions to perform: download, preprocess, train, evaluate, predict."
    )
    parser.add_argument('--model_type', type=str, default='binary_classification',
                        choices=['regression', 'binary_classification', 'multi_class_classification'],
                        help="Type of model to train. Choose from: regression, binary_classification, "
                             "multi_class_classification.")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for DataLoader.")
    parser.add_argument('--n_hidden', type=int, default=10, help="Number of hidden units in the model.")
    parser.add_argument('--data_path', type=str, default='data/raw/amazon-reviews.jsonl', help="Path to the dataset "
                                                                                               "file.")
    parser.add_argument('--train_path', type=str, default='data/processed/train.tsv', help="Path to the training "
                                                                                           "dataset.")
    parser.add_argument('--test_path', type=str, default='data/processed/test.tsv', help="Path to the testing dataset.")
    parser.add_argument('--review', type=str, default='', help="Review text for prediction.")
    parser.add_argument('--stars', type=int, default=0, help="Star rating for the review (for comparison).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Split the actions and validate them
    valid_actions = {'download', 'preprocess', 'train', 'evaluate', 'predict'}
    actions = args.action.split(',')

    for action in actions:
        if action not in valid_actions:
            print(f"Error: '{action}' is not a valid action. Valid actions are: {', '.join(valid_actions)}.")
            return

        if action == 'download':
            # Step 1: Download data
            url = 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz'
            raw_data_path = 'data/raw/amazon-reviews.jsonl.gz'
            extracted_data_path = 'data/raw/amazon-reviews.jsonl'

            download_data(url, raw_data_path)
            extract_gz_file(raw_data_path, extracted_data_path)

        elif action == 'preprocess':
            # Step 2: Preprocess data
            raw_data_path_jsonl = 'data/raw/amazon-reviews.jsonl'
            processed_data_path = 'data/processed/amazon-reviews.tsv'
            filtered_data_path = 'data/processed/dataset.tsv'
            preprocess_data(raw_data_path_jsonl, processed_data_path)
            filter_and_save_dataset(processed_data_path, filtered_data_path)
            split_data(filtered_data_path, args.train_path, args.test_path)
            print("Data preprocessing completed.")

        elif action == 'train':
            # Step 3: Train the model
            train_dataset = ReviewDataset(args.train_path, review2vector)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            n_features = len(get_markers())
            model = initialize_model(args.model_type, n_features, args.n_hidden)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0)

            # Define adaptation function
            if args.model_type == "binary_classification":
                adapt_fn = lambda x, y: (x, (y > 2.5).float())  # Binary thresholding
            elif args.model_type == "multi_class_classification":
                adapt_fn = lambda x, y: (x, (y - 1).long())  # Adjust labels for multi-class
            else:
                adapt_fn = lambda x, y: (x, y.float())  # Regression

            print("Starting training...")
            train_model(model, train_loader, optimizer, adapt_fn, epochs=args.epochs)
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/{args.model_type}_model.pth")
            print(f"Training completed. Model saved to models/{args.model_type}_model.pth")

        elif action == 'evaluate':
            # Step 4: Evaluate the model
            test_dataset = ReviewDataset(args.test_path, review2vector)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

            n_features = len(get_markers())
            model = initialize_model(args.model_type, n_features, args.n_hidden)
            model.load_state_dict(torch.load(f"models/{args.model_type}_model.pth", weights_only=True))

            # Define adaptation function
            if args.model_type == "binary_classification":
                adapt_fn = lambda x, y: (x, (y > 2.5).float())  # Binary thresholding
            elif args.model_type == "multi_class_classification":
                adapt_fn = lambda x, y: (x, (y - 1).long())  # Adjust labels for multi-class
            else:
                adapt_fn = lambda x, y: (x, y.float())  # Regression

            print("Evaluating on test data...")
            evaluate_model(model, test_loader, adapt_fn)

        elif action == 'predict':
            # Step 5: Predict a single review
            n_features = len(get_markers())
            model = initialize_model(args.model_type, n_features, args.n_hidden)
            model.load_state_dict(torch.load(f"models/{args.model_type}_model.pth"))

            if args.model_type == "binary_classification":
                predict_and_print = predict_and_print_binary
            elif args.model_type == "multi_class_classification":
                predict_and_print = predict_and_print_categorical
            else:
                predict_and_print = predict_and_print_regression

            print("Predicting review sentiment...")
            predict_and_print(args.review, args.stars, model)

        else:
            print(f"Unknown action: {action}")


if __name__ == "__main__":
    main()
