# **Review Classifier**

A modular Python project for classifying reviews using machine learning models, supporting regression, binary
classification, and multi-class classification.

## **Table of Contents**

1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)
6. [Examples](#examples)
7. [Contributing](#contributing)
8. [License](#license)

---

## **Overview**

This project processes and classifies reviews (e.g., Amazon product reviews) using machine learning models. It includes
pipelines for downloading datasets, preprocessing text, training models, evaluating performance, and predicting
sentiments or ratings for individual reviews.

---

## **Features**

- Download raw datasets from a URL.
- Preprocess and filter datasets for training and evaluation.
- Train machine learning models for:
    - Regression
    - Binary classification
    - Multi-class classification
- Evaluate models' accuracy on test datasets.
- Predict sentiments or ratings for new reviews.

---

## **Project Structure**

```plaintext
ReviewClassifier/
│   .gitignore             
│   main.py                  
│   README.md               
│   requirements.txt         
│
├───data/                     
│   ├───raw/
│   │       amazon-reviews.jsonl.gz  
│   │       amazon-reviews.jsonl     
│   │
│   ├───processed/
│           amazon-reviews.tsv        
│           dataset.tsv              
│           train.tsv                
│           test.tsv                
│
├───models/                 
│       binary_classification_model.pth   
│       multi_class_classification_model.pth 
│       regression_model.pth              
│
├───notebooks/                # Reserved for exploratory data analysis (if needed)
│       
│
├───scripts/                  # Reserved for automation scripts
│       
│
├───src/                      # Core source code for the project
│   │   
│   │
│   ├───data/                 # Data handling modules
│   │       dataset.py                
│   │       data_loader.py            
│   │       data_preprocessing.py   
│   │      
│   │
│   ├───features/             # Feature extraction modules
│   │       feature_extraction.py   
│   │      
│   │
│   ├───models/               # Machine learning models
│   │       binary_classification_model.py   
│   │       multi_class_classification_model.py 
│   │       regression_model.py     
│   │       
│   │
│   ├───train/                # Training and evaluation logic
│   │       train.py             
│   │      
│   │
│   └───utils/                # Utility functions
│           util.py                
│         
│
└───tests/                    # Unit and integration tests for project modules
       

```

### Key Directories:

- **`data/`**: Raw and processed datasets.
- **`models/`**: Saved models after training.
- **`src/`**: Core functionality, including data handling, feature extraction, models, and utilities.
- **`tests/`**: Unit tests for the project.

---

## **Setup and Installation**

1. Clone the repository:
   ```bash
   git https://github.com/ashkanans/Review-Classifier/blob/main/main.py
   cd ReviewClassifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

Run the project using the `main.py` script with the `--action` argument to specify what you want to do. Valid actions
are:

- `download`: Download the dataset.
- `preprocess`: Preprocess and split the dataset.
- `train`: Train a machine learning model.
- `evaluate`: Evaluate the model's performance.
- `predict`: Predict the sentiment or rating of a review.

### General Syntax

```bash
python main.py --action <actions> [--model_type <model_type>] [other arguments...]
```

### Arguments:

- `--action`: Comma-separated list of actions (e.g., `download,train,evaluate`).
- `--model_type`: Type of model to use (`regression`, `binary_classification`, or `multi_class_classification`).
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size for training.
- `--n_hidden`: Number of hidden units in the model.
- `--data_path`: Path to the dataset file.
- `--train_path`: Path to the training dataset.
- `--test_path`: Path to the testing dataset.
- `--review`: Review text for prediction.
- `--stars`: Actual star rating (for comparison during prediction).

---

## **Examples**

### 1. Download the Dataset

```bash
python main.py --action download
```

### 2. Preprocess the Dataset

```bash
python main.py --action preprocess
```

### 3. Train a Binary Classification Model

```bash
python main.py --action train --model_type binary_classification --epochs 10 --batch_size 32
```

### 4. Evaluate the Model

```bash
python main.py --action evaluate --model_type binary_classification
```

### 5. Predict a Review's Sentiment

```bash
python main.py --action predict --model_type binary_classification --review "Fantastic product!" --stars 5
```

---

## **Contributing**

Contributions are welcome! If you have suggestions, bug fixes, or improvements, please submit a pull request.

### Steps to Contribute:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Commit changes and push to your fork.
4. Submit a pull request with details.

---

# **LICENSE**

Check out the license file.

--- 
