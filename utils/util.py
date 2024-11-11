# Regression
from src.features.feature_extraction import review2vector


def predict_regression(model, review: str):
    x = review2vector(review)
    batched_x = x.unsqueeze(0)
    batch_out = model(batched_x)

    # Directly return the predicted score for regression
    return batch_out['pred'][0].item()


def predict_and_print_regression(review: str, stars: int, model):
    print(f'=' * 125)
    print(f'# review: {review}')
    print(f'# gold stars: {stars}')
    score = predict_regression(model, review)
    print(f'# predicted score: {score} -> {round(score)}')
    print(f'=' * 125)


# Binary Cross-Entropy
def predict_binary(model, review: str):
    x = review2vector(review)
    batched_x = x.unsqueeze(0)
    batch_out = model(batched_x)

    # Sigmoid output for binary classification
    pred_prob = torch.sigmoid(batch_out['pred'][0]).item()

    # Convert probability to binary output (1 for positive, 0 for negative)
    return 1 if pred_prob > 0.5 else 0


def predict_and_print_binary(review: str, stars: int, model):
    print(f'=' * 125)
    print(f'# review: {review}')
    print(f'# gold stars: {stars} -> {"Positive" if stars > 2.5 else "Negative"}')
    score = predict_binary(model, review)
    print(f'# predicted score: {score} -> {"Positive" if score == 1 else "Negative"}')
    print(f'=' * 125)


# Categorical Cross-Entropy
def predict_categorical(model, review: str):
    x = review2vector(review)
    batched_x = x.unsqueeze(0)
    batch_out = model(batched_x)

    # Softmax for multi-class classification
    pred_probs = batch_out['pred'][0]
    pred = pred_probs.argmax()  # Index of the highest probability

    # Convert the predicted class index to a star rating (assuming 1-5 stars)
    return pred.item() + 1


def predict_and_print_categorical(review: str, stars: int, model):
    print(f'=' * 125)
    print(f'# review: {review}')
    print(f'# gold stars: {stars}')
    score = predict_categorical(model, review)
    print(f'# predicted score: {score}')
    print(f'=' * 125)
