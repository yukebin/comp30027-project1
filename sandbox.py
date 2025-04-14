import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List, Set, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
labeled_train: pd.DataFrame = pd.read_csv('data/sms_supervised_train.csv')

# Split text into tokens
labeled_train.dropna(subset=['textPreprocessed'], inplace=True)
labeled_train['tokens'] = labeled_train['textPreprocessed'].apply(lambda x: x.split())

# Define vocabulary
vocabulary: Set[str] = set()
for tokens in labeled_train['tokens']:
    vocabulary.update(tokens)
vocab_list = list(vocabulary)

# Create count matrix
# vectorizer = CountVectorizer(vocabulary=vocabulary)
# X = vectorizer.transform(data['textPreprocessed'])

vectorizer: CountVectorizer = CountVectorizer(vocabulary=vocab_list)
X: np.ndarray = vectorizer.transform(labeled_train['textPreprocessed'])
y: np.ndarray = labeled_train['class'].values

# Compute prior probabilities

def calc_prior(data: pd.DataFrame, label_col: str = 'class') -> Dict[int, float]:
    """
    Calculates prior probabilities P(class) for each class label.

    Args:
        data: DataFrame containing labeled instances
        label_col: Name of the column that contains class labels

    Returns:
        Series of prior probabilities indexed by class label
    """
    class_counts = data[label_col].value_counts()
    total = class_counts.sum()
    return (class_counts / total).to_dict()


# Compute likelihoods

def calc_likelihood(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> dict:
    """
    Calculates likelihood P(word|class) with Laplace smoothing.
    
    Args:
        X: Count matrix (documents × vocabulary), from CountVectorizer
        y: Class labels (0 or 1)
        alpha: Smoothing parameter
    
    Returns:
        A dictionary {class_label: np.ndarray of P(word | class)}
    """
    classes = np.unique(y)
    vocab_size = X.shape[1]
    likelihoods = {}

    for c in classes:
        X_c = X[y == c]                     # rows for class c (y == c) is a mask
        word_counts = X_c.sum(axis=0)       # sum over rows → word frequency vector
        word_counts = np.asarray(word_counts).flatten()  # convert to 1D array

        total_count = word_counts.sum()
        likelihood = (word_counts + alpha) / (total_count + alpha * vocab_size)
        likelihoods[c] = likelihood

    return likelihoods

y = labeled_train['class'].values
likelihoods = calc_likelihood(X, y)


# Example: probability of each word in class 1 (scam)
print("Top P(word|scam):")

# Get indices of top 10 highest likelihoods for class 1 (scam)
top_words = np.array(vectorizer.get_feature_names_out())[np.argsort(-likelihoods[1])[:10]]

# Get the top 10 sorted likelihood values for class 1
top_probs = np.sort(likelihoods[1])[::-1][:10]

# Print each top word with its corresponding probability
for word, prob in zip(top_words, top_probs):
    print(f"{word}: {prob:.4f}")


def calc_posterior(counts: np.ndarray, priors: Dict[int, float], 
                   likelihoods: Dict[int, np.ndarray]) -> Dict[int, float]:
    """
    Calculates the log-posterior probability for each class.

    Args:
        counts: word count vector for the instance (1D array)
        priors: dict of P(class)
        likelihoods: dict of P(word | class) for each class

    Returns:
        Dict mapping class → log posterior score
    """
    scores = {}
    for c in priors:
        log_prior = np.log(priors[c])
        log_likelihood = np.log(likelihoods[c])
        score = log_prior + np.dot(counts, log_likelihood)
        scores[c] = score
    return scores

def predict_NB(text: str, priors: Dict[int, float], likelihoods: Dict[int, np.ndarray], vectorizer: CountVectorizer) -> int:
    """
    Predicts the class for a single preprocessed SMS using Multinomial NB.

    Args:
        text: preprocessed message (space-separated string)
        priors: prior probabilities
        likelihoods: word likelihoods
        vectorizer: fitted CountVectorizer with known vocabulary

    Returns:
        Predicted class label (e.g., 0 or 1)
    """
    # Transform text to count vector (shape: (1, vocab_size))
    count_vector = vectorizer.transform([text])
    # Convert to 1D array
    counts = count_vector.toarray().flatten()
    # Calculate posterior scores
    scores = calc_posterior(counts, priors, likelihoods)
    # Return class with max posterior
    return max(scores, key=scores.get)



###############################################################################

# Train the NB model
priors = calc_prior(labeled_train, 'class')
print(priors)
likelihoods = calc_likelihood(X, labeled_train['class'].values)
# Try a test instance (already preprocessed!)
test_text = "call for free mclicious!"
prediction = predict_NB(test_text, priors, likelihoods, vectorizer)
print(f"Prediction for: '{test_text}' →", "scam" if prediction == 1 else "non-malicious")


print()


###############################################################################

test_df = pd.read_csv('data/sms_test.csv')
test_df.dropna(subset=['textPreprocessed'], inplace=True)


# Prediction function for a batch of texts
def predict_batch(texts: List[str], priors: Dict[int, float], likelihoods: Dict[int, np.ndarray], vectorizer: CountVectorizer) -> Tuple[np.ndarray, np.ndarray]:
    predictions: List[int] = []
    confidence_ratios: List[float] = []

    for text in texts:
        # Transform the text into a count vector
        counts: np.ndarray = vectorizer.transform([text]).toarray().flatten()
        # Calculate posterior scores for each class
        scores: Dict[int, float] = calc_posterior(counts, priors, likelihoods)
        # Append the predicted class (class with the highest posterior score)
        predictions.append(max(scores, key=scores.get))
        # Append the confidence ratio (P(class 1) / P(class 0))
        confidence_ratios.append(np.exp(scores[1] - scores[0]))

    return np.array(predictions), np.array(confidence_ratios)

# Get predictions on test set
test_texts = test_df['textPreprocessed'].tolist()
true_labels = test_df['class'].values
predicted_labels, conf_test = predict_batch(test_texts, priors, likelihoods, vectorizer)

# === 1. Accuracy and confusion matrix ===
acc = accuracy_score(true_labels, predicted_labels)
prec = precision_score(true_labels, predicted_labels)
rec = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)

print("\n--- Evaluation Metrics ---")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix:")

# Plot it
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Non-Malicious", "Scam"],
            yticklabels=["Non-Malicious", "Scam"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()




# === 2. Out-of-vocabulary (OOV) and skipped messages ===
vocab_set = set(vocabulary)
oov_messages = 0
skipped_messages = 0

for text in test_texts:
    tokens = text.split()
    tokens_in_vocab = [token for token in tokens if token in vocab_set]
    if not tokens_in_vocab:
        skipped_messages += 1
    elif len(tokens) != len(tokens_in_vocab):
        oov_messages += 1

print(f"\nOOV Messages (some tokens missing): {oov_messages}")
print(f"Skipped Messages (no tokens in vocab): {skipped_messages}")

# === 3. Examples of model confidence ===
test_df['prediction'] = predicted_labels
test_df['confidence_ratio'] = conf_test

print("\n--- High Confidence Scam Predictions ---")
high_conf_scam = test_df[test_df['prediction'] == 1].sort_values(by='confidence_ratio', ascending=False).head(3)
print(high_conf_scam[['textOriginal', 'textPreprocessed', 'confidence_ratio']])

print("\n--- High Confidence Non-Malicious Predictions ---")
high_conf_nonmal = test_df[test_df['prediction'] == 0].sort_values(by='confidence_ratio', ascending=True).head(5)
print(high_conf_nonmal[['textOriginal', 'textPreprocessed', 'confidence_ratio']])

print("\n--- Boundary Cases (Confidence Ratio ≈ 1) ---")
boundary = test_df[(test_df['confidence_ratio'] > 0.9) & (test_df['confidence_ratio'] < 1.1)].head(3)
print(boundary[['textOriginal', 'textPreprocessed', 'confidence_ratio']])


###############################################################################

unlabelled_train = pd.read_csv('data/sms_unlabelled.csv')
unlabelled_train.dropna(subset=['textPreprocessed'], inplace=True)

_, conf_unlabelled = predict_batch(unlabelled_train['textPreprocessed'].tolist(), priors, likelihoods, vectorizer)
unlabelled_train['confidence_ratio'] = conf_unlabelled

# strategy 1, randomly sampling 200 instances

random_sample = unlabelled_train.sample(n=200, random_state=528)
expanded_random = pd.concat([labeled_train, random_sample], ignore_index=True)

def train_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame, vectorizer: CountVectorizer) -> Dict[str, float]:
    X_train = vectorizer.transform(train_df['textPreprocessed'])
    y_train = train_df['class'].values
    priors = calc_prior(train_df)
    likelihoods = calc_likelihood(X_train, y_train)
    preds, _ = predict_batch(test_df['textPreprocessed'].tolist(), priors, likelihoods, vectorizer)
    y_true = test_df['class'].values
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds),
        "Recall": recall_score(y_true, preds),
        "F1 Score": f1_score(y_true, preds),
        "Training Size": len(train_df)
    }


metrics_random = train_and_eval(expanded_random, test_df, vectorizer)
print("\n--- Random Sampling Evaluation ---")
print(metrics_random)

unlabelled_train['distance_to_boundary'] = np.abs(unlabelled_train['confidence_ratio'] - 1)
# strategy 2, using distance to boundary
low_conf_sample = unlabelled_train.nsmallest(200, 'distance_to_boundary')
expanded_low_conf = pd.concat([labeled_train, low_conf_sample], ignore_index=True)
metrics_uncertain = train_and_eval(expanded_low_conf, test_df, vectorizer)
print("\n--- Uncertain Sampling Evaluation ---")
print(metrics_uncertain)

###############################################################################

sampled_base_df = labeled_train.sample(n=200, random_state=7)
metrics_sampled_base = train_and_eval(sampled_base_df, test_df, vectorizer)

print("\n--- Random Sampling Evaluation (Base) ---")
print(metrics_sampled_base)

expanded_random_small = pd.concat([sampled_base_df, random_sample], ignore_index=True)
metrics_expanded_random = train_and_eval(expanded_random_small, test_df, vectorizer)
print("\n--- Expanded Random Sampling Evaluation ---")
print(metrics_expanded_random)

expanded_small = pd.concat([sampled_base_df, low_conf_sample], ignore_index=True)
metrics_expanded_small = train_and_eval(expanded_small, test_df, vectorizer)
print("\n--- Expanded Small Evaluation ---")
print(metrics_expanded_small)


# this is using sklean for double checking

# Train model
model = MultinomialNB(alpha=1.0)
model.fit(X, labeled_train['class'])

# Get learned priors (in log space)
print("Class log priors:", model.class_log_prior_)

# Get learned likelihoods (in log space)
feature_names = vectorizer.get_feature_names_out()
for class_index, class_log_probs in enumerate(model.feature_log_prob_):
    print(f"\nTop words for class {class_index}:")
    top_indices = np.argsort(class_log_probs)[-10:][::-1]
    for i in top_indices:
        print(f"{feature_names[i]}: {np.exp(class_log_probs[i]):.5f}")

# # Evaluate model
# y_pred = model.predict(X)
# accuracy = accuracy_score(data['class'], y_pred)
# precision = precision_score(data['class'], y_pred)
# recall = recall_score(data['class'], y_pred)
# f1 = f1_score(data['class'], y_pred)

# print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")