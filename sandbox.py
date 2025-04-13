import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Set, Dict


# Load data
labeled_train: pd.DataFrame = pd.read_csv('data/sms_supervised_train.csv')

# Split text into tokens
labeled_train.dropna(subset=['textPreprocessed'], inplace=True)
labeled_train['tokens'] = labeled_train['textPreprocessed'].apply(lambda x: x.split())

# Define vocabulary
vocabulary: Set[str] = set()
for tokens in labeled_train['tokens']:
    vocabulary.update(tokens)
vocabulary = list(vocabulary)

# Create count matrix
# vectorizer = CountVectorizer(vocabulary=vocabulary)
# X = vectorizer.transform(data['textPreprocessed'])

vectorizer: CountVectorizer = CountVectorizer(vocabulary=vocabulary)
X: np.ndarray = vectorizer.transform(labeled_train['textPreprocessed'])

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