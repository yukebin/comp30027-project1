import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
data = pd.read_csv('data/sms_supervised_train.csv')

# Split text into tokens
data.dropna(subset=['textPreprocessed'], inplace=True)
data['tokens'] = data['textPreprocessed'].apply(lambda x: x.split())

# Define vocabulary
vocabulary = set()
for tokens in data['tokens']:
    vocabulary.update(tokens)
vocabulary = list(vocabulary)

# Create count matrix
vectorizer = CountVectorizer(vocabulary=vocabulary)
X = vectorizer.transform(data['textPreprocessed'])

# Compute prior probabilities
class_counts = data['class'].value_counts()
prior_probabilities = class_counts / class_counts.sum()

# Train model
model = MultinomialNB(alpha=1.0)
model.fit(X, data['class'])

# Get learned priors (in log space)
print("Class log priors:", model.class_log_prior_)

# Get learned likelihoods (in log space)
feature_names = vectorizer.get_feature_names_out()
for class_index, class_log_probs in enumerate(model.feature_log_prob_):
    print(f"\nTop words for class {class_index}:")
    top_indices = np.argsort(class_log_probs)[-10:][::-1]
    for i in top_indices:
        print(f"{feature_names[i]}: {np.exp(class_log_probs[i]):.5f}")

# Evaluate model
y_pred = model.predict(X)
accuracy = accuracy_score(data['class'], y_pred)
precision = precision_score(data['class'], y_pred)
recall = recall_score(data['class'], y_pred)
f1 = f1_score(data['class'], y_pred)

print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")