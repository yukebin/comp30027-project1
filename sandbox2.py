import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import List, Set, Dict, Tuple


def load_and_preprocess_data(filepath: str, text_col: str = 'textPreprocessed', label_col: str = 'class') -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    """
    Load and preprocess the dataset.
    
    Args:
        filepath (str): Path to the CSV file.
        text_col (str): Column name for preprocessed text.
        label_col (str): Column name for class labels.
    
    Returns:
        Tuple[pd.DataFrame, List[str], np.ndarray]: Preprocessed DataFrame, tokenized texts, and labels.
    """
    data = pd.read_csv(filepath)
    data.dropna(subset=[text_col], inplace=True)
    data['tokens'] = data[text_col].apply(lambda x: x.split())
    return data, data['tokens'].tolist(), data[label_col].values


def build_vocabulary(tokens: List[List[str]]) -> List[str]:
    """
    Build a vocabulary from tokenized texts.
    
    Args:
        tokens (List[List[str]]): List of tokenized texts.
    
    Returns:
        List[str]: Vocabulary list.
    """
    vocabulary = set()
    for token_list in tokens:
        vocabulary.update(token_list)
    return list(vocabulary)


def train_naive_bayes(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Tuple[Dict[int, float], Dict[int, np.ndarray]]:
    """
    Train a Naive Bayes model.
    
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        alpha (float): Smoothing parameter.
    
    Returns:
        Tuple[Dict[int, float], Dict[int, np.ndarray]]: Priors and likelihoods.
    """
    # Calculate priors
    classes, class_counts = np.unique(y, return_counts=True)
    priors = {c: count / len(y) for c, count in zip(classes, class_counts)}
    
    # Calculate likelihoods
    vocab_size = X.shape[1]
    likelihoods = {}
    for c in classes:
        X_c = X[y == c]
        word_counts = X_c.sum(axis=0)
        word_counts = np.asarray(word_counts).flatten()
        total_count = word_counts.sum()
        likelihoods[c] = (word_counts + alpha) / (total_count + alpha * vocab_size)
    
    return priors, likelihoods