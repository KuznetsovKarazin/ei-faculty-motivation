"""Shared evaluation helpers for all three experiments."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
from sklearn.metrics.pairwise import cosine_similarity


def classification_metrics(y_true, y_pred, labels):
    """Return a dict with accuracy, macro-F1, a full classification report
    (as text) and the confusion matrix (as a nested list, JSON-friendly)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro",
                              zero_division=0),
        "report": classification_report(y_true, y_pred, labels=labels,
                                          zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred,
                                              labels=labels).tolist(),
        "labels": labels,
    }


def pairwise_tfidf_similarity(messages, references):
    """For each (message, reference) pair, fit one shared TF-IDF space over
    all messages+references and return the list of cosine similarities.

    This is a fully offline, dependency-light stand-in for a neural
    embedding similarity (e.g. sentence-transformers), which would need a
    model download from the Hugging Face Hub. Swap this out for a neural
    embedding model if you have full internet access and want a richer
    similarity signal.
    """
    assert len(messages) == len(references)
    vectorizer = TfidfVectorizer()
    all_text = list(messages) + list(references)
    matrix = vectorizer.fit_transform(all_text)
    n = len(messages)
    msg_vecs = matrix[:n]
    ref_vecs = matrix[n:]
    sims = cosine_similarity(msg_vecs, ref_vecs)
    return np.diag(sims).tolist()
