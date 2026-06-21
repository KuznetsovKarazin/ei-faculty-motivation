"""
Three emotion classifiers used in Experiment 1, in increasing order of
sophistication:

1. NRCLexiconBaseline  — word-counting against the NRC emotion lexicon.
   This mirrors the method used in prior teacher-emotion research
   (e.g. Chen et al., 2020, Frontiers in Psychology), which also relied on
   word-frequency-based discrete emotion counting. It needs no training.

2. TfidfBaseline       — classic supervised ML (TF-IDF + Logistic
   Regression). Needs training data, but no internet access at train time.

3. TransformerBaseline — fine-tunes a pretrained transformer
   (e.g. distilbert-base-uncased) from Hugging Face. This needs internet
   access to download the pretrained weights the first time it runs, so it
   is optional and is skipped automatically if that download fails (e.g. in
   a sandboxed / offline environment). Run it on a machine with normal
   internet access to get transformer-level results.
"""

import json
import os
import re
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEXICON_PATH = os.path.join(REPO_ROOT, "data", "lexicon", "nrc_emotion_lexicon.json")

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text):
    return [w.lower() for w in _WORD_RE.findall(text)]


class NRCLexiconBaseline:
    """Zero-shot, rule-based emotion classifier using the NRC lexicon.

    For each text, counts how many words are associated with each
    candidate emotion in the NRC lexicon and predicts the most frequent
    one. Predicts "neutral" when no lexicon word matches and "neutral" is
    one of the allowed labels.
    """

    # emotion names as used by the NRC lexicon
    NRC_EMOTIONS = {"anger", "disgust", "fear", "joy", "sadness", "surprise"}

    def __init__(self, lexicon_path=LEXICON_PATH):
        with open(lexicon_path, encoding="utf-8") as f:
            self.lexicon = json.load(f)

    def fit(self, X, y):
        # rule-based: nothing to learn, kept for a consistent sklearn-like API
        return self

    def predict(self, X, labels):
        """labels: the list of allowed output classes for this run
        (e.g. EKMAN_PLUS_NEUTRAL or COMMON_LABELS)."""
        allowed_nrc = self.NRC_EMOTIONS.intersection(labels)
        fallback = "neutral" if "neutral" in labels else sorted(labels)[0]

        preds = []
        for text in X:
            tokens = _tokenize(text)
            counts = Counter()
            for tok in tokens:
                for emo in self.lexicon.get(tok, []):
                    if emo in allowed_nrc:
                        counts[emo] += 1
            if not counts:
                preds.append(fallback)
            else:
                preds.append(counts.most_common(1)[0][0])
        return preds


class TfidfBaseline:
    """TF-IDF features + multinomial Logistic Regression."""

    def __init__(self, max_features=20000, ngram_range=(1, 2)):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features,
                                       ngram_range=ngram_range,
                                       sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000,
                                        class_weight="balanced")),
        ])

    def fit(self, X, y):
        self.pipeline.fit(list(X), list(y))
        return self

    def predict(self, X, labels=None):
        # `labels` is accepted for interface symmetry with NRCLexiconBaseline
        # but not used: the model only knows the labels it was trained on.
        return list(self.pipeline.predict(list(X)))


class TransformerUnavailableError(RuntimeError):
    """Raised when the transformer baseline cannot be used (no internet
    access to download pretrained weights, or missing dependencies)."""


class TransformerBaseline:
    """Fine-tunes a small pretrained transformer for emotion classification.

    Optional: requires `transformers` + `torch` and internet access to
    download the base model on first use. Intended to be run by the user
    on a machine with full internet access, not necessarily inside a
    restricted sandbox.
    """

    def __init__(self, model_name="distilbert-base-uncased", num_epochs=3,
                 batch_size=16, lr=2e-5):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.label2id = None
        self.id2label = None
        self.model = None
        self.tokenizer = None

    def fit(self, X, y):
        try:
            import torch
            from torch.utils.data import DataLoader, Dataset
            from transformers import (AutoModelForSequenceClassification,
                                       AutoTokenizer)
        except ImportError as exc:
            raise TransformerUnavailableError(
                "transformers/torch not installed. "
                "Run: pip install transformers torch"
            ) from exc

        labels_sorted = sorted(set(y))
        self.label2id = {lab: i for i, lab in enumerate(labels_sorted)}
        self.id2label = {i: lab for lab, i in self.label2id.items()}

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=len(labels_sorted)
            )
        except Exception as exc:  # noqa: BLE001 - any download/network error
            raise TransformerUnavailableError(
                f"Could not load pretrained model '{self.model_name}'. "
                "This usually means there is no internet access to the "
                "Hugging Face Hub from this machine. Run this experiment "
                "with --use_transformer on a machine with normal internet "
                "access."
            ) from exc

        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, label2id):
                self.enc = tokenizer(list(texts), truncation=True,
                                      padding=True, max_length=64)
                self.labels = [label2id[lab] for lab in labels]

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.device = device

        dataset = TextDataset(X, y, self.tokenizer, self.label2id)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.num_epochs):
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        return self

    def predict(self, X, labels=None):
        import torch
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_texts = list(X[i:i + self.batch_size])
                enc = self.tokenizer(batch_texts, truncation=True,
                                      padding=True, max_length=64,
                                      return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                ids = logits.argmax(dim=-1).tolist()
                preds.extend(self.id2label[i] for i in ids)
        return preds
