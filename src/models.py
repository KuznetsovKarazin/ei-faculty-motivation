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

4. LLMFewShotBaseline  — classifies via an LLM API call (Anthropic) with a
   handful of labeled in-context examples per class, instead of keyword/
   n-gram matching. This is the most likely candidate to handle *implied*
   emotion (e.g. quiet isolation phrased without any "sad" vocabulary),
   which the other three methods structurally cannot detect. Needs an
   ANTHROPIC_API_KEY environment variable and internet access; costs one
   API call per *predicted* example (no training cost). Responses are
   cached on disk so repeated runs do not re-spend API credits.
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

    Trained models are cached to disk (`cache_dir`) keyed by model name +
    label set + training set size, so e.g. running --use_transformer in
    both Experiment 1 and Experiment 2 only pays the training cost once -
    the second call loads the cached fine-tuned model instead of
    retraining from scratch, which matters a lot on CPU.
    """

    def __init__(self, model_name="distilbert-base-uncased", num_epochs=3,
                 batch_size=16, lr=2e-5, cache_dir=None, seed=42,
                 val_texts=None, val_labels=None):
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.cache_dir = cache_dir or os.path.join(REPO_ROOT, "results", ".transformer_cache")
        # optional held-out set (e.g. GoEmotions dev split) for a per-epoch
        # validation accuracy printout - purely diagnostic, not used for
        # early stopping, to keep this simple and predictable.
        self.val_texts = val_texts
        self.val_labels = val_labels
        self.label2id = None
        self.id2label = None
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

    def _cache_key(self, y):
        import hashlib
        key_str = f"{self.model_name}|{sorted(set(y))}|{len(y)}|{self.num_epochs}|{self.lr}"
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]

    def fit(self, X, y):
        try:
            import torch
            from torch.utils.data import DataLoader, Dataset
            from transformers import (AutoModelForSequenceClassification,
                                       AutoTokenizer, get_linear_schedule_with_warmup)
        except ImportError as exc:
            raise TransformerUnavailableError(
                "transformers/torch not installed. "
                "Run: pip install transformers torch"
            ) from exc

        torch.manual_seed(self.seed)
        X = list(X)
        y = list(y)
        labels_sorted = sorted(set(y))
        self.label2id = {lab: i for i, lab in enumerate(labels_sorted)}
        self.id2label = {i: lab for lab, i in self.label2id.items()}

        cache_path = os.path.join(self.cache_dir, self._cache_key(y))
        if os.path.exists(os.path.join(cache_path, "config.json")):
            print(f"[TransformerBaseline] loading cached fine-tuned model from {cache_path} "
                  "(matches model/labels/train-size/epochs/lr of a previous run)")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(cache_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(cache_path)
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(self.device)
                return self
            except Exception as exc:  # noqa: BLE001 - corrupted cache, just retrain
                print(f"[TransformerBaseline] cache load failed ({exc}), retraining.")

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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        if self.device == "cpu":
            import os as _os
            n_cores = _os.cpu_count() or 1
            torch.set_num_threads(n_cores)
            print(f"[TransformerBaseline] CPU mode: using torch.set_num_threads({n_cores}) "
                  f"- intra-op matmul/attention math is parallelized across all "
                  f"{n_cores} logical cores automatically (this is PyTorch's "
                  "built-in CPU parallelism, not something this code adds on top of).")
        print(f"[TransformerBaseline] training {self.model_name} on {len(X)} examples, "
              f"{self.num_epochs} epoch(s), device={self.device}")

        dataset = TextDataset(X, y, self.tokenizer, self.label2id)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        total_steps = len(loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

        import time
        self.model.train()
        for epoch in range(self.num_epochs):
            t0 = time.time()
            running_loss, n_batches = 0.0, 0
            for step, batch in enumerate(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                running_loss += loss.item()
                n_batches += 1
                if (step + 1) % 200 == 0:
                    elapsed = time.time() - t0
                    rate = elapsed / (step + 1)
                    remaining = (len(loader) - step - 1) * rate
                    print(f"  epoch {epoch + 1}/{self.num_epochs} "
                          f"batch {step + 1}/{len(loader)} "
                          f"avg_loss={running_loss / n_batches:.4f} "
                          f"~{remaining / 60:.1f} min left this epoch")
            msg = (f"[TransformerBaseline] epoch {epoch + 1}/{self.num_epochs} done, "
                   f"avg_loss={running_loss / max(n_batches, 1):.4f}, "
                   f"{(time.time() - t0):.0f}s")
            if self.val_texts is not None and self.val_labels is not None:
                val_preds = self.predict(self.val_texts)
                val_acc = sum(p == t for p, t in zip(val_preds, self.val_labels)) / len(self.val_labels)
                msg += f", val_accuracy={val_acc:.3f}"
                self.model.train()
            print(msg)

        os.makedirs(cache_path, exist_ok=True)
        self.model.save_pretrained(cache_path)
        self.tokenizer.save_pretrained(cache_path)
        print(f"[TransformerBaseline] cached fine-tuned model to {cache_path}")
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


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM baseline cannot be used (no API key, or the
    API call failed for any other reason)."""


class LLMFewShotBaseline:
    """Emotion classifier that calls an LLM with a few in-context examples
    per class instead of matching keywords or n-grams.

    Unlike NRCLexiconBaseline/TfidfBaseline, this has near-zero "training"
    cost (it just stores a handful of example texts at fit() time) but a
    real, recurring cost at predict() time: one API call per example. Use
    `max_predict` / sampling at the experiment level to control cost on
    large test sets; on the 40 faculty vignettes the full set is cheap.
    """

    def __init__(self, n_examples_per_label=2, model=None, cache_path=None,
                 sleep_between_calls=0.0, max_workers=8, max_retries=4):
        self.n_examples_per_label = n_examples_per_label
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.cache_path = cache_path or os.path.join(REPO_ROOT, "results", ".llm_cache.json")
        self.sleep_between_calls = sleep_between_calls
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.examples = []  # list of (text, label)
        self._cache = {}

    def fit(self, X, y):
        import random
        by_label = {}
        for text, label in zip(X, y):
            by_label.setdefault(label, []).append(text)
        rng = random.Random(42)
        self.examples = []
        for label, texts in by_label.items():
            chosen = rng.sample(texts, min(self.n_examples_per_label, len(texts)))
            for t in chosen:
                self.examples.append((t, label))
        rng.shuffle(self.examples)
        return self

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, encoding="utf-8") as f:
                self._cache = json.load(f)
        else:
            self._cache = {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, indent=2)

    def _build_prompt(self, text, labels):
        lines = [
            "Classify the dominant emotion expressed by the speaker in the "
            "text below. The text may be a short social-media comment, a "
            "self-reported emotional situation, or a first-person statement "
            "by a university faculty member.",
            f"Choose exactly one label from this list: {', '.join(labels)}.",
            "Reply with only the label, lowercase, nothing else.",
            "",
        ]
        for ex_text, ex_label in self.examples:
            if ex_label in labels:
                lines.append(f'Text: "{ex_text}"')
                lines.append(f"Label: {ex_label}")
                lines.append("")
        lines.append(f'Text: "{text}"')
        lines.append("Label:")
        return "\n".join(lines)

    def _call_api(self, prompt, api_key, max_retries=None):
        import time as _time
        import requests

        max_retries = max_retries or self.max_retries
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 10,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=30,
                )
            except requests.RequestException:
                if attempt == max_retries - 1:
                    raise
                _time.sleep(2 ** attempt)
                continue

            # rate limit (429) or transient server error (5xx): back off and retry
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == max_retries - 1:
                    resp.raise_for_status()
                retry_after = resp.headers.get("retry-after")
                wait = float(retry_after) if retry_after else float(2 ** attempt)
                print(f"[LLMFewShotBaseline] status {resp.status_code}, "
                      f"retrying in {wait:.0f}s (attempt {attempt + 1}/{max_retries}) ...")
                _time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            text = "".join(b["text"] for b in data["content"] if b["type"] == "text")
            return text.strip().lower()

        raise RuntimeError(f"Exceeded {max_retries} retries calling the Anthropic API")

    def predict(self, X, labels):
        import hashlib
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMUnavailableError(
                "ANTHROPIC_API_KEY is not set. Set it to use LLMFewShotBaseline, "
                "e.g.: export ANTHROPIC_API_KEY=sk-ant-..."
            )

        self._load_cache()
        fallback = "neutral" if "neutral" in labels else sorted(labels)[0]
        X = list(X)
        n_total = len(X)

        # work out which rows are already cached vs need a real API call
        keys = [hashlib.sha256(f"{self.model}|{sorted(labels)}|{text}".encode("utf-8")).hexdigest()
                for text in X]
        preds = [None] * n_total
        todo = []  # (index, text, key) for rows not yet cached
        for i, (text, key) in enumerate(zip(X, keys)):
            if key in self._cache:
                preds[i] = self._cache[key]
            else:
                todo.append((i, text, key))

        if not todo:
            return preds

        cache_lock = threading.Lock()
        first_error = []
        t_start = time.time()
        completed = [0]

        def _work(item):
            idx, text, key = item
            prompt = self._build_prompt(text, labels)
            raw = self._call_api(prompt, api_key)
            pred = raw if raw in labels else fallback
            return idx, key, pred

        print(f"[LLMFewShotBaseline] {len(todo)} new API calls needed "
              f"({n_total - len(todo)} already cached), "
              f"{self.max_workers} parallel workers ...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(_work, item): item for item in todo}
            try:
                for fut in as_completed(futures):
                    idx, text, key = futures[fut]
                    try:
                        idx, key, pred = fut.result()
                    except Exception as exc:  # noqa: BLE001 - any API/network error
                        first_error.append(exc)
                        for f in futures:
                            f.cancel()
                        break
                    preds[idx] = pred
                    with cache_lock:
                        self._cache[key] = pred
                        completed[0] += 1
                        if completed[0] % 25 == 0:
                            self._save_cache()
                            elapsed = time.time() - t_start
                            rate = elapsed / completed[0]
                            remaining = (len(todo) - completed[0]) * rate / self.max_workers
                            print(f"[LLMFewShotBaseline] {completed[0]}/{len(todo)} done, "
                                  f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")
            finally:
                with cache_lock:
                    if completed[0]:
                        self._save_cache()

        if first_error:
            raise LLMUnavailableError(f"LLM API call failed: {first_error[0]}") from first_error[0]
        return preds

