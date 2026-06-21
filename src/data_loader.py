"""
Download and load the two public emotion datasets used in this project:

- GoEmotions (Demszky et al., 2020, Google Research) — 27 fine-grained
  emotions + neutral, ~58k Reddit comments. We use the official train/dev/
  test split that ships with the dataset repository.
- ISEAR (Scherer & Wallbott) — ~7.6k self-reported emotional situations,
  7 emotions, collected across many countries.

Both datasets are downloaded from public GitHub mirrors and cached under
data/raw/ on first use. No GitHub auth / API is needed, only plain file
downloads, so this works behind a restrictive network that only allows
raw.githubusercontent.com.
"""

import csv
import io
import os
import re

import pandas as pd
import requests

from src.label_mapping import fine_index_to_name, fine_name_to_ekman

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")

GOEMOTIONS_BASE = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/goemotions/data"
)
ISEAR_URL = (
    "https://raw.githubusercontent.com/sinmaniphel/py_isear_dataset/"
    "master/isear.csv"
)

GOEMOTIONS_FILES = {
    "train": f"{GOEMOTIONS_BASE}/train.tsv",
    "dev": f"{GOEMOTIONS_BASE}/dev.tsv",
    "test": f"{GOEMOTIONS_BASE}/test.tsv",
}


def _download(url, dest_path, timeout=30):
    """Download url to dest_path if not already cached. Fails loudly with
    a clear message if the network is unreachable (e.g. sandboxed env)."""
    if os.path.exists(dest_path):
        return dest_path

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not download {url}.\n"
            "This usually means there is no internet access to GitHub from "
            "this machine. Re-run on a machine with normal internet access, "
            "or manually place the file at:\n  " + dest_path
        ) from exc

    with open(dest_path, "wb") as f:
        f.write(resp.content)
    return dest_path


def download_goemotions():
    """Download the 3 GoEmotions split files into data/raw/."""
    paths = {}
    for split, url in GOEMOTIONS_FILES.items():
        dest = os.path.join(RAW_DIR, f"goemotions_{split}.tsv")
        paths[split] = _download(url, dest)
    return paths


def download_isear():
    """Download the ISEAR CSV into data/raw/."""
    dest = os.path.join(RAW_DIR, "isear_raw.csv")
    return _download(ISEAR_URL, dest)


def load_goemotions(split="train", single_label_only=True):
    """Load one GoEmotions split as a DataFrame with columns [text, label].

    `label` is the Ekman+neutral category (7 classes), not the raw 27-way
    fine-grained label. Rows annotated with more than one fine-grained
    emotion are dropped by default (single_label_only=True) to keep the
    classification task simple and comparable to ISEAR, which is single
    label by construction.
    """
    path = download_goemotions()[split]

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            text, label_ids = parts[0], parts[1]
            ids = [int(x) for x in label_ids.split(",") if x != ""]
            if single_label_only and len(ids) != 1:
                continue
            if not ids:
                continue
            # if multiple labels are kept, just use the first one
            fine_name = fine_index_to_name(ids[0])
            ekman_label = fine_name_to_ekman(fine_name)
            rows.append((text, ekman_label))

    return pd.DataFrame(rows, columns=["text", "label"])


_ARTIFACT_PATTERN = re.compile(r"\s*\u00e1\s*")  # stray "á" encoding glitch


def load_isear():
    """Load ISEAR as a DataFrame with columns [text, label].

    `label` is one of: joy, fear, anger, sadness, disgust, shame, guilt.
    """
    path = download_isear()

    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="|")
        header = next(reader)
        text_idx = header.index("SIT")
        label_idx = header.index("Field1")

        rows = []
        for r in reader:
            if len(r) <= max(text_idx, label_idx):
                continue
            text = r[text_idx].strip()
            label = r[label_idx].strip().lower()
            if not text or not label:
                continue
            # clean up a known encoding artifact in this distribution of ISEAR
            text = _ARTIFACT_PATTERN.sub(" ", text)
            text = re.sub(r"\s+", " ", text).strip()
            rows.append((text, label))

    return pd.DataFrame(rows, columns=["text", "label"])


def load_isear_common_labels():
    """Load ISEAR restricted to the 5 labels shared with GoEmotions/Ekman
    (anger, disgust, fear, joy, sadness). Drops 'shame' and 'guilt' rows,
    which have no Ekman equivalent."""
    from src.label_mapping import COMMON_LABELS
    df = load_isear()
    return df[df["label"].isin(COMMON_LABELS)].reset_index(drop=True)


def stratified_sample(df, label_col, n_per_class, seed=42):
    """Sample up to n_per_class rows per label, capped by how many rows
    actually exist for that label (rather than failing or silently using
    fewer). This matters for rare classes: a flat random sample of a small
    fraction of a 7-class imbalanced dataset can leave 2-3 examples for the
    rarest class, which is not enough to estimate per-class accuracy
    reliably. Returns a new, shuffled DataFrame; also prints how many rows
    were used per class so under-sized classes are visible, not silent.
    """
    parts = []
    for label, group in df.groupby(label_col):
        n = min(n_per_class, len(group))
        parts.append(group.sample(n=n, random_state=seed))
        if n < n_per_class:
            print(f"  [stratified_sample] '{label}': only {n} rows available "
                  f"(asked for {n_per_class}) - using all of them")
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


if __name__ == "__main__":
    # Running this file directly just triggers (and verifies) the downloads.
    print("Downloading GoEmotions ...")
    download_goemotions()
    print("Downloading ISEAR ...")
    download_isear()
    print("Done. Files cached in:", RAW_DIR)
