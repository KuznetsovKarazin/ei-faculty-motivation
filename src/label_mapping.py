"""
Label taxonomies and the mapping between them.

GoEmotions uses 27 fine-grained emotions + neutral. ISEAR uses 7 different
emotions. To compare the two datasets we reduce GoEmotions to the standard
Ekman-6 + neutral grouping (the same grouping shipped by the GoEmotions
authors), and we use the *intersection* of that set with ISEAR's labels as
the common label space for cross-dataset evaluation.
"""

# Official GoEmotions fine-grained label order (index == label id in the tsv files).
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

# Official Ekman-6 grouping used by the GoEmotions paper/repo. "neutral" is
# added here for completeness (it is not part of the original Ekman taxonomy).
EKMAN_MAPPING = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude",
            "love", "optimism", "relief", "pride", "admiration", "desire",
            "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief",
                "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}

# Full label set used for in-domain GoEmotions evaluation (Experiment 1).
EKMAN_PLUS_NEUTRAL = ["anger", "disgust", "fear", "joy", "sadness",
                      "surprise", "neutral"]

# ISEAR's own 7 emotion categories.
ISEAR_LABELS = ["joy", "fear", "anger", "sadness", "disgust", "shame", "guilt"]

# Labels shared by both datasets. Used as the label space for the
# cross-dataset generalisation test and for the faculty vignettes.
COMMON_LABELS = ["anger", "disgust", "fear", "joy", "sadness"]


def _build_fine_to_ekman():
    """Invert EKMAN_MAPPING into a fine_label -> ekman_category lookup."""
    out = {}
    for ekman_cat, fine_list in EKMAN_MAPPING.items():
        for fine_label in fine_list:
            out[fine_label] = ekman_cat
    return out


FINE_TO_EKMAN = _build_fine_to_ekman()


def fine_index_to_name(idx):
    """Map a GoEmotions label id (int) to its fine-grained label name."""
    return GOEMOTIONS_LABELS[idx]


def fine_name_to_ekman(fine_label):
    """Map a fine-grained GoEmotions label to its Ekman+neutral category."""
    return FINE_TO_EKMAN[fine_label]
