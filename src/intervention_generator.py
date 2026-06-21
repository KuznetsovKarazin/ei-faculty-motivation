"""
Generates a motivational message for a faculty member, given their detected
emotion and the SDT need it maps to (see sdt_mapping.py).

Two generation modes:
- "template" (default): picks from a small set of hand-written, need-specific
  templates. Deterministic-ish (one random choice), needs no network access,
  used for the experiments in this repo so results are reproducible without
  an API key.
- "llm": calls an LLM API (Anthropic) to generate a less repetitive,
  more context-aware message. Requires an ANTHROPIC_API_KEY environment
  variable. Falls back to "template" mode with a warning if the API call
  fails for any reason (no key, no internet, rate limit, etc.).

GENERIC_MESSAGE is the non-personalized baseline used for comparison in
Experiment 3: the same message regardless of detected emotion/need.
"""

import os
import random

GENERIC_MESSAGE = (
    "Thank you for sharing how things are going. Keep up the great work, "
    "you are doing fine!"
)

TEMPLATES = {
    "autonomy": [
        "It sounds like a decision was made without your input. What is "
        "one part of this you could still shape on your own terms?",
        "You may have more room to decide how you approach this than it "
        "feels like right now. What would reclaiming a small piece of "
        "control look like today?",
    ],
    "competence": [
        "Feeling uncertain before a high-stakes moment is common, even for "
        "very experienced people. What is one thing you already know you "
        "do well here?",
        "It is understandable to worry about being judged. What would "
        "count as 'good enough' for you this time, separate from how "
        "others might score it?",
    ],
    "relatedness": [
        "It sounds like this has felt isolating. Is there one colleague "
        "you could reach out to this week, even briefly?",
        "Your work seems to be going unnoticed right now, and that is "
        "worth naming. Who would you want to know about it?",
    ],
    "reinforcement": [
        "That is genuinely worth celebrating, well done. What made this "
        "moment work so well, so you can look for it again?",
        "This is a good sign that things are on the right track. Take a "
        "moment to actually notice it before moving to the next task.",
    ],
    "none": [
        "How are things going for you at the moment?",
    ],
}


def _template_message(need, rng=None):
    rng = rng or random
    options = TEMPLATES.get(need, TEMPLATES["none"])
    return rng.choice(options)


def _llm_message(emotion, need, model=None):
    """Call the Anthropic API to generate a message. Raises on any failure
    so the caller can fall back to template mode."""
    import requests

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    prompt = (
        f"A university faculty member is expressing the emotion '{emotion}', "
        f"which research suggests reflects an unmet '{need}' need "
        "(Self-Determination Theory). Write one short (<=40 words), warm, "
        "professional, non-clinical motivational message addressed to them "
        "directly. Do not mention SDT, emotions, or psychology terminology "
        "explicitly."
    )

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 120,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return "".join(b["text"] for b in data["content"] if b["type"] == "text").strip()


def generate_message(emotion, need, mode="template", rng=None):
    """Return a single motivational message string.

    emotion: detected emotion label (used only by "llm" mode, for context)
    need:    SDT need category from sdt_mapping.get_need(emotion)
    mode:    "template" or "llm"
    """
    if mode == "llm":
        try:
            return _llm_message(emotion, need)
        except Exception as exc:  # noqa: BLE001 - any API/network error
            print(f"[intervention_generator] LLM mode failed ({exc}), "
                  "falling back to template mode.")
    return _template_message(need, rng=rng)
