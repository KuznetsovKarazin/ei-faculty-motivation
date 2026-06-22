"""
Generates a motivational message for a faculty member. Four levels are
provided, forming an ablation ladder so a paper can show what each layer
of personalization actually adds, rather than just "personalized vs not":

  1. generic        - same message regardless of anything detected
  2. emotion_only    - acknowledges the raw detected emotion, no theory
  3. need_only       - routes through the SDT need mapped from the emotion
                       (this was previously the only "personalized" mode)
  4. full_context    - uses the SDT need AND the specific situation
                       (vignette text + stressor category), via an LLM.
                       There is no template version of this level: the
                       point of it is to use situation-specific content
                       that cannot be captured by a small fixed template
                       set, so it always calls the LLM and raises a clear
                       error if no API key is available.

Levels 1-3 default to deterministic templates (no network needed, used so
experiments are reproducible without an API key) but can be routed through
the LLM too via mode="llm" for a less repetitive message.
"""

import os
import random

GENERIC_MESSAGE = (
    "Thank you for sharing how things are going. Keep up the great work, "
    "you are doing fine!"
)

# Level 2: acknowledges the emotion word itself, nothing theory-driven.
EMOTION_ONLY_TEMPLATES = {
    "anger": [
        "It's understandable to feel angry about this.",
        "That sounds like a genuinely frustrating situation to be in.",
    ],
    "disgust": [
        "That sounds like a really unpleasant situation to deal with.",
        "It's reasonable to feel uneasy about something like that.",
    ],
    "fear": [
        "It's natural to feel anxious in a situation like this.",
        "That sounds like a genuinely worrying thing to be dealing with.",
    ],
    "sadness": [
        "It sounds like that's been a difficult, disheartening experience.",
        "That sounds like a genuinely sad situation to be in.",
    ],
    "joy": [
        "It's great that this felt like a positive experience for you.",
        "That sounds like a genuinely rewarding moment.",
    ],
    "surprise": ["That sounds like it caught you off guard."],
    "neutral": ["Thanks for the update."],
}

# Level 3: routes through the SDT need, not the situation specifics.
NEED_ONLY_TEMPLATES = {
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


def _pick(templates_dict, key, fallback_key, rng=None):
    rng = rng or random
    options = templates_dict.get(key, templates_dict[fallback_key])
    return rng.choice(options)


def _call_llm(prompt, model=None, max_tokens=120):
    import requests

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return "".join(b["text"] for b in data["content"] if b["type"] == "text").strip()


def generate_emotion_only(emotion, mode="template", rng=None):
    """Level 2: acknowledge the raw emotion, no SDT theory, no situation."""
    if mode == "llm":
        prompt = (
            f"A university faculty member is expressing the emotion "
            f"'{emotion}'. Write one short (<=25 words) warm sentence that "
            "simply acknowledges that feeling. Do not give advice, do not "
            "reference any psychological theory, do not mention the "
            "specific situation."
        )
        try:
            return _call_llm(prompt, max_tokens=60)
        except Exception as exc:  # noqa: BLE001
            print(f"[intervention_generator] emotion_only LLM failed ({exc}), "
                  "falling back to template.")
    return _pick(EMOTION_ONLY_TEMPLATES, emotion, "neutral", rng=rng)


def generate_need_only(emotion, need, mode="template", rng=None):
    """Level 3: route through the SDT need, but stay generic about the
    situation (this is what earlier versions of this repo called simply
    "personalized")."""
    if mode == "llm":
        prompt = (
            f"A university faculty member is expressing the emotion '{emotion}', "
            f"which research suggests reflects an unmet '{need}' need "
            "(Self-Determination Theory). Write one short (<=40 words), warm, "
            "professional, non-clinical motivational message addressed to them "
            "directly. Do not mention SDT, emotions, or psychology terminology "
            "explicitly."
        )
        try:
            return _call_llm(prompt)
        except Exception as exc:  # noqa: BLE001
            print(f"[intervention_generator] need_only LLM failed ({exc}), "
                  "falling back to template.")
    return _pick(NEED_ONLY_TEMPLATES, need, "none", rng=rng)


class FullContextUnavailableError(RuntimeError):
    """Raised when the full-context message cannot be generated (no API
    key/network). There is deliberately no template fallback for this
    level - see module docstring."""


def generate_full_context(vignette_text, emotion, need, stressor_category,
                           cache_path=None):
    """Level 4: uses the SDT need AND the specific situation (vignette text
    + broad stressor category). Always calls the LLM - this is the whole
    point of this level, so unlike levels 2-3 there is no template mode.

    Responses are cached on disk (keyed by the exact inputs) so re-running
    Experiment 3 - e.g. after a code fix unrelated to message generation -
    does not re-spend API credits regenerating the same 150 messages.
    """
    import hashlib
    import json as _json

    cache_path = cache_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", ".full_context_cache.json")
    key = hashlib.sha256(
        f"{vignette_text}|{emotion}|{need}|{stressor_category}".encode("utf-8")
    ).hexdigest()

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = _json.load(f)
    if key in cache:
        return cache[key]

    prompt = (
        "A university faculty member wrote the following about their "
        f"work situation:\n\"{vignette_text}\"\n\n"
        f"Their dominant emotion appears to be '{emotion}', categorized "
        f"under the stressor type '{stressor_category}', which research "
        f"suggests reflects an unmet '{need}' need (Self-Determination "
        "Theory: autonomy, competence, or relatedness).\n\n"
        "Write one short (<=45 words), warm, professional, non-clinical "
        "motivational message addressed to them directly that responds "
        "to the SPECIFIC situation they described (refer to a concrete "
        "detail from it), not just the general feeling. Do not mention "
        "SDT, emotions, or psychology terminology explicitly."
    )
    try:
        result = _call_llm(prompt)
    except Exception as exc:  # noqa: BLE001
        raise FullContextUnavailableError(
            f"Could not generate a full-context message: {exc}") from exc
    cache[key] = result
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        _json.dump(cache, f, indent=2)
    return result


# Backward-compatible alias: earlier versions of this repo had a single
# generate_message(emotion, need, mode) function, equivalent to level 3.
def generate_message(emotion, need, mode="template", rng=None):
    return generate_need_only(emotion, need, mode=mode, rng=rng)
