"""
Maps a detected emotion to a Self-Determination Theory (SDT) need category.

SDT (Deci & Ryan) proposes three basic psychological needs: autonomy,
competence, and relatedness. Frustration of a specific need tends to show
up as a specific emotional signature; this mapping operationalises that
link for the purpose of routing an automatic intervention. It is a
literature-informed proposal, not an empirically validated instrument: it
should ideally be reviewed by domain experts (e.g. organisational
psychologists) before being used with real faculty.

Rationale per emotion:
- anger    -> autonomy:     anger commonly follows a perceived unfair
                             constraint or loss of control over one's work.
- disgust  -> autonomy:     disgust at policies/practices often reflects a
                             values-vs-autonomy conflict.
- fear     -> competence:   work-related fear/anxiety commonly reflects a
                             perceived threat to one's competence (e.g.
                             evaluation, failure, being replaced).
- sadness  -> relatedness:  sadness at work is commonly linked to social
                             disconnection or lack of recognition.
- joy      -> reinforcement: needs already appear met; the right action is
                             to reinforce, not to "fix" anything.
- surprise -> competence:   often follows unexpected feedback, which is
                             most related to competence/clarity needs.
- neutral  -> none:         no clear emotional signal was detected.
"""

EMOTION_TO_NEED = {
    "anger": "autonomy",
    "disgust": "autonomy",
    "fear": "competence",
    "sadness": "relatedness",
    "joy": "reinforcement",
    "surprise": "competence",
    "neutral": "none",
    # ISEAR-only labels, mapped for completeness / future extensions
    "shame": "relatedness",
    "guilt": "relatedness",
}


def get_need(emotion):
    """Return the SDT need category for a given emotion label.

    Falls back to "none" for any emotion not in the table, rather than
    raising, since this function is called inside batch pipelines.
    """
    return EMOTION_TO_NEED.get(emotion, "none")
