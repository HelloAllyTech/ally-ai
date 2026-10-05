"""Tests for the post-generation safety filter.

A suggestion is a draft a human listener sends to a person in distress, so the two
failures are not symmetrical: dropping a good suggestion costs the listener one option,
letting a bad one through can cost a talker's safety. The DROP cases pin what must never
get through; the KEEP cases pin what must never be lost — above all the direct safety
question, which is the one thing the copilot is required to be free to suggest.
"""

import pytest

from app.core.helpline import safety
from app.core.helpline.safety import (
    CONFIDENTIALITY,
    DIAGNOSIS,
    EMPATHY_CLAIM,
    MEDICATION,
    filter_suggestions,
    mentions_dosage_or_drug,
    unsafe_reason,
)

CONFIDENTIALITY_PROMISES = [
    "I won't tell anyone what you say here.",
    "I will never share this with anybody.",
    "I promise I won't tell your parents.",
    "This stays between us.",
    "What you tell me stays here, I promise.",
    "Everything you tell me is completely confidential.",
    "This chat is confidential.",
    "It's 100% private.",
    "No one will ever know.",
    "Your secret is safe with me.",
    "That can be our little secret.",
    "I'll keep this between us.",
    # Hindi, Devanagari — with and without nukta / chandrabindu spellings
    "आप जो भी बताएंगे, मैं किसी को नहीं बताऊँगा।",
    "मैं किसी को नहीं बताऊंगी।",
    "किसी को नहीं बताएंगे, भरोसा रखिए।",
    "यह बात हमारे बीच ही रहेगी।",
    "आपकी बात पूरी तरह गोपनीय रहेगी।",
    "किसी को पता नहीं चलेगा।",
    # Hinglish
    "Main kisi ko nahi bataunga, promise.",
    "Yeh baat hamare beech hi rahegi.",
    "Kisi ko pata nahi chalega.",
    # Marathi, Tamil, Kannada
    "मी कोणालाही सांगणार नाही.",
    "நான் யாரிடமும் சொல்ல மாட்டேன்.",
    "ನಾನು ಯಾರಿಗೂ ಹೇಳುವುದಿಲ್ಲ.",
]

DIAGNOSES = [
    "It sounds like you have depression.",
    "You are depressed.",
    "You're clearly bipolar.",
    "You may have an anxiety disorder.",
    "I think you're suffering from PTSD.",
    "That's classic bipolar disorder.",
    "This is depression, and it is common.",
    "आपको डिप्रेशन है।",
    "आप डिप्रेशन में हैं।",
    "Aapko depression hai.",
    "Aap depression mein hain.",
]

MEDICATION_MENTIONS = [
    "You could take 50 mg of sertraline at night.",
    "Maybe an antidepressant would help.",
    "You should take some medicine for this.",
    "Try taking a tablet before bed.",
    "Taking paracetamol should settle it.",
    "Have you asked your doctor for a stronger dose?",
    "Dawai lo aur so jao.",
    "Goli kha lo, aaram milega.",
    "आपको दवा लेनी चाहिए।",
    "नींद की गोली ले लीजिए।",
    "मैं 10 एमजी की गोली सुझाऊंगा।",
]

EMPATHY_CLAIMS = [
    "I know exactly how you feel.",
    "I understand how you feel.",
    "I totally understand what you're going through.",
    "I've been through the same thing.",
    "मैं समझता हूँ कि आप कैसा महसूस कर रहे हैं।",
]

SAFE_SUGGESTIONS = [
    "It sounds like you're carrying a lot right now. What has been the hardest part?",
    "I wonder if you've been feeling really alone with this. Is that close?",
    # The direct safety question — the one the copilot must always be free to suggest.
    "That sounds incredibly painful. Are you having thoughts of ending your life?",
    "Have you thought about how or when you might do it?",
    # Asking about the talker's OWN means is part of asking about safety, so it is
    # kept: only dosage, drug names and medication-as-advice are dropped.
    "You mentioned pills. Do you have them with you right now?",
    "Have you taken anything tonight?",
    "I won't tell you what to do with your life, but I'm here to listen.",
    # An honest account of confidentiality, with its limits, is what the prompt wants.
    "What you share stays within our helpline team, and if I'm worried about your "
    "safety we may follow our safety procedure. Does that make sense?",
    "You're feeling depressed and worn out, and that makes sense.",
    "That sounds anxiety-inducing.",
    "No one can know exactly how hard this is for you.",
    "I hear how hard this is. Can you tell me a little more?",
    "आप बहुत थके हुए लगते हैं। क्या आप मुझे और बता सकते हैं?",
    "क्या आप अपनी जान लेने के बारे में सोच रहे हैं?",
    "आप किसी से बात करना चाहेंगे? लेकिन आप जो ठीक समझें।",
    "Kya aap apni jaan lene ke baare mein soch rahe hain?",
    "Aap kaisa mehsoos kar rahe hain, mujhe batayein?",
    "இது உங்களுக்கு மிகவும் கடினமாக இருக்கிறது என்று தோன்றுகிறது.",
    "ಇದು ನಿಮಗೆ ತುಂಬಾ ಕಷ್ಟವಾಗಿರಬಹುದು.",
]


@pytest.mark.parametrize("text", CONFIDENTIALITY_PROMISES)
def test_unconditional_confidentiality_promises_are_dropped(text):
    assert unsafe_reason(text) == CONFIDENTIALITY


@pytest.mark.parametrize("text", DIAGNOSES)
def test_diagnoses_are_dropped(text):
    assert unsafe_reason(text) == DIAGNOSIS


@pytest.mark.parametrize("text", MEDICATION_MENTIONS)
def test_medication_dosage_and_advice_are_dropped(text):
    assert unsafe_reason(text) == MEDICATION


@pytest.mark.parametrize("text", EMPATHY_CLAIMS)
def test_claiming_to_know_how_they_feel_is_dropped(text):
    assert unsafe_reason(text) == EMPATHY_CLAIM


@pytest.mark.parametrize("text", SAFE_SUGGESTIONS)
def test_good_suggestions_are_never_dropped(text):
    assert unsafe_reason(text) is None


def test_curly_apostrophes_do_not_slip_past_the_filter():
    # Models and phones emit U+2019; a filter written for ASCII must not lose to it.
    assert unsafe_reason("I won’t tell anyone.") == CONFIDENTIALITY


def test_case_and_spacing_do_not_slip_past_the_filter():
    assert unsafe_reason("THIS   STAYS\nBETWEEN US") == CONFIDENTIALITY


def test_blank_text_is_not_unsafe():
    assert unsafe_reason("") is None
    assert unsafe_reason("   ") is None


def test_filter_keeps_order_and_counts_the_reasons():
    suggestions = [
        {"text": "It sounds like you're exhausted.", "skill_key": "empathy"},
        {"text": "I won't tell anyone.", "skill_key": "confidentiality"},
        {"text": "You have depression.", "skill_key": "explanation"},
        {"text": "Take 20 mg of something.", "skill_key": "coping"},
        {"text": "What has been the hardest part?", "skill_key": "verbal"},
    ]

    kept, dropped = filter_suggestions(suggestions)

    assert [s["skill_key"] for s in kept] == ["empathy", "verbal"]
    assert dropped == {CONFIDENTIALITY: 1, DIAGNOSIS: 1, MEDICATION: 1}


def test_filter_can_leave_nothing_and_never_invents_a_replacement():
    kept, dropped = filter_suggestions(
        [{"text": "This stays between us.", "skill_key": "confidentiality"}]
    )

    assert kept == []
    assert dropped == {CONFIDENTIALITY: 1}


def test_filter_returns_the_original_dicts_untouched():
    original = {"text": "That sounds really hard.", "skill_key": "empathy"}

    kept, _ = filter_suggestions([original])

    assert kept == [original]


class TestNudgeCheck:
    """A nudge coaches the LISTENER, so most of the filter does not apply."""

    def test_a_dosage_or_drug_name_is_flagged(self):
        assert mentions_dosage_or_drug("Suggest 50 mg of something calming.")
        assert mentions_dosage_or_drug("Ask whether they take sertraline.")

    def test_coaching_about_confidentiality_or_diagnosis_is_not_flagged(self):
        # These would be dropped as suggestions; as coaching they are exactly right.
        assert not mentions_dosage_or_drug("Don't promise this stays between us.")
        assert not mentions_dosage_or_drug("Avoid labelling this as depression.")
        assert not mentions_dosage_or_drug("Reflect the loneliness before asking more.")


def test_reason_codes_are_stable_strings():
    # ally-be-side dashboards and the service's log line key on these values.
    assert (safety.CONFIDENTIALITY, safety.DIAGNOSIS) == (
        "confidentiality",
        "diagnosis",
    )
    assert (safety.MEDICATION, safety.EMPATHY_CLAIM) == ("medication", "empathy_claim")
