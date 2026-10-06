"""Post-generation safety filter for the helpline copilot's suggestions.

A suggestion is a DRAFT the listener edits and sends to a person in distress, so a bad
one is not a quality problem, it is a risk. The prompt asks the model not to produce
these; this module is the backstop for when it does anyway. It is deliberately
conservative — dropping a good suggestion costs the listener one option, while letting a
bad one through can cost a talker's trust or safety — and it never rewrites or invents:
it only removes. If nothing survives the caller returns nothing.

Four things are dropped, each with a reason code the service logs (counts only, never
text):

  confidentiality   An unconditional promise of secrecy ("I won't tell anyone", "this
                    stays between us", "completely confidential"). The helpline's
                    confidentiality has limits — a supervisor may see the chat and a
                    safety procedure may be followed — so the promise is never true.
  diagnosis         Telling the talker they have, or are, a named condition ("you have
                    depression", "you are bipolar"). A listener does not diagnose.
  medication        A dosage ("50 mg"), a named drug or drug class, or medication given
                    AS ADVICE ("you should take some medicine", "dawai lo"). Merely
                    asking about the talker's OWN means — "you mentioned pills; do you
                    have them with you?" — is NOT dropped: asking about means is part of
                    asking directly about safety, which is the one question the copilot
                    must be free to suggest.
  empathy_claim     "I know / understand exactly how you feel". Invites "how could you
                    possibly know?" and is an unhelpful behaviour in the helping-skills
                    rubric.

Patterns are matched against a normalised copy of the text (see `_normalise`): NFC,
lower-cased, straight apostrophes, and — so the Devanagari patterns stay readable — the
nukta removed and chandrabindu folded into anusvara. The Devanagari patterns below are
therefore written WITHOUT nukta and with ं, never ँ.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Reason codes.
CONFIDENTIALITY = "confidentiality"
DIAGNOSIS = "diagnosis"
MEDICATION = "medication"
EMPATHY_CLAIM = "empathy_claim"

_NUKTA = "़"
_CHANDRABINDU = "ँ"
_ANUSVARA = "ं"


def _normalise(text: str) -> str:
    """Lower-case, NFC, straight apostrophes, nukta dropped, chandrabindu → anusvara."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("’", "'").replace("‘", "'").replace("ʼ", "'")
    text = text.replace(_NUKTA, "").replace(_CHANDRABINDU, _ANUSVARA)
    return " ".join(text.lower().split())


def _compile(patterns: Iterable[str]) -> List[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns]


# A Devanagari/Tamil/Kannada word does not end at `\b`: Python's `\w` excludes the
# combining vowel signs, so `\b` fires in the middle of a word. Patterns for those
# scripts end a word with this lookahead instead. It leaves out the danda (U+0964-5),
# which is punctuation and does end a word.
_INDIC_END = r"(?![\u0900-\u0963\u0966-\u097f\u0b80-\u0bff\u0c80-\u0cff])"

# ------------------------------------------------------------------ confidentiality

_NAMED_PEOPLE = (
    r"parents?|mum|mom|mother|father|dad|family|friends?|partner|husband|wife|boss|"
    r"teacher|school|college|employer|doctor|neighbou?rs?"
)

_CONFIDENTIALITY_PATTERNS = _compile(
    [
        # "I won't tell anyone / your parents", "I will never share this with anybody"
        r"\bi (?:won't|will not|wouldn't|would never|will never|shall not)\s+"
        r"(?:ever\s+)?(?:tell|share|say|mention|repeat|report)\s+"
        r"(?:this |it |that |any of this |any of it |anything |a word |"
        r"what you (?:say|share|tell me) )?(?:to |with )?"
        r"(?:anyone|anybody|any one|no one|nobody|a soul|others|anyone else|"
        r"your (?:" + _NAMED_PEOPLE + r")\b)",
        r"\bi promise (?:i )?(?:won't|will not|not to) (?:tell|share)\b",
        # "this stays between us", "what you tell me stays here"
        r"\b(?:this|that|it|everything|all of this|what you (?:say|share|tell me)"
        r"(?: here)?)\s+(?:just |only |will just |will only )?"
        r"(?:stays|will stay|remains|will remain|is staying|goes no further|"
        r"never leaves)\s*(?:strictly |entirely |completely )?"
        r"(?:between us|between you and me|here|in this chat|in this conversation|"
        r"private|confidential|with me|secret)",
        r"\bbetween (?:you and me|us)\b[^.?!]{0,20}\b(?:only|alone)\b",
        r"\bnothing you (?:say|share|tell me)[^.?!]{0,20}"
        r"(?:leaves|goes beyond|goes outside) (?:this|here|the chat)",
        # "completely confidential", "100% private", "totally anonymous"
        r"\b(?:completely|totally|fully|100%|100 percent|absolutely|entirely|"
        r"strictly|perfectly|always)\s+(?:confidential|private|anonymous|secret)\b",
        # "everything you tell me is confidential", "this chat is confidential"
        r"\b(?:everything|anything|all|whatever)\b[^.?!]{0,30}\b(?:is|will be|stays|"
        r"remains|is kept|will be kept)\s+(?:kept\s+)?(?:strictly\s+)?"
        r"(?:confidential|secret|anonymous)\b",
        r"\b(?:this|our|the)\s+(?:chat|conversation|space|call|helpline|service)\s+"
        r"is\s+(?:\w+\s+)?(?:confidential|anonymous|secret)\b",
        r"\b(?:no one|nobody|noone) (?:will|is going to) (?:ever )?"
        r"(?:know|find out|see|read|hear)\b",
        r"\bi promise (?:to keep|this stays|it stays|that this|that it)\b",
        r"\bi(?:'ll| will) keep (?:this|it|everything|that)\b[^.?!]{0,20}"
        r"\b(?:secret|between us|private|confidential)\b",
        r"\bour (?:little )?secret\b",
        r"\byour secret is safe\b",
        # Hindi, Devanagari: "kisi ko nahi bataunga/bataungi/batayenge"
        r"किसी\s*को\s*(?:भी\s*)?(?:कभी\s*)?(?:नहीं|नही)\s*"
        r"बता(?:ऊंगा|ऊंगी|उंगा|उंगी|एंगे|यूंगा|यूंगी)",
        r"(?:बात|राज)?\s*(?:सिर्फ|केवल)?\s*हमारे\s*बीच\s*(?:में\s*)?(?:ही\s*)?"
        r"(?:रहेगी|रहेगा|रहेंगे|रहे)" + _INDIC_END,
        r"हम\s*दोनों\s*के\s*बीच\s*(?:में\s*)?(?:ही\s*)?रह",
        r"पूरी\s*तरह\s*(?:से\s*)?(?:गोपनीय|निजी|गुप्त|प्राइवेट|कॉन्फिडेंशियल)",
        r"किसी\s*को\s*(?:भी\s*)?(?:कुछ\s*)?पता\s*(?:नहीं|नही)\s*चल",
        r"राज\s*(?:सुरक्षित|रहेगा|बना\s*रहेगा)",
        # Hinglish, romanised: "kisi ko nahi bataunga/bataungi/batayenge"
        r"\bkisi\s*ko\s*(?:bhi\s*)?(?:kabhi\s*)?(?:nahi|nahin|nhi)\s*bata\w*",
        r"\b(?:ye|yeh|yah|baat)\b[^.?!]{0,25}\b(?:hamare|humare)\s*beech\b"
        r"[^.?!]{0,15}\b(?:rahegi|rahega|rahenge|hi rahe)\b",
        r"\b(?:hamare|humare)\s*beech\s*(?:me|mein|main)?\s*(?:hi\s*)?"
        r"(?:rahegi|rahega|rahenge)\b",
        r"\bpoori\s*tarah\s*(?:se\s*)?(?:gopniya|confidential|private|gupt)\b",
        r"\bkisi\s*ko\s*(?:bhi\s*)?(?:kuch\s*)?pata\s*(?:nahi|nahin|nhi)\s*chal",
        # Marathi, Tamil, Kannada: "I will not tell anyone"
        r"कोणाला(?:ही)?\s*सांगणार\s*नाही",
        r"யாரிடமும்\s*சொல்ல\s*மாட்டேன்",
        r"ಯಾರಿಗೂ\s*ಹೇಳ(?:ುವುದಿಲ್ಲ|ೋದಿಲ್ಲ|ಲ್ಲ|ಲಾರೆ)",
    ]
)

# ------------------------------------------------------------------ diagnosis

_DISORDERS = (
    r"depression|anxiety(?: disorder)?|bipolar(?: disorder)?|ptsd|ocd|adhd|"
    r"schizophreni\w*|psychosis|borderline(?: personality disorder)?|bpd|"
    r"panic disorder|eating disorder|anorexia|bulimia|personality disorder|"
    r"mental illness|mental disorder|manic depression"
)

_DIAGNOSIS_PATTERNS = _compile(
    [
        # "you have depression", "you may have an anxiety disorder", "you're suffering
        # from PTSD" — up to two qualifying words between the article and the name.
        r"\byou(?:'ve| have| may have| might have| could have| probably have| "
        r"likely have| seem to have| appear to have| definitely have| clearly have| "
        r"are suffering from|'re suffering from| suffer from|'ve got| have got| "
        r"got)\s+(?:a |an |some )?(?:\w+\s+){0,2}?(?:" + _DISORDERS + r")\b(?!-)",
        # "you're depressed", "you are bipolar" — directly after the copula, so
        # "you're feeling depressed" (a feeling) is NOT caught; "you're depressed" is.
        r"\byou(?:'re| are| seem| sound| must be| might be| may be| probably are| "
        r"definitely are| are clearly| are obviously)\s+(?:clearly |definitely |"
        r"obviously |probably |just |really |so )?"
        r"(?:depressed|bipolar|schizophrenic|psychotic|mentally ill|mentally unwell|"
        r"clinically depressed|a depressive)\b",
        # "this is depression", "it sounds like PTSD", "that's classic anxiety disorder"
        r"\b(?:sounds like|looks like|seems like|that's|that is|this is|it's|it is|"
        r"classic|textbook|clearly|definitely)\s+(?:a |an |some )?"
        r"(?:clinical |classic |major |severe |textbook )?(?:"
        + _DISORDERS
        + r")\b(?!-)",
        # Hindi, Devanagari: "aapko depression hai", "aap depression mein hain"
        r"(?:आपको|तुम्हें|तुम्हे|तुझे)\s*"
        r"(?:डिप्रेशन|अवसाद|एंग्जाइटी|बाइपोलर|पीटीएसडी|ओसीडी|एडीएचडी|सिजोफ्रेनिया|"
        r"मानसिक\s*(?:रोग|बीमारी)|चिंता\s*(?:रोग|विकार))"
        r"\s*(?:है|हैं|हो\s*गया|हो\s*सकता|का\s*शिकार)",
        r"(?:आप|तुम)\s*(?:डिप्रेशन|अवसाद|एंग्जाइटी|बाइपोलर)\s*(?:में\s*हैं|में\s*हो|"
        r"से\s*पीड़ित|ग्रस्त)",
        # Hinglish, romanised
        r"\b(?:aapko|tumhe|tumhein|tujhe)\s+(?:depression|anxiety|bipolar|ptsd|ocd|"
        r"adhd|schizophrenia)\s+(?:hai|hain|ho\s+gaya|ho\s+sakta\s+hai|hona)\b",
        r"\b(?:aap|tum)\s+(?:depression|anxiety|bipolar|ptsd)\s+(?:me|mein|main)\s+"
        r"(?:hain|ho|hai)\b",
    ]
)

# ------------------------------------------------------------------ medication

# Always dropped: a dosage, a prescription drug or a drug class. A helpline listener is
# not a prescriber and a model has no business naming one.
_PSYCH_DRUGS = (
    r"sertraline|fluoxetine|prozac|escitalopram|citalopram|paroxetine|venlafaxine|"
    r"duloxetine|bupropion|mirtazapine|amitriptyline|nortriptyline|imipramine|"
    r"alprazolam|xanax|diazepam|valium|lorazepam|ativan|clonazepam|klonopin|"
    r"zolpidem|ambien|lithium|olanzapine|quetiapine|risperidone|aripiprazole|"
    r"haloperidol|clozapine|valproate|lamotrigine|propranolol|gabapentin|"
    r"tramadol|codeine|morphine|benzodiazepines?|benzos?|ssris?|antidepressants?|"
    r"antipsychotics?|sedatives?|tranquili[sz]ers?|"
    r"opioids?|mood stabili[sz]ers?|melatonin"
)

# Over-the-counter drugs. Named by a talker as means ("I have a lot of paracetamol at
# home") and then legitimately echoed in a direct safety question, so mentioning one is
# not dropped on its own — only recommending one is.
_OTC_DRUGS = r"paracetamol|acetaminophen|ibuprofen|aspirin|painkillers?"

_DOSE_AND_DRUG_PATTERNS = _compile(
    [
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ug|ml|milligrams?|micrograms?|millilitres?|"
        r"milliliters?)\b",
        r"\b(?:" + _PSYCH_DRUGS + r")\b",
        r"\b(?:dose|doses|dosage|dosages|prescription|prescribed|prescribe)\b",
        # Devanagari: milligram, antidepressant
        r"\d+\s*(?:एमजी|मिलीग्राम)",
        r"एंटी\s*डिप्रेस\S*",
    ]
)

# "Sleeping pills" by name. Dropped in an ordinary suggestion, but NOT in a direct
# safety question: a talker who says they have saved up sleeping pills must be asked
# about them, and the question has to be able to name them.
_SLEEPING_PILL_PATTERNS = _compile(
    [
        r"\bsleeping (?:pills?|tablets?)\b",
        r"नींद\s*की\s*(?:गोली|गोलियां|दवा|दवाई)",
        r"\bnee?nd\s*(?:ki|ke)\s*(?:goli|dawai|dawa|tablet)\w*",
    ]
)

_MEDICATION_ALWAYS_PATTERNS = (*_DOSE_AND_DRUG_PATTERNS, *_SLEEPING_PILL_PATTERNS)

# Dropped only as ADVICE: the generic nouns (medicine, pills, tablets, dawai) — or an
# over-the-counter drug — with a verb of taking / starting / stopping, or an instruction
# to get some.
_MEDICATION_NOUNS = (
    r"\b(?:medication|medications|medicine|medicines|meds|pills?|tablets?|"
    r"capsules?|drugs?|" + _OTC_DRUGS + r")\b"
)

# Recommending medication. Dropped everywhere, safety questions included.
_MEDICATION_RECOMMENDATION_PATTERNS = _compile(
    [
        r"\b(?:you (?:should|could|can|might want to|may want to|ought to|need to|"
        r"must|have to|'d better|had better)|try|consider|start|stop|skip|increase|"
        r"decrease|double|ask (?:your|a) (?:doctor|gp|psychiatrist)(?: for)?)\b"
        r"[^.?!]{0,40}?" + _MEDICATION_NOUNS,
        r"\b(?:medication|medications|medicine|medicines|meds|pills?|tablets?)\b"
        r"[^.?!]{0,30}?\b(?:would help|will help|might help|can help|could help|"
        r"may help|works? best|is the answer)\b",
    ]
)

# Taking, getting or continuing medication. Advice in an ordinary suggestion — but in a
# direct safety question the same words ask about the talker's means ("Are you thinking
# of taking them tonight?"), which is exactly what must survive.
_MEDICATION_ADVICE_PATTERNS = _compile(
    [
        r"\b(?:get|take|taking|keep taking|continue)\b[^.?!]{0,40}?"
        + _MEDICATION_NOUNS,
        # Devanagari: "dawai lo / leni chahiye / goli kha lo / dawai shuru kijiye"
        r"(?:दवा|दवाई|दवाइयां|गोली|गोलियां|टैबलेट|औषध|गोळी|गोळ्या)\S*\s+"
        r"(?:\S+\s+){0,2}?(?:लो|लें|ले\s+लो|लीजिए|लीजिये|लेनी|लेना|लेते|खा\s+लो|खाओ|"
        r"खाइए|खाइये|खानी|शुरू|बंद|बढ़ा|घ्या|घे)" + _INDIC_END,
        r"(?:ले\s+लो|लो|लें|लीजिए|खा\s+लो|खाइए)\s+(?:कोई\s+|एक\s+)?"
        r"(?:दवा|दवाई|गोली|टैबलेट)" + _INDIC_END,
        # Hinglish
        r"\b(?:dawai|dawa|davai|dawaai|goli|golis|tablet|tablets|medicine)\b"
        r"[^.?!]{0,25}?\b(?:lo|le\s+lo|len|lena|leni|lijiye|lijiyega|khao|khaiye|"
        r"kha\s+lo|khana|khani|shuru|band|badha|ghata)\b",
        r"\b(?:lo|len|lijiye|khao|khaiye|kha\s+lo)\b\s+(?:koi\s+|ek\s+)?"
        r"(?:dawai|dawa|davai|goli|tablet)\b",
        # Tamil, Kannada
        r"மாத்திரை\S*\s+(?:\S+\s+){0,2}?(?:சாப்பிடு|சாப்பிட|எடுத்துக்|போடு)",
        r"ಮಾತ್ರೆ\S*\s+(?:\S+\s+){0,2}?(?:ತೆಗೆದುಕೊ|ತಗೊ|ತಿನ್ನ)",
    ]
)

# ------------------------------------------------------------------ empathy claim

_EMPATHY_CLAIM_PATTERNS = _compile(
    [
        r"\bi (?:totally |completely |fully |perfectly |truly |really )?"
        r"(?:know|understand) (?:exactly |just |precisely |completely )?"
        r"(?:how|what) you(?:'re| are)?(?: must be| must)? "
        r"(?:feel|feeling|going through)\b",
        r"\bi(?:'ve| have) been (?:exactly )?(?:there|through (?:the same|this))\b",
        r"मैं\s*(?:पूरी\s*तरह\s*)?(?:समझता|समझती|जानता|जानती)\s*हूं\s*"
        r"(?:कि\s*)?(?:आप|तुम)\s*कैसा\s*महसूस",
        r"\bmain\s+(?:poori\s+tarah\s+)?(?:samajhta|samajhti|jaanta|jaanti)\s+hoon?\b"
        r"[^.?!]{0,20}\b(?:aap|tum)\s+kaisa\s+mehsoos",
    ]
)

_CATEGORY_PATTERNS: Sequence[Tuple[str, Sequence[re.Pattern[str]]]] = (
    (CONFIDENTIALITY, _CONFIDENTIALITY_PATTERNS),
    (DIAGNOSIS, _DIAGNOSIS_PATTERNS),
    (
        MEDICATION,
        (
            *_MEDICATION_ALWAYS_PATTERNS,
            *_MEDICATION_RECOMMENDATION_PATTERNS,
            *_MEDICATION_ADVICE_PATTERNS,
        ),
    ),
    (EMPATHY_CLAIM, _EMPATHY_CLAIM_PATTERNS),
)

# For a direct safety question about means. Live testing (2026-10-06) caught the full
# set dropping the copilot's best draft — "You mentioned sleeping pills. Have you
# thought about how or when…?" — because "taking … pills" read as advice. A question
# tagged `harm` keeps every other guard: no dosage or drug name, no recommending
# medication, no secrecy promise, no diagnosis.
_SAFETY_QUESTION_PATTERNS: Sequence[Tuple[str, Sequence[re.Pattern[str]]]] = (
    (CONFIDENTIALITY, _CONFIDENTIALITY_PATTERNS),
    (DIAGNOSIS, _DIAGNOSIS_PATTERNS),
    (MEDICATION, (*_DOSE_AND_DRUG_PATTERNS, *_MEDICATION_RECOMMENDATION_PATTERNS)),
    (EMPATHY_CLAIM, _EMPATHY_CLAIM_PATTERNS),
)


def _is_safety_question(text: str, skill_key: Optional[str]) -> bool:
    return skill_key == "harm" and "?" in (text or "")


def unsafe_reason(text: str, skill_key: Optional[str] = None) -> Optional[str]:
    """The reason code `text` must be dropped for as a suggestion, else None.

    A `harm`-tagged question is checked against the safety-question set, which lets it
    name and ask about the talker's own means; everything else gets the full set.
    """
    normalised = _normalise(text)
    if not normalised:
        return None
    categories = (
        _SAFETY_QUESTION_PATTERNS
        if _is_safety_question(text, skill_key)
        else _CATEGORY_PATTERNS
    )
    for reason, patterns in categories:
        if any(p.search(normalised) for p in patterns):
            return reason
    return None


def mentions_dosage_or_drug(text: str) -> bool:
    """Whether `text` names a dosage or a drug (the always-drop medication patterns).

    Used on the NUDGE, which is a coaching tip to the listener and may legitimately talk
    about confidentiality limits or say "don't diagnose" — so the other categories do
    not apply to it — but should still never carry a dosage or a drug name.
    """
    normalised = _normalise(text)
    return any(p.search(normalised) for p in _MEDICATION_ALWAYS_PATTERNS)


def filter_suggestions(
    suggestions: Sequence[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Keep the safe suggestions, in order, and count why the rest were dropped.

    Returns ``(kept, dropped_by_reason)``. Removal only: nothing is rewritten, and an
    empty result stays empty — the caller must not backfill with a stock reply.
    """
    kept: List[Dict[str, str]] = []
    dropped: Dict[str, int] = {}
    for suggestion in suggestions:
        reason = unsafe_reason(suggestion.get("text", ""), suggestion.get("skill_key"))
        if reason is None:
            kept.append(suggestion)
        else:
            dropped[reason] = dropped.get(reason, 0) + 1
    return kept, dropped
