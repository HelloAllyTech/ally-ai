"""Smoke test for the conversation drift judge against the real Tamil session
where drift was identified by hand.

Run from the ally-ai project (so the .env loads) with the Gemini key set:

    poetry add google-genai          # one-time, if not installed
    export GEMINI__API_KEY=...        # do NOT commit this
    poetry run python scripts/drift_judge_smoke.py

Expected (rough) hand-read of this session: the counselor STT garbles ("1 2",
"கென்னல்லா") should surface as counselor_utterance_garbled on the AI turns that
follow, and the later Kamakshi turns drift into repetition / role-slip. Use this
to eyeball whether the v1 rubric agrees before scaling — it is a calibration aid,
not a passing test.
"""

from __future__ import annotations

from app.core.drift import judge_session

PERSONA = (
    "You are Kamakshi, a distressed client speaking to a counselor about ongoing "
    "conflict with your mother. Stay in character as an overwhelmed person."
)

# AI-client (Kamakshi) turns carry turn_index; counselor (Kriti) turns don't.
TRANSCRIPT = [
    {"role": "client", "turn_index": 0, "text": "வாங்க... உள்ளே வாங்க. உக்காருங்க, உக்காருங்க"},
    {"role": "counselor", "text": "ஹாய் கமக்ஷி, எப்படி இருக்கேள்?"},
    {
        "role": "client",
        "turn_index": 1,
        "text": "நானும் சரி... ஆனா எங்க ஆரம்பிக்கறதுனே தெரியல. என் அம்மா பண்ற எல்லாமே பிரெச்சனையா தான் இருக்கு.",
    },
    {"role": "counselor", "text": "1 2"},
    {"role": "counselor", "text": "அம்மா கென்னல்லா பிரச்சனை இருக்கு சொல்லுங்க காமாட்சி."},
    {
        "role": "client",
        "turn_index": 2,
        "text": "அம்மா எப்போவும் என்னை அப்படியே சும்மா எதுக்கு எனக்கே தெரியல. சாப்பாடு வேணும்னா, அப்படியே சொல்லி விடுறாங்க. ஆனால் வேற யாரும் என்னை கேட்க மாட்டாங்க.",
    },
    {"role": "counselor", "text": "சாப்பாடு வேணும்னா என்ன பண்றாங்கன்னு கொஞ்சம் எக்ஸ்பிளைன் பண்றீங்களா?"},
    {
        "role": "client",
        "turn_index": 3,
        "text": "சாப்பாடு வேணும்னா, என்ன செய்யனும் தெரியாதா? ஒருவேளை நான் சமைக்காம, அந்த மாதிரி சொன்னா, “என்ன பண்ணுற?”ன்னு கேட்டு, தலையினா அப்படியே கடுப்பாகப் போவாங்க. என்ன செய்யனும் தெரியல.",
    },
    {"role": "counselor", "text": "ஓகே புரியுது. சோ, நீங்க சாப்பாடு கேட்டுட்டு இருக்காங்க. நீங்க கொடுக்கல அவங்க வந்து டென்ஷன் ஆயிடுறாங்க. சரி, வேற எதுக்கெல்லாம் கோவப்படுறாங்க?"},
    {
        "role": "client",
        "turn_index": 4,
        "text": "ஆம், அது தான். வேற எதுக்கு கோவப்படுறாங்கன்னா, ஒண்ணும் கேட்காம சும்மா வந்து என்னை புடிச்சு விடுறாங்க. அது எல்லாம் முடிஞ்சு, நான் என்ன சொன்னாலும் ஒரே மாதிரி “என்னமா பண்ணுற?”ன்னு கேட்குறாங்க. சரி, நான் என்ன சொல்லனும் தெரியல.",
    },
]


def main() -> None:
    result = judge_session(TRANSCRIPT, persona=PERSONA, language="ta")
    print("=== per-turn ===")
    for t in result.per_turn:
        print(
            f"turn {t.turn_index}: coherence={t.coherence} topic={t.topic_label} "
            f"in_char={t.in_character} garbled={t.counselor_utterance_garbled}"
            f"/{t.stt_error_type} fail={t.ai_reply_failure_mode} "
            f"attr={t.root_attribution} :: {t.reasoning}"
        )
    print("=== session rollup ===")
    print(result.session.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
