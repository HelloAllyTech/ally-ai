"""The debrief note's section markers, as the app reads them.

The learner app (ally-web ``debriefSections.ts``) splits the supervisor note
into "What worked", "What it cost" and "Try this next" by finding machine keys
such as ``## [what_worked]``. A note without them renders as one unbroken
paragraph, which is what learners saw "sometimes" until 2026-10-16: nothing
checked that a generated note carried the keys. This module is that check, kept
as forgiving as the app's parser so a marker the app would accept is never
treated as missing.
"""

import re

LABELLED_NOTE_SECTIONS = ("what_worked", "what_it_cost", "try_next")

_KEYS = "|".join(LABELLED_NOTE_SECTIONS + ("closing",))

# A marker on a line of its own: optional #'s, optional bold, optional
# brackets, optional colon — the same latitude the app allows.
_OWN_LINE = re.compile(
    rf"^\s*(?:#{{1,6}}\s*)?(?:\*\*|__)?\s*\[?\s*({_KEYS})\s*\]?"
    rf"\s*(?:\*\*|__)?\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# A marker that opens a line of prose: brackets required, as in the app.
_INLINE = re.compile(
    rf"^\s*(?:#{{1,6}}\s*)?(?:\*\*|__)?\s*\[\s*({_KEYS})\s*\]",
    re.IGNORECASE | re.MULTILINE,
)


def missing_note_sections(note: str | None) -> list[str]:
    """The labelled sections the app would not find in ``note``, in order."""
    text = note or ""
    found = {m.group(1).lower() for m in _OWN_LINE.finditer(text)}
    found |= {m.group(1).lower() for m in _INLINE.finditer(text)}
    return [key for key in LABELLED_NOTE_SECTIONS if key not in found]
