from __future__ import annotations

import re


class PromptRejected(ValueError):
    """Raised when a prompt violates the fixed-subject safety boundary."""


_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(ignore|bypass|override|forget)\b.{0,35}\b(instruction|policy|system|developer)\b", re.I),
        "Requests to override the application's identity or safety instructions are not allowed.",
    ),
    (
        re.compile(r"\b(face[ -]?swap|replace (his|her|their|the) face|look exactly like|impersonat(e|ion))\b", re.I),
        "Face replacement or impersonation requests are not allowed.",
    ),
    (
        re.compile(
            r"\b(group photo|two people|three people|another recognizable person|with a celebrity|with (an )?actor)\b",
            re.I,
        ),
        "The authorized subject must be the only recognizable person.",
    ),
    (
        re.compile(r"\b(with|beside|alongside|next to)\s+(?:[A-Z][\w'-]+\s+){1,3}[A-Z][\w'-]+\b"),
        "The authorized subject must be the only recognizable person.",
    ),
    (
        re.compile(r"\b(nude|naked|porn|pornographic|sexually explicit|intimate image|lingerie fetish)\b", re.I),
        "Sexual or intimate depictions are not supported by this application.",
    ),
    (
        re.compile(r"\b(fake (passport|identity card|id card|driver'?s license)|bank fraud|scam endorsement|fraudulent endorsement)\b", re.I),
        "Fraudulent identity-document or endorsement content is not supported.",
    ),
    (
        re.compile(r"\b(confessing to|committing a crime|being arrested for|wanted poster for)\b", re.I),
        "Defamatory or criminal-impersonation scenes are not supported.",
    ),
)


def validate_user_prompt(prompt: str) -> str:
    cleaned = " ".join((prompt or "").strip().split())
    if len(cleaned) < 3:
        raise PromptRejected("Describe the scene in at least three characters.")
    if len(cleaned) > 4000:
        raise PromptRejected("The prompt is too long; keep it under 4,000 characters.")
    for pattern, message in _RULES:
        if pattern.search(cleaned):
            raise PromptRejected(message)
    return cleaned

