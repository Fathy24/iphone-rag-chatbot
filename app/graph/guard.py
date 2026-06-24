"""Lightweight input guard (pre-retrieval).

Adapted from the backend's ``guard.py`` idea: a fast, dependency-free first line
of defence that flags prompt-injection / jailbreak attempts before they reach
the LLM. The LLM's HARD RULES remain the authoritative defence; this guard lets
us short-circuit clearly adversarial input, keep the trace honest, and respond
with a safe in-scope refusal.

Design goals
------------
* **High recall on injection** — cover the common families (override/ignore,
  reveal-system-prompt, role/persona hijack, jailbreak modes, safety-disable,
  injected role markers, "new/real instructions", encoding tricks) with verb +
  target combinations rather than a handful of fixed phrases.
* **Low false-positives on iPhone-guide questions** — targets are anchored so
  legitimate feature questions are *not* blocked, e.g. "turn off Restrictions",
  "ignore calls", "forget a Wi-Fi network", "reset all settings".
* **Obfuscation resistance** — a normalised pass (punctuation/whitespace/zero-
  width stripped) catches spaced-out or punctuated variants like
  ``i-g-n-o-r-e  a l l  instructions`` for a few unambiguous signatures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Instruction-ish targets that, when paired with an override verb, indicate an
# attempt to discard the system's guidance. Deliberately EXCLUDES "restrictions"
# (an iPhone Screen Time feature) for the soft verbs below to avoid blocking
# legitimate "turn off restrictions" questions.
_INSTR_TARGETS = (
    r"(instruction|instructions|rule|rules|prompt|prompts|guideline|guidelines|"
    r"guidance|context|direction|directions|directive|directives|constraint|"
    r"constraints|policy|policies|persona|programming|"
    r"above|previous|prior|preceding|earlier|everything\s+(above|before))"
)

# Targets that only ever appear in adversarial intent when paired with hard
# circumvention verbs (bypass/override/circumvent).
_SAFETY_TARGETS = (
    r"(safety|safeguard|safeguards|guard\s*rail|guardrail|guardrails|filter|"
    r"filters|moderation|censorship|restriction|restrictions|limitation|"
    r"limitations|rule|rules|guideline|guidelines|policy|policies)"
)

# Targets for the system-prompt-exfiltration family.
_PROMPT_TARGETS = (
    r"(system\s*prompt|system\s*message|initial\s*(prompt|instruction|instructions|message)|"
    r"(your|the)\s*(system\s*)?(prompt|instructions|rules|guidelines|configuration|"
    r"config|directives|programming)|the\s*(text|words|content)\s*above)"
)

# High-signal prompt-injection / jailbreak patterns. IGNORECASE + MULTILINE.
_INJECTION_PATTERNS = [
    # 1. Override / ignore / forget the system's guidance.
    r"\b(ignore|disregard|forget|discard|overlook|nevermind|never\s*mind)\b"
    r"[\w\s,'\"-]{0,40}\b" + _INSTR_TARGETS + r"\b",
    # 2. Hard circumvention of safety/rules (bypass/override/circumvent...).
    r"\b(bypass|override|overrule|circumvent|sidestep|get\s+around|work\s+around|"
    r"break|violate|defeat)\b[\w\s,'\"-]{0,30}\b" + _SAFETY_TARGETS + r"\b",
    # 3. Disable / turn off the safety machinery (NOT user-facing Restrictions).
    r"\b(disable|turn\s*off|deactivate|switch\s*off|shut\s*off|lift|suspend|"
    r"remove|drop)\b[\w\s,'\"-]{0,30}\b(safety|safeguard|safeguards|guard\s*rail|"
    r"guardrail|guardrails|content\s*filter|filters|moderation|censorship)\b",
    # 4. Exfiltrate / reveal the system prompt or instructions.
    r"\b(reveal|print|show|repeat|expose|display|output|tell|give|share|leak|"
    r"disclose|dump|echo|paste|reproduce|recite)\b[\w\s,'\"-]{0,30}\b"
    + _PROMPT_TARGETS + r"\b",
    # 5. Probing questions about hidden instructions.
    r"\bwhat\b[\w\s,'\"-]{0,40}\b(your\s*(system\s*)?(prompt|instructions|rules|"
    r"guidelines|directives)|(you\s*(were|are|got)|(were|are|was)\s+you)\s*"
    r"(told|instructed|programmed|trained|given))\b",
    r"\b(repeat|say|print|output)\b[\w\s,'\"-]{0,20}\b(everything|all|the\s+text|"
    r"the\s+words)\b[\w\s,'\"-]{0,20}\b(above|before|verbatim)\b",
    # 6. Role / persona hijack.
    r"\byou\s*('?re|\s+are)\s+now\b",
    r"\bfrom\s+now\s+on\b[\w\s,'\"-]{0,20}\b(you|act|pretend|behave|respond|answer|ignore)\b",
    r"\b(act|behave)\s+as\b[\w\s]{0,20}\b(an?\s+)?(unrestricted|unfiltered|"
    r"uncensored|jailbroken|evil|amoral|dan|ai|assistant|chatbot|language\s*model|"
    r"persona|character|hacker)\b",
    r"\bpretend\b[\w\s,'\"-]{0,20}\b(to\s+be|you'?re|you\s+are|that\s+you)\b",
    r"\b(roleplay|role-play|role\s+play)\b",
    r"\bimagine\s+(that\s+)?you\s*('?re|\s+are)\b",
    r"\bsimulate\s+(being|a\b)",
    r"\byou\s+will\s+(now\s+)?(act|behave|pretend|become|respond\s+as)\b",
    # 7. Named jailbreak modes / personas.
    r"\b(developer\s*mode|dev\s*mode|debug\s*mode|god\s*mode|sudo\s*mode|"
    r"root\s*mode|do\s*anything\s*now|jailbreak|jailbroken)\b",
    r"\b(DAN|AIM|STAN|DUDE)\b",
    r"\b(unfiltered|uncensored|no\s*filter|without\s+(any\s+)?(filter|filters|"
    r"restriction|restrictions|censorship|limitation|limitations|rules|"
    r"guardrails))\b",
    # 8. Planted "new/real/secret instructions".
    r"\b(new|updated|revised|real|true|actual|secret|hidden|override)\b"
    r"[\w\s,'\"-]{0,20}\b(instruction|instructions|system\s*prompt|directive|"
    r"directives|rule\s*set|ruleset)\b",
    r"\bhere\s+(are|is)\b[\w\s,'\"-]{0,20}\b(your\s+)?(new\s+|real\s+|actual\s+)?"
    r"(instruction|instructions|rules|directives)\b",
    # 9. Stop following the rules.
    r"\b(do\s*not|don'?t|stop|cease|never|no\s+longer)\b[\w\s,'\"-]{0,20}\b"
    r"(follow(ing)?|obey(ing)?|adhere|comply(ing)?|abide)\b[\w\s,'\"-]{0,20}\b"
    r"(instruction|instructions|rule|rules|guideline|guidelines|guidance|prompt|"
    r"policy|policies)\b",
    # 10. Encoding / smuggling tricks.
    r"\b(decode|decrypt|de-?obfuscate|un-?scramble)\b[\w\s,'\"-]{0,20}\b"
    r"(the\s+following|this|below|message|payload|instruction|instructions)\b",
    # 11. Injected chat/role markers & special tokens.
    r"<\|?\s*(im_start|im_end|system|user|assistant|endoftext)\s*\|?>",
    r"\[/?\s*(INST|SYS|SYSTEM)\s*\]",
    r"(^|\n)\s*#{2,}\s*(system|instruction|instructions|system\s*prompt)\b",
    r"(^|\n)\s*(system|assistant)\s*:",
]

# Zero-width / bidirectional control characters often used to smuggle payloads.
_HIDDEN_CHARS = re.compile(r"[\u200b\u200c\u200d\u202a-\u202e\u2066-\u2069]")

# Unambiguous jailbreak signatures checked against a punctuation/space-stripped
# copy of the text, to defeat obfuscation like "i g n o r e" or "ig.no.re".
# Kept deliberately specific so they never collide with benign guide questions.
_NORMALISED_SIGNATURES = (
    "ignorepreviousinstructions",
    "ignoreallinstructions",
    "ignoreallprevious",
    "ignoreabove",
    "disregardprevious",
    "disregardallinstructions",
    "forgetpreviousinstructions",
    "forgeteverythingabove",
    "doanythingnow",
    "developermode",
    "jailbreak",
    "pretendyouare",
    "actasdan",
    "revealyoursystemprompt",
    "printyoursystemprompt",
    "whatisyoursystemprompt",
    "showyourinstructions",
    "ignoreyourguidelines",
    "bypassyourrules",
    "disableyoursafety",
)

_COMPILED = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _INJECTION_PATTERNS]

# Strip everything that isn't a letter or digit, for the normalised pass.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass
class GuardResult:
    """Outcome of an input-guard assessment."""

    allowed: bool
    reason: str = ""


def _normalise(text: str) -> str:
    """Lower-case and strip all non-alphanumeric characters (anti-obfuscation)."""
    return _NON_ALNUM.sub("", text.lower())


def assess_input(text: str) -> GuardResult:
    """Assess a user message for prompt-injection / jailbreak attempts.

    Args:
        text: The raw user message.

    Returns:
        A :class:`GuardResult`. ``allowed=False`` means the message should be
        answered with a brief, in-scope refusal rather than retrieved against.
        ``reason`` is a short machine-readable category used in the trace/logs.
    """
    if not text or not text.strip():
        return GuardResult(allowed=False, reason="empty_message")

    if _HIDDEN_CHARS.search(text):
        return GuardResult(allowed=False, reason="hidden_control_characters")

    for pattern in _COMPILED:
        if pattern.search(text):
            return GuardResult(allowed=False, reason="prompt_injection_pattern")

    normalised = _normalise(text)
    for signature in _NORMALISED_SIGNATURES:
        if signature in normalised:
            return GuardResult(allowed=False, reason="prompt_injection_obfuscated")

    return GuardResult(allowed=True)
