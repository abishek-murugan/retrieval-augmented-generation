from __future__ import annotations

import logging
import re

logging.getLogger("opentelemetry.exporter.otlp.proto.http.trace_exporter").disabled = True
logging.getLogger("opentelemetry.exporter.otlp.proto.grpc.trace_exporter").disabled = True

from guardrails.settings import settings as _gr_settings

_gr_settings.disable_tracing = True

from guardrails import Guard
from guardrails.classes import ValidationOutcome
from guardrails.validator_base import FailResult, PassResult, ValidationResult, Validator
from guardrails.validator_base import register_validator

_EMAIL = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
_PHONE = r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{4})"
_SSN = r"\b\d{3}-\d{2}-\d{4}\b"
_IP = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
_CARD = r"\b(?:\d[ -]*?){13,16}\b"

_INJECTION_PATTERNS = [
    r"ignore (all |any |previous )?(instructions|prompts|rules|directives)",
    r"disregard (all |the )?(previous |above )?(instructions|rules)",
    r"system prompt",
    r"reveal your (system )?(prompt|instructions)",
    r"jailbreak",
    r"developer mode",
    r"act as (dan|sudo|root)",
    r"pretend you('re| are) (dan|unfiltered|uncensored)",
    r"forget (all |the )?rules",
    r"hidden instructions",
    r"base64",
    r"dall[e]?knife",
]
_TOXIC_WORDS = {
    "fuck", "shit", "bitch", "asshole", "bastard", "cunt", "nigga", "nigger",
    "kys", "kill yourself", "retard", "rape", "pedophile",
}
_OFFTOPIC_HINTS = {
    "cricket", "football", "bollywood", "recipe", "cookbook", "movie", "celebrity",
    "concert", "trading", "cryptocurrency", "forex", "lottery", "pregnancy",
}

_GR_PII: list[tuple[str, str]] = [
    ("email", _EMAIL),
    ("ssn", _SSN),
    ("ip", _IP),
    ("card", _CARD),
]


@register_validator("prohibited-patterns", data_type="string")
class ProhibitedPatterns(Validator):
    def _validate(self, value, metadata) -> ValidationResult:
        found = []
        for name, pattern in _GR_PII:
            if re.search(pattern, value):
                found.append(name)
        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                found.append("prompt-injection")
        lower = value.lower()
        for w in _TOXIC_WORDS:
            if w in lower:
                found.append("toxic")
        if found:
            return FailResult(error_message=f"Blocked: detected {', '.join(dict.fromkeys(found))}")
        return PassResult()


@register_validator("off-topic", data_type="string")
class OffTopic(Validator):
    MIN_LEN = 6

    def _validate(self, value, metadata) -> ValidationResult:
        lower = value.lower().strip()
        greet = re.fullmatch(r"(hi|hello|hey|yo|thanks|thank you|ok|okay)[. ,!?]*", lower)
        hits = [k for k in _OFFTOPIC_HINTS if k in lower]
        if hits or (greet is None and len(lower) > self.MIN_LEN and not self._defenceish(lower)):
            return FailResult(error_message="Off-topic: question is outside defence scope")
        return PassResult()

    @staticmethod
    def _defenceish(text: str) -> bool:
        tokens = {
            "nato", "defence", "defense", "military", "army", "air force", "navy",
            "doctrine", "weapon", "war", "conflict", "security", "missile", "drone",
            "autonomy", "deterrence", "strategy", "rate treaty", "cyber", "intelligence",
            "tank", "fighter", "combat", "nuclear", "treaty", "alliance", "geopolitic",
            "logistics", "procurement", "interoperability", "command", "doctrine",
            "doctrinal", "forces", "command control", "munitions", "raid", "feasibility",
        }
        return any(tok in text for tok in tokens)


@register_validator("citations-present", data_type="string")
class CitationsPresent(Validator):
    def _validate(self, value, metadata) -> ValidationResult:
        if not value.strip():
            return FailResult(error_message="Empty answer")
        if "Source" in value or "source" in value.lower() and "p." in value:
            return PassResult()
        return FailResult(error_message="Answer lacks source citations")


def _summarize(outcome: ValidationOutcome) -> list[str]:
    reasons = []
    for s in outcome.validation_summaries:
        if getattr(s, "validator_status", None) == "fail":
            reasons.append(getattr(s, "failure_reason", "") or "guard failed")
    return reasons


class DefenceGuards:
    def __init__(self):
        self.input_guard: Guard = Guard().use(
            ProhibitedPatterns(on_fail="noop"),
            OffTopic(on_fail="noop"),
        )
        self.output_guard: Guard = Guard().use(
            ProhibitedPatterns(on_fail="noop"),
            CitationsPresent(on_fail="noop"),
        )

    def check_input(self, query: str) -> tuple[bool, list[str]]:
        outcome = self.input_guard.validate(query)
        return outcome.validation_passed, _summarize(outcome)

    def check_output(self, answer: str) -> tuple[bool, list[str]]:
        outcome = self.output_guard.validate(answer)
        return outcome.validation_passed, _summarize(outcome)


guards = DefenceGuards()