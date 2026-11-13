"""The one place that calls Claude (optional: only used when ANTHROPIC_API_KEY is set).

Both jobs are small - rating headlines, writing a short "why it moved" - so thinking
runs at low effort. Otherwise it can use up max_tokens before the answer is written.
A reply that was cut off or declined raises ClaudeIncomplete, and the caller falls back
to its keyword / rule-based version instead of using half an answer.
"""
from __future__ import annotations

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

_client = None


class ClaudeIncomplete(RuntimeError):
    """Claude stopped before finishing the answer (hit max_tokens, or declined)."""


def ask(prompt: str, max_tokens: int) -> str:
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=60.0)
    r = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
    )
    if r.stop_reason != "end_turn":
        raise ClaudeIncomplete(f"reply incomplete (stop_reason={r.stop_reason})")
    return "".join(b.text for b in r.content if b.type == "text").strip()
