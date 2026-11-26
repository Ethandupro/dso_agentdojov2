"""A compact, S0-redacted digest of the victim's run, shown to the evolving attacker (item 4, decided 2026-11-20).

The attacker already sees the victim's final answer (PR #38). The victim now reasons (Magistral), so its step-by-step
run -- its thinking, what it said, the tools it called and what they returned -- shows *why* an injection was ignored
or refused, which a flat "failed" never did. `victim_trace_digest` turns the run's messages into that digest.

It is a deliberate grey-box signal: the attacker sees the victim's reasoning, not only its output. To keep the
evolving defence from leaking into the attacker (above all in C4, where S0 changes during the run), any wording the
victim copied from the current S0 skill is redacted first: exact lines and any 6-word span shared with S0 become
`[S0 redacted]`. Identical in C2 and C4.

The reasoning (2026-11-25): vLLM's reasoning parser returns Magistral's `[THINK]` text in a separate
`reasoning_content` field, which LangChain's `ChatOpenAI` dropped, so traces before this date held no thinking at all.
The harness now keeps it (`ReasoningChatOpenAI`) and records it per message as `deepagents.reasoning`, outside the
message content so AgentDojo's scoring is unchanged; `victim_trace_digest` shows it as `thought:` lines, S0-redacted
like everything else.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

MASK = "[S0 redacted]"
_REDACT_NGRAM = 6
"""Shared word-span length that triggers redaction. Short enough to catch a paraphrased S0 rule, long enough that
ordinary shared phrases ("I will not") survive."""

_WORD = re.compile(r"\S+")


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)} if len(words) >= n else set()


def redact_s0(text: str, s0_markdown: str | None, n: int = _REDACT_NGRAM) -> str:
    """Replace any `n`-word span that `text` shares with the S0 skill body with `MASK`, merging adjacent masked
    spans. Case-insensitive on words; the rest of `text` is returned unchanged. No-op when there is no S0."""
    if not s0_markdown:
        return text
    banned = _ngrams(s0_markdown.lower().split(), n)
    if not banned:
        return text
    tokens = list(_WORD.finditer(text))
    lowered = [m.group().lower() for m in tokens]
    masked = [False] * len(tokens)
    for i in range(len(tokens) - n + 1):
        if tuple(lowered[i : i + n]) in banned:
            for j in range(i, i + n):
                masked[j] = True
    if not any(masked):
        return text
    out: list[str] = []
    pos = 0
    i = 0
    while i < len(tokens):
        if masked[i]:
            start = tokens[i].start()
            while i < len(tokens) and masked[i]:
                i += 1
            end = tokens[i - 1].end()
            out.append(text[pos:start])
            out.append(MASK)
            pos = end
        else:
            i += 1
    out.append(text[pos:])
    return "".join(out)


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def _digest_lines(
    messages: Sequence[Any],
    reasoning: Mapping[int, str],
    step_chars: int,
    thought_chars: int,
    hidden_tools: frozenset[str] = frozenset(),
) -> list[str]:
    lines: list[str] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            thought = (reasoning.get(index) or "").strip()
            if thought:
                lines.append(f"thought: {_clip(thought, thought_chars)}")
            for block in message.get("content") or []:
                kind = block.get("type")
                body = (block.get("content") or "").strip()
                if not body:
                    continue
                if kind in ("thinking", "redacted_thinking"):
                    lines.append(f"thought: {_clip(body, thought_chars)}")
                else:
                    lines.append(f"said: {_clip(body, step_chars)}")
            for call in message.get("tool_calls") or []:
                args = ", ".join(f"{k}={v!r}" for k, v in dict(call.args).items())
                lines.append(_clip(f"called: {call.function}({args})", step_chars))
        elif role == "tool":
            call = message.get("tool_call")
            if call is not None and call.function in hidden_tools:
                lines.append(f"tool result: (the {call.function} output, not shown)")
                continue
            body = " ".join(b.get("content") or "" for b in message.get("content") or []).strip()
            if body:
                lines.append(f"tool result: {_clip(body, step_chars)}")
    return lines


def victim_trace_digest(
    messages: Sequence[Any],
    s0_markdown: str | None,
    max_chars: int = 4000,
    step_chars: int = 400,
    reasoning: Mapping[int, str] | None = None,
    thought_chars: int = 800,
) -> str:
    """A short, ordered digest of the victim's run for the attacker's leaderboard: each assistant step's thinking and
    text, the tools it called, and each tool result, S0-redacted and truncated. Capped at `max_chars`.

    `reasoning` maps a message's index in `messages` to the reasoning the model produced for it, when the server
    returned it outside the message content (the DeepAgents harness records it as `deepagents.reasoning`; see
    `ReasoningChatOpenAI`). Each thought is cut to `thought_chars`, other lines to `step_chars`. Thinking blocks
    inside a message's content are shown too."""
    lines = _digest_lines(messages, reasoning or {}, step_chars, thought_chars)
    digest = redact_s0("\n".join(lines), s0_markdown)
    return digest if len(digest) <= max_chars else digest[:max_chars] + " ...[truncated]"


def judge_trace_digest(
    messages: Sequence[Any],
    reasoning: Mapping[int, str] | None = None,
    hidden_tools: frozenset[str] = frozenset(),
    max_chars: int = 4000,
    step_chars: int = 400,
    thought_chars: int = 1200,
) -> str:
    """The same digest for the C3/C4 judge (PR 2 of the 2026-11-26 design note): the judge owns S0, so nothing is
    redacted, which lets it see which rules the victim quoted. The outputs of `hidden_tools` (`load_skill`: the
    skill bodies, which the judge already has) are left out to save room."""
    lines = _digest_lines(messages, reasoning or {}, step_chars, thought_chars, hidden_tools)
    digest = "\n".join(lines)
    return digest if len(digest) <= max_chars else digest[:max_chars] + " ...[truncated]"


def victim_own_text(messages: Sequence[Any], reasoning: Mapping[int, str] | None = None, max_chars: int = 30000) -> str:
    """Everything the victim itself wrote (thinking and replies, not tool results), uncut up to `max_chars`: what the
    judge's rule-quote counts are computed from."""
    reasoning = reasoning or {}
    parts: list[str] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if reasoning.get(index):
            parts.append(reasoning[index])
        for block in message.get("content") or []:
            if block.get("content"):
                parts.append(block["content"])
    return "\n".join(parts)[:max_chars]
