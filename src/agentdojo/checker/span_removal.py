"""Applies the checker's removals to a tool output (sanitize mode, `skillfile_evolution/DESIGN.md`, "Checker").

The checker returns `{"remove": [{"span": "...", "reason": "..."}]}`, where each span should be a verbatim substring of
the tool output. In practice a verbatim copy is rare: AgentDojo formats tool outputs as YAML, which wraps long strings
at 80 columns, indents continuation lines, writes newlines as `\\n` escapes or blank lines, and doubles single quotes.
An injected paragraph therefore never appears in the output the way it was written, and a model quoting it tends to
undo the wrapping and escapes. Exact matching would leave most such spans unmatched (and so ignored), and the injection
would survive for a formatting reason rather than a security one.

So spans are matched on a *canonical* form of both texts, in which whitespace runs, YAML/JSON escapes and line
continuations, and doubled quotes are normalised, and each match is mapped back to its exact character range in the
raw output. Only that range is deleted; everything else in the output is kept byte for byte.

The same canonical form is used by `injection_survival` to measure whether injected text reached the agent.
"""

import difflib
import json
import re
from dataclasses import dataclass, field

SPAN_REMOVED_MARKER = "[removed by security check]"
"""Put in place of each (merged) deletion."""

WHOLE_OUTPUT_REMOVED_MARKER = "[entire output removed by security check]"
"""Replaces the whole output when the checker's reply doesn't parse (fail-closed). Differs from the per-span
marker, so logs and the agent can tell the two apart."""

_SIMPLE_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "'": "'"}
_WHITESPACE_ESCAPES = frozenset("ntr")
_HEX = re.compile(r"[0-9a-fA-F]{4}")
_FENCED_JSON = re.compile(r"^\s*```(?:json)?\s*\n(.*)\n\s*```\s*$", re.DOTALL)


@dataclass
class CanonicalText:
    """A normalised copy of a text, with the raw character range each canonical character came from.

    `text[k]` comes from `raw[starts[k]:ends[k]]`. Any run of whitespace (including escaped newlines/tabs and YAML
    line continuations) becomes one space.
    """

    raw: str
    text: str
    starts: list[int]
    ends: list[int]


def canonicalize(raw: str) -> CanonicalText:
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []

    def emit(char: str, start: int, end: int) -> None:
        if char == " " and chars and chars[-1] == " ":
            ends[-1] = end  # extend the current whitespace run
            return
        chars.append(char)
        starts.append(start)
        ends.append(end)

    i, n = 0, len(raw)
    while i < n:
        c = raw[i]
        if c == "\\" and i + 1 < n:
            nxt = raw[i + 1]
            if nxt in _WHITESPACE_ESCAPES or nxt.isspace():
                # `\n`, `\t`, `\r` escapes; YAML's `\` + newline (line continuation) and `\ ` (escaped space)
                emit(" ", i, i + 2)
                i += 2
                continue
            if nxt in _SIMPLE_ESCAPES:
                emit(_SIMPLE_ESCAPES[nxt], i, i + 2)
                i += 2
                continue
            if nxt == "u" and _HEX.fullmatch(raw, i + 2, i + 6):
                decoded = chr(int(raw[i + 2 : i + 6], 16))
                emit(" " if decoded.isspace() else decoded, i, i + 6)
                i += 6
                continue
            emit(c, i, i + 1)
            i += 1
        elif c == "'" and raw.startswith("''", i):
            emit("'", i, i + 2)  # YAML single-quoted strings double their quotes
            i += 2
        elif c.isspace():
            emit(" ", i, i + 1)
            i += 1
        else:
            emit(c, i, i + 1)
            i += 1
    return CanonicalText(raw=raw, text="".join(chars), starts=starts, ends=ends)


def canonical_form(text: str) -> str:
    """The canonical form of `text`, without leading or trailing whitespace."""
    return canonicalize(text).text.strip()


@dataclass
class SpanResult:
    span: str
    reason: str | None
    occurrences: int
    """How many times the span was found (and removed). 0 means unmatched."""
    too_short: bool = False
    """The span was shorter than the minimum length and was not applied."""


@dataclass
class SanitizeResult:
    text: str
    """The output the agent is shown."""
    spans: list[SpanResult] = field(default_factory=list)
    parse_failure: bool = False
    n_markers: int = 0
    """Number of `SPAN_REMOVED_MARKER`s inserted (after merging overlapping/adjacent deletions)."""
    chars_removed: int = 0
    """Characters of the raw output that were deleted."""

    @property
    def matched(self) -> list[SpanResult]:
        return [s for s in self.spans if s.occurrences > 0]

    @property
    def unmatched(self) -> list[SpanResult]:
        return [s for s in self.spans if s.occurrences == 0 and not s.too_short]

    @property
    def too_short(self) -> list[SpanResult]:
        return [s for s in self.spans if s.too_short]


def parse_checker_reply(reply: str) -> list[tuple[str, str | None]] | None:
    """Parses `{"remove": [{"span": "...", "reason": "..."}]}` into `(span, reason)` pairs.

    A single Markdown code fence around the JSON is tolerated. Returns `None` if the reply doesn't parse or doesn't
    have that shape (which removes the whole output).
    """
    fenced = _FENCED_JSON.match(reply)
    if fenced is not None:
        reply = fenced.group(1)
    try:
        data = json.loads(reply)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("remove"), list):
        return None
    removals: list[tuple[str, str | None]] = []
    for item in data["remove"]:
        if not isinstance(item, dict) or not isinstance(item.get("span"), str):
            return None
        reason = item.get("reason")
        removals.append((item["span"], reason if isinstance(reason, str) else None))
    return removals


def _find_all(haystack: str, needle: str) -> list[int]:
    found = []
    start = haystack.find(needle)
    while start != -1:
        found.append(start)
        start = haystack.find(needle, start + 1)
    return found


def remove_spans(output: str, removals: list[tuple[str, str | None]], min_span_chars: int = 0) -> SanitizeResult:
    """Deletes every occurrence of each span from `output`, matching on the canonical form.

    Overlapping or adjacent deletions (including ones separated only by whitespace) are merged and replaced by one
    `SPAN_REMOVED_MARKER`. A span not found in the output is ignored and reported as unmatched. A span whose canonical
    form is shorter than `min_span_chars` is not applied and is reported as too short.
    """
    canonical = canonicalize(output)
    results: list[SpanResult] = []
    ranges: list[tuple[int, int]] = []
    for span, reason in removals:
        needle = canonical_form(span)
        if needle and len(needle) < min_span_chars:
            results.append(SpanResult(span=span, reason=reason, occurrences=0, too_short=True))
            continue
        positions = _find_all(canonical.text, needle) if needle else []
        for pos in positions:
            ranges.append((canonical.starts[pos], canonical.ends[pos + len(needle) - 1]))
        results.append(SpanResult(span=span, reason=reason, occurrences=len(positions)))

    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and output[merged[-1][1] : start].strip() == "":
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    pieces: list[str] = []
    last = 0
    for start, end in merged:
        pieces.append(output[last:start])
        pieces.append(SPAN_REMOVED_MARKER)
        last = end
    pieces.append(output[last:])
    return SanitizeResult(
        text="".join(pieces),
        spans=results,
        n_markers=len(merged),
        chars_removed=sum(end - start for start, end in merged),
    )


def sanitize_output(output: str, checker_reply: str, min_span_chars: int = 0) -> SanitizeResult:
    """Parses the checker's reply and applies it to `output`. An unparseable reply removes the whole output."""
    removals = parse_checker_reply(checker_reply)
    if removals is None:
        return SanitizeResult(
            text=WHOLE_OUTPUT_REMOVED_MARKER, parse_failure=True, n_markers=0, chars_removed=len(output)
        )
    return remove_spans(output, removals, min_span_chars=min_span_chars)


@dataclass
class InjectionSurvival:
    survived_whole: bool
    """The whole injection (canonical form) is still in the output."""
    surviving_fraction: float
    """Share of the injection's words that are still in the output, counting only runs of at least `min_run` words in
    the injection's order. 0.0 when nothing of it is left; 1.0 when all of it is."""


def injection_survival(output: str, injection: str, min_run: int = 4) -> InjectionSurvival:
    """How much of an injected text reached the agent, comparing canonical forms (see module docstring)."""
    needle = canonical_form(injection)
    haystack = canonical_form(output)
    injection_words = needle.split(" ") if needle else []
    if not injection_words:
        return InjectionSurvival(survived_whole=False, surviving_fraction=0.0)
    if needle in haystack:
        return InjectionSurvival(survived_whole=True, surviving_fraction=1.0)
    matcher = difflib.SequenceMatcher(None, injection_words, haystack.split(" "), autojunk=False)
    surviving = sum(block.size for block in matcher.get_matching_blocks() if block.size >= min_run)
    return InjectionSurvival(survived_whole=False, surviving_fraction=surviving / len(injection_words))
