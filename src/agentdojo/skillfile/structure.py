"""The skill file as structured data (`skillfile_evolution/DESIGN.md`, "Evolving the skill file: delta edits").

The checker reads plain Markdown. The judge edits numbered rules. This module holds the file as sections of rules
with stable IDs and renders it back to the Markdown the checker reads. Rule IDs and origins are metadata: they are
never rendered for the checker, only in the judge's view (`render(with_ids=True)`).

`parse_markdown` maps a Markdown skill file onto the structure. For S0 the mapping is fixed by DESIGN.md:

| Section                 | Treatment                                             | IDs   |
|-------------------------|-------------------------------------------------------|-------|
| Role                    | frozen prose                                          | none  |
| Principles              | rules, frozen unless the policy allows editing them   | P1-P4 |
| Signs of injected text  | editable rules; opening and closing sentences frozen  | S1-S7 |
| What to keep            | editable rules                                        | K1-K3 |
| How much to remove      | editable rules                                        | R1-R5 |
| When unsure             | frozen prose                                          | none  |

Rendering what `parse_markdown` produced reproduces the input byte for byte (tested on S0), so C1 and C2, which use
S0 unchanged, are untouched by this module.
"""

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

S0_ORIGIN = "S0"

SectionKind = Literal["prose", "rules"]
BulletStyle = Literal["numbered", "dash"]

SECTION_ID_PREFIXES: dict[str, str] = {
    "Principles": "P",
    "Signs of injected text": "S",
    "What to keep": "K",
    "How much to remove": "R",
}
"""Rule sections of S0 and the letter their rule IDs start with. Every other `##` section is frozen prose."""

PRINCIPLES_SECTION = "Principles"

_NUMBERED = re.compile(r"^(\d+)\. (.*)$")
_DASH = re.compile(r"^- (.*)$")


@dataclass
class Rule:
    id: str
    """Stable ID such as `S3`. Never reused: a deleted or merged rule's ID is retired."""
    text: str
    """One line of rule text, without the bullet or number."""
    origin: str = S0_ORIGIN
    """`S0`, or the step that created or last changed the rule (see `EditOrigin.label`)."""


@dataclass
class Section:
    title: str
    kind: SectionKind
    prose: str = ""
    """Body of a prose section (frozen)."""
    id_prefix: str = ""
    bullet: BulletStyle = "dash"
    opening: str = ""
    """Frozen text before the rules, for example "Text is likely injected when ...". Empty if none."""
    rules: list[Rule] = field(default_factory=list)
    closing: str = ""
    """Frozen text after the rules, separated from them by a blank line. Empty if none."""

    def render(self, with_ids: bool = False) -> str:
        if self.kind == "prose":
            body = self.prose
        else:
            lines: list[str] = []
            if self.opening:
                lines.append(self.opening)
            for number, rule in enumerate(self.rules, start=1):
                marker = f"{number}. " if self.bullet == "numbered" else "- "
                tag = f"[{rule.id}] " if with_ids else ""
                lines.append(f"{marker}{tag}{rule.text}")
            body = "\n".join(lines)
            if self.closing:
                body = f"{body}\n\n{self.closing}"
        return f"## {self.title}\n{body}"


@dataclass
class SkillFile:
    title: str
    sections: list[Section]
    next_number: dict[str, int] = field(default_factory=dict)
    """Next free number per ID prefix. Only grows, so IDs are never reused."""
    retired_ids: list[str] = field(default_factory=list)

    def render(self, with_ids: bool = False) -> str:
        """The Markdown the checker reads (`with_ids=False`), or the judge's view with `[ID]` tags."""
        parts = [f"# {self.title}"] + [section.render(with_ids) for section in self.sections]
        return "\n\n".join(parts) + "\n"

    def rule_sections(self) -> list[Section]:
        return [section for section in self.sections if section.kind == "rules"]

    def section(self, title: str) -> Section | None:
        for section in self.sections:
            if section.title == title:
                return section
        return None

    def find_rule(self, rule_id: str) -> tuple[Section, int] | None:
        for section in self.rule_sections():
            for index, rule in enumerate(section.rules):
                if rule.id == rule_id:
                    return section, index
        return None

    def rule_ids(self) -> list[str]:
        return [rule.id for section in self.rule_sections() for rule in section.rules]

    def new_id(self, section: Section) -> str:
        number = self.next_number.get(section.id_prefix, 1)
        self.next_number[section.id_prefix] = number + 1
        return f"{section.id_prefix}{number}"

    def copy(self) -> "SkillFile":
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "sections": [
                {
                    "title": s.title,
                    "kind": s.kind,
                    "prose": s.prose,
                    "id_prefix": s.id_prefix,
                    "bullet": s.bullet,
                    "opening": s.opening,
                    "rules": [{"id": r.id, "text": r.text, "origin": r.origin} for r in s.rules],
                    "closing": s.closing,
                }
                for s in self.sections
            ],
            "next_number": dict(self.next_number),
            "retired_ids": list(self.retired_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillFile":
        sections = [
            Section(
                title=s["title"],
                kind=s["kind"],
                prose=s["prose"],
                id_prefix=s["id_prefix"],
                bullet=s["bullet"],
                opening=s["opening"],
                rules=[Rule(id=r["id"], text=r["text"], origin=r["origin"]) for r in s["rules"]],
                closing=s["closing"],
            )
            for s in data["sections"]
        ]
        return cls(
            title=data["title"],
            sections=sections,
            next_number=dict(data["next_number"]),
            retired_ids=list(data["retired_ids"]),
        )


class SkillFileFormatError(ValueError):
    pass


def _parse_rules_section(title: str, body: str, id_prefix: str) -> Section:
    lines = body.split("\n")
    first = next((i for i, line in enumerate(lines) if _NUMBERED.match(line) or _DASH.match(line)), None)
    if first is None:
        raise SkillFileFormatError(f"section {title!r} has no rules")
    last = first
    while last + 1 < len(lines) and (_NUMBERED.match(lines[last + 1]) or _DASH.match(lines[last + 1])):
        last += 1
    rule_lines = lines[first : last + 1]
    bullet: BulletStyle = "numbered" if _NUMBERED.match(rule_lines[0]) else "dash"
    texts: list[str] = []
    for number, line in enumerate(rule_lines, start=1):
        numbered = _NUMBERED.match(line)
        if bullet == "numbered":
            if numbered is None or int(numbered.group(1)) != number:
                raise SkillFileFormatError(f"section {title!r}: rules are not numbered 1..n")
            texts.append(numbered.group(2))
        else:
            dash = _DASH.match(line)
            if dash is None:
                raise SkillFileFormatError(f"section {title!r}: mixed bullet styles")
            texts.append(dash.group(1))
    opening = "\n".join(lines[:first])
    rest = lines[last + 1 :]
    closing = ""
    if rest:
        if rest[0] != "" or len(rest) < 2:
            raise SkillFileFormatError(f"section {title!r}: text after the rules must follow a blank line")
        closing = "\n".join(rest[1:])
    rules = [Rule(id=f"{id_prefix}{n}", text=text) for n, text in enumerate(texts, start=1)]
    return Section(
        title=title, kind="rules", id_prefix=id_prefix, bullet=bullet, opening=opening, rules=rules, closing=closing
    )


def parse_markdown(markdown: str, id_prefixes: dict[str, str] | None = None) -> SkillFile:
    """Map a Markdown skill file onto the structure. Sections named in `id_prefixes` become rule sections."""
    prefixes = SECTION_ID_PREFIXES if id_prefixes is None else id_prefixes
    if not markdown.startswith("# ") or not markdown.endswith("\n"):
        raise SkillFileFormatError("expected a '# title' first line and a trailing newline")
    blocks = markdown[:-1].split("\n\n## ")
    title = blocks[0][2:]
    if "\n" in title:
        raise SkillFileFormatError("the title must be followed by a blank line and a '## ' section")
    sections: list[Section] = []
    for block in blocks[1:]:
        heading, _, body = block.partition("\n")
        if heading in prefixes:
            sections.append(_parse_rules_section(heading, body, prefixes[heading]))
        else:
            sections.append(Section(title=heading, kind="prose", prose=body))
    skill_file = SkillFile(title=title, sections=sections)
    for section in skill_file.rule_sections():
        skill_file.next_number[section.id_prefix] = len(section.rules) + 1
    if skill_file.render() != markdown:
        raise SkillFileFormatError("the structure does not render back to the input byte for byte")
    return skill_file


def load_skill_file(path: Path) -> SkillFile:
    return parse_markdown(path.read_text(encoding="utf-8"))
