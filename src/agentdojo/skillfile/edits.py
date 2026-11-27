"""The judge's reply format and how code applies its delta edits (DESIGN.md, "Judge output").

The judge never writes the file. It returns a diagnosis and a list of edits:

- `ADD` a rule to a section,
- `MODIFY` a rule by ID,
- `DELETE` a rule by ID,
- `MERGE` two or more rules of one section into one new rule,
- `DESCRIBE`: replace the skill's one-line description, which the victim sees in its skill list before deciding
  whether to load the skill (only when `EditPolicy.description_editable`; decided 2026-11-19 by user).

With the security-skill library (`library`, design note 2026-11-26 "Specialised defender skills"), `ADD` and
`DESCRIBE` may name a `skill` (empty: S0), and two more operations exist:

- `CREATE` a specialised security skill: a name (`defend-<kind>`), a one-line "covers" description, and its rules;
- `RETIRE` a specialised skill.

`MODIFY`, `DELETE` and `MERGE` need no skill: rule IDs are unique across the library.

`apply_edits` applies them to a copy of the file. Everything the edits don't name stays byte-identical. A new or
merged rule gets the next unused ID of its section; deleted and merged IDs are retired and never reused.
Validity (IDs exist, frozen parts untouched, budget, identifier ban) is checked separately, in `checks`.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from agentdojo.skillfile.structure import Rule, SkillFile

DiagnosisType = Literal[
    "missing",
    "vague",
    "conflict",
    "not_applied",
    "over_removed",
    "not_loaded",
    "over_restrictive",
    "no_skill",
    "wrong_skill",
]
DIAGNOSIS_TYPES: tuple[str, ...] = (
    "missing",
    "vague",
    "conflict",
    "not_applied",
    "over_removed",
    "not_loaded",
    "over_restrictive",
    "no_skill",
    "wrong_skill",
)
"""`over_removed` belongs to the external-checker prompt; `not_loaded` and `over_restrictive` to the skills prompt;
`no_skill` and `wrong_skill` to the security-skill library (2026-11-26)."""


@dataclass
class Diagnosis:
    type: DiagnosisType
    rules: list[str]
    """IDs of the rules involved; empty for `missing`."""
    explanation: str


@dataclass
class AddRule:
    section: str
    text: str
    keeps: str = ""
    """A benign text the rule would keep. Asked for in the prompt only; no code checks it."""
    skill: str = ""
    """Library only: the skill the rule goes to; empty means S0."""
    op: Literal["ADD"] = "ADD"


@dataclass
class ModifyRule:
    id: str
    text: str
    keeps: str = ""
    op: Literal["MODIFY"] = "MODIFY"


@dataclass
class DeleteRule:
    id: str
    op: Literal["DELETE"] = "DELETE"


@dataclass
class MergeRules:
    ids: list[str]
    text: str
    keeps: str = ""
    op: Literal["MERGE"] = "MERGE"


@dataclass
class DescribeSkill:
    text: str
    """The new one-line description of the skill."""
    skill: str = ""
    """Library only: the skill whose description changes; empty means S0."""
    op: Literal["DESCRIBE"] = "DESCRIBE"


@dataclass
class CreateSkill:
    """Library only: a new specialised security skill."""

    name: str
    """`defend-<kind>`, lowercase words joined by hyphens."""
    covers: str
    """One line: the kind of injected text the skill covers, as the victim can see it. It is the skill's description
    in the victim's skill list and its line in S0's index."""
    rules: list[str]
    op: Literal["CREATE"] = "CREATE"


@dataclass
class RetireSkill:
    """Library only: remove a specialised skill. Its name is never reused."""

    skill: str
    op: Literal["RETIRE"] = "RETIRE"


Edit = AddRule | ModifyRule | DeleteRule | MergeRules | DescribeSkill | CreateSkill | RetireSkill
LIBRARY_OPS = frozenset({"CREATE", "RETIRE"})


@dataclass
class JudgeReply:
    diagnosis: list[Diagnosis]
    edits: list[Edit] = field(default_factory=list)
    """May be empty: the judge sees no problem with the file."""


@dataclass
class EditOrigin:
    pair: str
    trial: int
    step: int

    @property
    def label(self) -> str:
        return f"{self.pair} trial {self.trial} step {self.step}"


class JudgeReplyError(ValueError):
    pass


_FENCE = re.compile(r"^```(?:json)?\s*\n(.*)\n```\s*$", re.DOTALL)


def _str(item: dict[str, Any], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise JudgeReplyError(f"{item.get('op', 'diagnosis')}: '{key}' must be a non-empty string")
    return value.strip()


def _skill(item: dict[str, Any]) -> str:
    value = item.get("skill", "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise JudgeReplyError(f"{item.get('op')}: 'skill' must be a skill name")
    return value.strip()


def parse_judge_reply(reply: str) -> JudgeReply:
    """Parse the judge's JSON reply. Raises `JudgeReplyError` with a message the judge can act on in its retry."""
    text = reply.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise JudgeReplyError(f"the reply is not valid JSON: {e}") from e
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("diagnosis"), list)
        or not isinstance(data.get("edits"), list)
    ):
        raise JudgeReplyError("the reply must be an object with a 'diagnosis' list and an 'edits' list")

    diagnosis: list[Diagnosis] = []
    for item in data["diagnosis"]:
        if not isinstance(item, dict):
            raise JudgeReplyError("each diagnosis entry must be an object")
        kind = item.get("type")
        if kind not in DIAGNOSIS_TYPES:
            raise JudgeReplyError(f"diagnosis type must be one of {', '.join(DIAGNOSIS_TYPES)}; got {kind!r}")
        rules = item.get("rules", [])
        if not isinstance(rules, list) or not all(isinstance(r, str) for r in rules):
            raise JudgeReplyError("diagnosis 'rules' must be a list of rule IDs")
        diagnosis.append(Diagnosis(type=kind, rules=rules, explanation=_str(item, "explanation")))

    edits: list[Edit] = []
    for item in data["edits"]:
        if not isinstance(item, dict):
            raise JudgeReplyError("each edit must be an object")
        op = item.get("op")
        keeps = item.get("keeps", "") if isinstance(item.get("keeps", ""), str) else ""
        if op == "ADD":
            edits.append(
                AddRule(section=_str(item, "section"), text=_str(item, "text"), keeps=keeps, skill=_skill(item))
            )
        elif op == "MODIFY":
            edits.append(ModifyRule(id=_str(item, "id"), text=_str(item, "text"), keeps=keeps))
        elif op == "DELETE":
            edits.append(DeleteRule(id=_str(item, "id")))
        elif op == "MERGE":
            ids = item.get("ids")
            if not isinstance(ids, list) or len(ids) < 2 or not all(isinstance(i, str) for i in ids):
                raise JudgeReplyError("MERGE: 'ids' must list at least two rule IDs")
            edits.append(MergeRules(ids=ids, text=_str(item, "text"), keeps=keeps))
        elif op == "DESCRIBE":
            edits.append(DescribeSkill(text=_str(item, "text"), skill=_skill(item)))
        elif op == "CREATE":
            rules = item.get("rules")
            if not isinstance(rules, list) or not rules or not all(isinstance(r, str) and r.strip() for r in rules):
                raise JudgeReplyError("CREATE: 'rules' must be a non-empty list of rule texts")
            edits.append(
                CreateSkill(name=_str(item, "name"), covers=_str(item, "covers"), rules=[r.strip() for r in rules])
            )
        elif op == "RETIRE":
            edits.append(RetireSkill(skill=_str(item, "skill")))
        else:
            raise JudgeReplyError(
                f"edit 'op' must be ADD, MODIFY, DELETE, MERGE, DESCRIBE, CREATE or RETIRE; got {op!r}"
            )
    return JudgeReply(diagnosis=diagnosis, edits=edits)


def edit_to_dict(edit: Edit) -> dict[str, Any]:
    """`skill` appears only when set, so step files of single-skill runs are unchanged."""
    if isinstance(edit, AddRule):
        add = {"op": "ADD", "section": edit.section, "text": edit.text, "keeps": edit.keeps}
        return {**add, "skill": edit.skill} if edit.skill else add
    if isinstance(edit, ModifyRule):
        return {"op": "MODIFY", "id": edit.id, "text": edit.text, "keeps": edit.keeps}
    if isinstance(edit, DeleteRule):
        return {"op": "DELETE", "id": edit.id}
    if isinstance(edit, DescribeSkill):
        describe = {"op": "DESCRIBE", "text": edit.text}
        return {**describe, "skill": edit.skill} if edit.skill else describe
    if isinstance(edit, CreateSkill):
        return {"op": "CREATE", "name": edit.name, "covers": edit.covers, "rules": list(edit.rules)}
    if isinstance(edit, RetireSkill):
        return {"op": "RETIRE", "skill": edit.skill}
    return {"op": "MERGE", "ids": list(edit.ids), "text": edit.text, "keeps": edit.keeps}


def reply_to_dict(reply: JudgeReply) -> dict[str, Any]:
    return {
        "diagnosis": [{"type": d.type, "rules": d.rules, "explanation": d.explanation} for d in reply.diagnosis],
        "edits": [edit_to_dict(e) for e in reply.edits],
    }


class EditApplyError(ValueError):
    pass


def _locate(skill_file: SkillFile, rule_id: str):
    found = skill_file.find_rule(rule_id)
    if found is None:
        raise EditApplyError(f"no rule with ID {rule_id}")
    return found


def apply_edits(skill_file: SkillFile, edits: list[Edit], origin: EditOrigin) -> SkillFile:
    """Apply the edits in order to a copy of `skill_file` and return the copy. The input is not changed. `skill` fields
    are ignored here (the library routes each edit to its file); CREATE and RETIRE need the library."""
    result = skill_file.copy()
    for edit in edits:
        if isinstance(edit, CreateSkill | RetireSkill):
            raise EditApplyError(f"{edit.op} needs the security-skill library")
        if isinstance(edit, AddRule):
            section = result.section(edit.section)
            if section is None or section.kind != "rules":
                raise EditApplyError(f"no rule section named {edit.section!r}")
            section.rules.append(Rule(id=result.new_id(section), text=edit.text, origin=origin.label))
        elif isinstance(edit, ModifyRule):
            section, index = _locate(result, edit.id)
            section.rules[index] = Rule(id=edit.id, text=edit.text, origin=origin.label)
        elif isinstance(edit, DeleteRule):
            section, index = _locate(result, edit.id)
            del section.rules[index]
            result.retired_ids.append(edit.id)
        elif isinstance(edit, DescribeSkill):
            result.description = edit.text
        else:
            located = [_locate(result, rule_id) for rule_id in edit.ids]
            section = located[0][0]
            if any(s is not section for s, _ in located):
                raise EditApplyError("MERGE: all merged rules must be in the same section")
            position = min(index for _, index in located)
            merged = Rule(id=result.new_id(section), text=edit.text, origin=origin.label)
            section.rules = [rule for rule in section.rules if rule.id not in edit.ids]
            section.rules.insert(position, merged)
            result.retired_ids.extend(edit.ids)
    return result
