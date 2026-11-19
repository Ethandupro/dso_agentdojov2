"""The victim's skill library: SKILL.md parsing, the prompt listing, and `load_skill` through a real DeepAgents run."""

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("deepagents")

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from agentdojo import experiment_models
from agentdojo.agent_pipeline.agent_skills import (
    LOAD_SKILL_TOOL_NAME,
    AgentSkill,
    AgentSkillsMiddleware,
    SkillFormatError,
    SkillLibrary,
    load_skills_dir,
    parse_skill_md,
    skills_prompt,
)
from agentdojo.agent_pipeline.deepagents_pipeline import DeepAgentsPipeline
from agentdojo.task_suite.load_suites import get_suite

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_project_skills_load_and_s0_is_not_among_them():
    skills = load_skills_dir(REPO_ROOT / experiment_models.AGENT_SKILLS_DIR)
    assert [s.name for s in skills] == [
        "brainstorming",
        "budgeting",
        "calendar-planning",
        "document-summarizing",
        "writing-messages",
    ]
    assert experiment_models.S0_SKILL_NAME not in [s.name for s in skills]
    assert all(s.body.startswith("# ") for s in skills)


def test_parse_skill_md_and_errors():
    skill = parse_skill_md("---\nname: my-skill\ndescription: Use when testing.\n---\n\n# Body\n")
    assert (skill.name, skill.description, skill.body) == ("my-skill", "Use when testing.", "# Body\n")
    with pytest.raises(SkillFormatError):
        parse_skill_md("# no front matter\n")
    with pytest.raises(SkillFormatError):
        parse_skill_md("---\nname: Bad Name\ndescription: x\n---\n\nbody\n")


def test_prompt_lists_names_and_descriptions_alphabetically_not_bodies():
    skills = [AgentSkill("zeta", "Use for z.", "SECRET BODY Z"), AgentSkill("alpha", "Use for a.", "SECRET BODY A")]
    prompt = skills_prompt(skills)
    assert prompt.index("**alpha**: Use for a.") < prompt.index("**zeta**: Use for z.")
    assert "SECRET BODY" not in prompt and LOAD_SKILL_TOOL_NAME in prompt


def test_library_records_loads():
    library = SkillLibrary([AgentSkill("alpha", "Use for a.", "body a"), AgentSkill("s-zero", "Security.", "S0 body")])
    assert library.load("s-zero") == "S0 body"
    assert "no skill named" in library.load("missing")
    summary = library.summary("s-zero")
    assert summary == {
        "listed": ["alpha", "s-zero"],
        "loaded": ["s-zero"],
        "failed_loads": ["missing"],
        "security_skill": "s-zero",
        "s0_loaded": True,
    }
    assert library.summary(None)["s0_loaded"] is None
    with pytest.raises(SkillFormatError):
        SkillLibrary([AgentSkill("a", "x.", "1"), AgentSkill("a", "y.", "2")])


class LoadingModel(BaseChatModel):
    """Loads one skill, then answers; records every system prompt and tool list it was given."""

    skill: str
    systems: list[str] = Field(default_factory=list)
    step: int = 0

    @property
    def _llm_type(self) -> str:
        return "loading"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "LoadingModel":  # type: ignore[override]
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.systems.append(" ".join(str(m.content) for m in messages if isinstance(m, SystemMessage)))
        self.step += 1
        if self.step == 1:
            call = {"name": LOAD_SKILL_TOOL_NAME, "args": {"name": self.skill}, "id": "l1", "type": "tool_call"}
            message = AIMessage(content="", tool_calls=[call])
        else:
            message = AIMessage(content="Done.")
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_load_skill_through_a_deepagents_run():
    suite = get_suite("v1.2.2", "banking")
    task = suite.user_tasks["user_task_0"]
    library = SkillLibrary([AgentSkill("alpha", "Use for a.", "ALPHA BODY"), AgentSkill("s-zero", "Security.", "S0")])
    middleware = AgentSkillsMiddleware(library)
    model = LoadingModel(skill="alpha")
    pipeline = DeepAgentsPipeline(model, name="deepagents-test", tool_middleware_factory=lambda _q: [middleware])
    utility, _ = suite.run_task_with_pipeline(pipeline, task, None, {})
    assert library.summary("s-zero") == {
        "listed": ["alpha", "s-zero"],
        "loaded": ["alpha"],
        "failed_loads": [],
        "security_skill": "s-zero",
        "s0_loaded": False,
    }
    assert "**alpha**: Use for a." in model.systems[0] and "ALPHA BODY" not in model.systems[0]
    assert utility is False  # the scripted model never pays the bill
