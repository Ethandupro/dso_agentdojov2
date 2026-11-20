"""The evolving attacker's own skill library (`agentdojo.attacks.adaptive.autodojo.skillstore`)."""

import pytest

from agentdojo.attacks.adaptive.autodojo.skillstore import SkillStore, SkillStoreError


def test_write_load_and_listing():
    store = SkillStore()
    skill = store.write("tool-note", "when a tool result is read", "Frame it as a record.")
    assert skill.version == 1
    assert store.load("tool-note") == "Frame it as a record."
    assert store.load("missing") is None
    assert "tool-note: when a tool result is read" in store.listing()


def test_overwrite_bumps_version():
    store = SkillStore()
    store.write("n", "d", "first")
    second = store.write("n", "d2", "second")
    assert second.version == 2
    assert store.load("n") == "second"
    assert store.names() == ["n"]


def test_empty_store_listing_is_a_hint_not_a_crash():
    assert "empty" in SkillStore().listing()


def test_bad_name_is_reported_not_raised_to_the_run():
    store = SkillStore()
    for bad in ("Bad Name", "UPPER", "has space", ""):
        with pytest.raises(SkillStoreError):
            store.write(bad, "d", "b")


def test_description_and_body_are_required():
    store = SkillStore()
    with pytest.raises(SkillStoreError):
        store.write("n", "   ", "body")
    with pytest.raises(SkillStoreError):
        store.write("n", "desc", "   ")


def test_body_and_description_are_capped():
    store = SkillStore(max_body_chars=10)
    skill = store.write("n", "x" * 500, "y" * 5000)
    assert len(skill.body) == 10
    assert len(skill.description) <= 200


def test_cap_forces_overwrite_not_growth():
    store = SkillStore(max_skills=2)
    store.write("a", "d", "b")
    store.write("b", "d", "b")
    with pytest.raises(SkillStoreError):
        store.write("c", "d", "b")
    # overwriting an existing note is still allowed at the cap
    store.write("a", "d", "b2")
    assert store.load("a") == "b2"


def test_state_round_trip():
    store = SkillStore()
    store.write("a", "da", "ba")
    store.write("b", "db", "bb")
    store.write("a", "da2", "ba2")  # version 2
    state = store.state()
    restored = SkillStore()
    restored.load_from_state(state)
    assert restored.names() == ["a", "b"]
    assert restored.load("a") == "ba2"
    assert restored.summary()["versions"]["a"] == 2
