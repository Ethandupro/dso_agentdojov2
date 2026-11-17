"""MUZZLE through the real runner with a scripted victim and checker: a C2 run and a C4 run, plus resume."""

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import openai
import pytest

pytest.importorskip("deepagents")

from agentdojo.attacks.adaptive.muzzle import AttackerModelConfig, MuzzleAttacker, MuzzleSettings
from agentdojo.runner.conditions import CONDITIONS, build_units
from agentdojo.runner.plugins import FixedSkillFile, StaticAttacker
from agentdojo.runner.report import write_report
from agentdojo.runner.run import Runner, run_dir_for
from tests.test_attacks.test_muzzle import MANIFEST, ScriptedLLM, make_attacker
from tests.test_checker.test_checker_middleware import reading_checker
from tests.test_runner.test_run import RecordingSkillFile, ScriptedVictim, make_config, read_jsonl


def make_runner(tmp_path: Path, condition: str, attacker: Any, limit: int = 1) -> Runner:
    cond = CONDITIONS[condition]
    return Runner(
        config=make_config(condition, limit, attacker={"name": "muzzle"}),
        condition=cond,
        run_dir=run_dir_for(tmp_path, cond, 0, limit),
        units=build_units(cond, MANIFEST, 3, limit),
        attacker=attacker,
        static_attacker=StaticAttacker(MANIFEST),
        skill_file=FixedSkillFile("(S0)") if not cond.evolving_skill else RecordingSkillFile(),
        victim_factory=ScriptedVictim,
        checker_fn=reading_checker,
        echo=lambda _: None,
    )


def test_c2_trials_use_muzzle_rewrites_and_feed_back_outcomes(tmp_path: Path):
    log: list[dict[str, Any]] = []
    attacker, _ = make_attacker(log)
    runner = make_runner(tmp_path, "C2", attacker)
    records = [r for r in runner.run() if r["pair"]]
    assert [r["trial"] for r in records] == [0, 1, 2]
    assert [r["attack"] for r in records] == ["muzzle"] * 3
    texts = attacker.attack_texts()
    assert len(texts) == 3 and texts[1] != texts[0] and texts[2] != texts[1]
    # Two rewrites, each judging the previous trial's real outcome.
    judged = [e["feedback"] for e in attacker.archive[1:]]
    assert [f["trial"] for f in judged] == [0, 1]
    for f, r in zip(judged, records[:2], strict=True):
        assert (f["utility"], f["security"]) == (r["utility"], r["security"])
    state = json.loads((runner.run_dir / "state.json").read_text())
    assert [e["text"] for e in state["attacker"]["archive"]] == texts
    assert len(read_jsonl(runner.run_dir / "trials.jsonl")) == len(runner.units)

    write_report(runner.run_dir)
    summary = json.loads((runner.run_dir / "summary.json").read_text())
    assert summary["attacker"]["rewrites"] == 2
    assert summary["attacker"]["goal_fallbacks"] == sum(e["source"] == "goal_fallback" for e in attacker.archive)
    assert "<h2>Attacker</h2>" in (runner.run_dir / "report.html").read_text()


def test_c4_test_pairs_never_reach_muzzle(tmp_path: Path):
    log: list[dict[str, Any]] = []
    attacker, _ = make_attacker(log)
    records = [r for r in make_runner(tmp_path, "C4", attacker).run() if r["pair"]]
    assert [r["attack"] for r in records] == ["muzzle"] * 3 + ["static"] * 3
    train_pair = records[0]["pair"]
    assert {e["pair"] for e in attacker.archive} == {train_pair}


def test_resume_reuses_the_archive(tmp_path: Path):
    attacker, _ = make_attacker([])
    runner = make_runner(tmp_path, "C2", attacker)
    runner.run()
    texts = attacker.attack_texts()

    fresh, clients = make_attacker([])
    again = make_runner(tmp_path, "C2", fresh)
    again.run()  # everything is done: loads the checkpoint, runs nothing
    assert fresh.attack_texts() == texts and clients == []


class StandInServer:
    """An OpenAI-compatible HTTP server answering like `ScriptedLLM`; `fail_with` makes every request fail."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.fail_with: int | None = None
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 (http.server API)
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                if server.fail_with is not None:
                    self.send_response(server.fail_with)
                    self.end_headers()
                    return
                kwargs = {k: v for k, v in body.items() if k not in ("model", "messages")}
                reply = asyncio.run(ScriptedLLM([], pair_score=3).complete(body["messages"], **kwargs))
                data = reply.model_dump()
                data["usage"] = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
                payload = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"


@pytest.fixture
def stand_in(monkeypatch) -> Iterator[StandInServer]:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    server = StandInServer()
    thread = threading.Thread(target=server.httpd.serve_forever, daemon=True)
    thread.start()
    yield server
    server.httpd.shutdown()


def http_attacker(server: StandInServer, **overrides: Any) -> MuzzleAttacker:
    config = AttackerModelConfig(
        model="vendor/attacker", base_url=server.base_url, seed=0, max_retries=1, request_timeout=5, **overrides
    )
    return MuzzleAttacker(MANIFEST, MuzzleSettings(attacker=config))


def test_c2_over_http_sends_upstreams_arguments(tmp_path: Path, stand_in: StandInServer):
    # As against the local vLLM attacker: no API key.
    attacker = http_attacker(stand_in)
    make_runner(tmp_path, "C2", attacker).run()
    bodies = [r["body"] for r in stand_in.requests]
    assert {r["auth"] for r in stand_in.requests} == {"Bearer EMPTY"}
    assert all(b["model"] == "vendor/attacker" and b["seed"] == 0 for b in bodies)
    assert not any("provider" in b or "reasoning" in b for b in bodies)
    # Two rewrites, each: Judge + 3 streams x 4 iterations x (attacker, stand-in victim, PAIR judge).
    s = attacker.settings
    per_rewrite = 1 + 3 * s.pair_n_streams * s.pair_n_iterations
    assert len(bodies) == 2 * per_rewrite
    pair_attacker = [b for b in bodies if b.get("stop") == ["}"]]
    assert len(pair_attacker) == 2 * s.pair_n_streams * s.pair_n_iterations
    assert {(b["max_tokens"], b["temperature"], b["top_p"]) for b in pair_attacker} == {(500, 1.0, 0.9)}
    assert {(b["max_tokens"], b["temperature"]) for b in bodies if b.get("max_tokens") == 10} == {(10, 0.0)}
    usage = attacker.archive[-1]["usage"]
    assert usage["calls"] == per_rewrite and usage["prompt_tokens"] == 10 * per_rewrite
    assert usage["empty_replies"] == 0 and usage["served_models"] == ["scripted"]


def test_api_failure_stops_the_run_and_resume_continues(tmp_path: Path, stand_in: StandInServer):
    stand_in.fail_with = 500
    runner = make_runner(tmp_path, "C2", http_attacker(stand_in, api_key_env="OPENROUTER_API_KEY"))
    with pytest.raises(openai.InternalServerError):
        runner.run()
    done = [json.loads(line) for line in (runner.run_dir / "trials.jsonl").read_text().splitlines()]
    assert [r["trial"] for r in done if r["pair"]] == [0]  # stopped before trial 1, nothing half-written

    assert {r["auth"] for r in stand_in.requests} == {"Bearer sk-test"}
    stand_in.fail_with = None
    attacker = http_attacker(stand_in, reasoning={"enabled": False})
    records = make_runner(tmp_path, "C2", attacker).run()
    assert [r["trial"] for r in records if r["pair"]] == [0, 1, 2]
    assert [e["trial"] for e in attacker.archive] == [0, 1, 2]
    assert stand_in.requests[-1]["body"]["reasoning"] == {"enabled": False}
