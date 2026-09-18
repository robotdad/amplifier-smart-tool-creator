"""Real Engine lifecycle with scripted sessions, plus workspace tool boundaries."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("amplifier_agent_lib")

from smart_tool_creator.intelligence.amplifier_worker import run_agent, workspace_tools
from smart_tool_creator.intelligence.schemas import AgentRequest, HostWorkspace

SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}


class ScriptedPrepared:
    def __init__(self, actions: list[Any]) -> None:
        self.actions = actions
        self.sessions: list[Any] = []
        self.mount_plan: dict[str, Any] = {
            "session": {},
            "providers": ["unwanted"],
            "tools": ["bash"],
            "hooks": ["unsafe"],
        }
        self.bundle = SimpleNamespace(context={"unsafe": "context"})

    async def create_session(self, **kwargs: Any) -> Any:
        assert self.mount_plan["tools"] == []
        assert self.mount_plan["hooks"] == []
        assert self.mount_plan["agents"] == {}
        assert not self.bundle.context
        provider = self.mount_plan["providers"][0]
        assert provider["config"]["default_model"] == "explicit-model"
        assert provider["config"]["effort"] == "high"
        session = ScriptedSession(self.actions.pop(0))
        self.sessions.append(session)
        return session


class ScriptedSession:
    def __init__(self, action: Any) -> None:
        self.action = action
        self.tools: dict[str, Any] = {}
        self.messages: list[dict] = []
        self.closed = False
        self.coordinator = self
        self.loaded = False

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.closed = True

    def get(self, name: str) -> Any:
        assert name == "context"
        return self

    async def mount(self, kind: str, tool: Any, name: str) -> None:
        assert kind == "tools"
        self.tools[name] = tool

    async def set_messages(self, messages: list[dict]) -> None:
        self.messages = messages
        self.loaded = True

    async def get_messages(self) -> list[dict]:
        return self.messages

    async def execute(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})
        if isinstance(self.action, dict):
            await self.tools["submit"].execute(self.action)
        elif isinstance(self.action, Exception):
            raise self.action
        elif self.action == "sleep":
            await asyncio.sleep(30)
        self.messages.append({"role": "assistant", "content": "reply"})
        return "reply"


def payload(tmp_path: Path, structured: bool = False, resume: str | None = None) -> dict[str, Any]:
    request = AgentRequest(
        prompt="Test",
        model="explicit-model",
        reasoning_effort="high",
        timeout_seconds=1,
        output_schema=SCHEMA if structured else None,
        resume=resume,
    )
    return {
        "provider": "ollama",
        "session_id": "session-1",
        "state_directory": str(tmp_path),
        "request": request.model_dump(mode="json"),
    }


async def test_text_and_resume_replay(tmp_path: Path) -> None:
    prepared = ScriptedPrepared(["text", "text"])
    first = await run_agent(payload(tmp_path), prepared)
    assert first.text == "reply"
    assert first.error is None
    second = await run_agent(payload(tmp_path, resume=first.session_id), prepared)
    assert second.error is None
    assert prepared.sessions[1].loaded
    assert len(prepared.sessions[1].messages) == 4
    assert prepared.sessions[0].tools == {}
    assert all(s.closed for s in prepared.sessions)


async def test_invalid_submission_repair_then_valid(tmp_path: Path) -> None:
    prepared = ScriptedPrepared([{"answer": "wrong"}, None, {"answer": 42}])
    result = await run_agent(payload(tmp_path, structured=True), prepared)
    assert result.output == {"answer": 42}
    assert result.error is None
    assert len(prepared.sessions) == 3
    assert prepared.sessions[1].loaded
    assert prepared.sessions[2].loaded
    assert set(prepared.sessions[0].tools) == {"submit"}


async def test_missing_submission_is_bounded(tmp_path: Path) -> None:
    prepared = ScriptedPrepared([None] * 3)
    result = await run_agent(payload(tmp_path, structured=True), prepared)
    assert result.output is None
    assert result.error
    assert "2 retries" in result.error
    assert len(prepared.sessions) == 3


async def test_chatgpt_does_not_start_interactive_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from amplifier_agent_cli import provider_sources

    original = provider_sources.inject_provider

    def inject(prepared: Any, provider: str, **kwargs: Any) -> None:
        assert provider == "openai-chatgpt"
        assert kwargs["extra_config"] == {"login_on_mount": False}
        original(prepared, "ollama", **kwargs)

    monkeypatch.setattr(provider_sources, "inject_provider", inject)
    values = payload(tmp_path)
    values["provider"] = "openai-chatgpt"
    result = await run_agent(values, ScriptedPrepared(["text"]))
    assert result.error is None


@pytest.mark.parametrize("action", ["sleep", RuntimeError("provider failed")])
async def test_timeout_and_error_cleanup(tmp_path: Path, action: Any) -> None:
    prepared = ScriptedPrepared([action])
    result = await run_agent(payload(tmp_path), prepared)
    assert result.error
    assert result.session_id == "session-1"
    assert prepared.sessions[0].closed


async def test_resume_without_history_is_not_faked(tmp_path: Path) -> None:
    prepared = ScriptedPrepared([None])
    result = await run_agent(payload(tmp_path, resume="session-1"), prepared)
    assert result.error
    assert "No transcript" in result.error
    assert prepared.sessions[0].closed


async def test_workspace_permissions_and_escape(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "example.txt").write_text("hello")
    request = AgentRequest(prompt="Test", model="model", timeout_seconds=1, workspace=HostWorkspace(path=root))
    tools = workspace_tools(request)
    assert [t.name for t in tools] == ["read_file"]
    assert (await tools[0].execute({"path": "example.txt"})).output == "hello"
    assert not (await tools[0].execute({"path": "../secret"})).success
    (root / "link").symlink_to(tmp_path)
    assert not (await tools[0].execute({"path": "link/secret"})).success
    writable = workspace_tools(request.model_copy(update={"writable": True}))
    assert [t.name for t in writable] == ["read_file", "write_file", "bash"]
    assert (await writable[1].execute({"path": "new.txt", "content": "written"})).success
    assert (root / "new.txt").read_text() == "written"
    assert not (await writable[1].execute({"path": "../escape", "content": "no"})).success


@pytest.mark.parametrize("offset", [1_999_993, 2_000_000, 2_030_000, 2_100_000])
async def test_workspace_read_pages_unicode_files_beyond_two_million_characters(
    tmp_path: Path, offset: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = "\u00e9" * 2_000_000 + "TAIL_MARKER" + "\u7d42" * 60_000
    source = tmp_path / "large.txt"
    source.write_text(text, encoding="utf-8")
    original_open = Path.open
    reads: list[int] = []

    class BoundedReader:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            self.handle.close()

        def read(self, size: int = -1) -> str:
            reads.append(size)
            assert 0 < size <= 50_000, "Paging must not buffer the whole prefix or file."
            return self.handle.read(size)

    def tracked_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        handle = original_open(path, *args, **kwargs)
        return BoundedReader(handle) if path == source else handle

    monkeypatch.setattr(Path, "open", tracked_open)
    request = AgentRequest(prompt="Test", model="scripted", timeout_seconds=5, workspace=HostWorkspace(path=tmp_path))
    result = await workspace_tools(request)[0].execute({"path": "large.txt", "offset": offset})
    assert result.success, result.error
    assert result.output == text[offset : offset + 50_000]
    assert reads
