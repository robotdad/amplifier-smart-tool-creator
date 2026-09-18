"""Private subprocess entry point. Never import the Agent runtime in the host process."""

import asyncio
import contextlib
import copy
import json
from pathlib import Path
import sys
from typing import Any

import jsonschema

from smart_tool_creator.intelligence.schemas import AgentRequest, AgentResult

MAX_INVALID_SUBMISSIONS = 2


class AgentTool:
    """Small tool surface with explicit workspace authority, not the inherited bundle tools."""

    def __init__(self, name: str, description: str, schema: dict[str, Any], handler: Any) -> None:
        self.name = name
        self.description = description
        self.input_schema = schema
        self._handler = handler

    async def execute(self, input: dict[str, Any]) -> Any:
        from amplifier_core import ToolResult

        try:
            result = await self._handler(input)
            return ToolResult(success=True, output=result)
        except Exception as error:
            return ToolResult(success=False, error={"message": f"{type(error).__name__}: {error}"})


def workspace_tools(request: AgentRequest) -> list[AgentTool]:
    if request.workspace is None:
        return []
    root = request.workspace.path.resolve()

    def resolve(path: str) -> Path:
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Path escapes the selected workspace.")
        return target

    async def read(values: dict[str, Any]) -> str:
        target = resolve(values["path"])
        if target.is_dir():
            return "\n".join(sorted(f"{p.name}/" if p.is_dir() else p.name for p in target.iterdir()))
        offset = max(0, int(values.get("offset", 0)))
        with target.open(encoding="utf-8") as handle:
            # Text offsets count characters, so a byte seek would split UTF-8.
            while offset:
                skipped = handle.read(min(offset, 50_000))
                if not skipped:
                    return ""
                offset -= len(skipped)
            return handle.read(50_000)

    async def write(values: dict[str, Any]) -> str:
        target = resolve(values["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(values["content"], encoding="utf-8")
        return "Written."

    async def bash(values: dict[str, Any]) -> str:
        # Writable requests explicitly permit host commands, not an OS sandbox.
        process = await asyncio.create_subprocess_shell(
            values["command"],
            cwd=root,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output, _ = await process.communicate()
            return f"Exit {process.returncode}\n{output.decode(errors='replace')[-50_000:]}"
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    path_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}},
        "required": ["path"],
    }
    tools = [
        AgentTool("read_file", "Read a workspace file or list a directory. Offset is in characters.", path_schema, read)
    ]
    if request.writable:
        tools.extend(
            [
                AgentTool(
                    "write_file",
                    "Write a workspace file.",
                    {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"],
                    },
                    write,
                ),
                AgentTool(
                    "bash",
                    "Run a command in the workspace. Commands are not sandboxed.",
                    {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                    bash,
                ),
            ]
        )
    return tools


async def run_agent(payload: dict[str, Any], prepared: Any = None) -> AgentResult:
    from amplifier_agent_cli.provider_sources import inject_provider
    from amplifier_agent_lib import __version__
    from amplifier_agent_lib.bundle.cache import load_and_prepare_cached
    from amplifier_agent_lib.engine import Engine
    from amplifier_agent_lib.protocol import PROTOCOL_VERSION, server_default_capabilities
    from amplifier_agent_lib.protocol_points.defaults_cli import CliApprovalSystem, CliDisplaySystem
    from amplifier_agent_lib.session_store import SessionStore
    from amplifier_foundation.bundle import Bundle

    request = AgentRequest.model_validate(payload["request"])
    session_id = payload["session_id"]
    store = SessionStore(Path(payload["state_directory"]))
    submitted: dict[str, Any] | None = None

    async def capture(values: dict[str, Any]) -> str:
        nonlocal submitted
        jsonschema.validate(values, request.output_schema)
        submitted = values
        return "Submission accepted. Finish now without calling other tools."

    try:
        async with asyncio.timeout(payload.get("timeout_seconds", request.timeout_seconds)):
            prepared = copy.copy(
                prepared if prepared is not None else await load_and_prepare_cached(aaa_version=__version__)
            )
            # Keep only the loop/context mechanisms. No user hooks, context, agents, MCP or delegation.
            prepared.mount_plan = {
                key: copy.deepcopy(value) for key, value in prepared.mount_plan.items() if key == "session"
            }
            prepared.mount_plan.update(providers=[], tools=[], hooks=[], agents={})
            prepared.mount_plan["session"] = {
                key: value
                for key, value in prepared.mount_plan["session"].items()
                if key in {"raw", "orchestrator", "context"}
            }
            prepared.bundle = Bundle(name="smart-tool-agent")
            inject_provider(
                prepared,
                payload["provider"],
                model_override=request.model,
                effort_override=request.reasoning_effort,
                extra_config={"login_on_mount": False} if payload["provider"] == "openai-chatgpt" else None,
            )
            tools = workspace_tools(request)
            if request.output_schema is not None:
                tools.append(
                    AgentTool("submit", "Submit the final answer matching this schema.", request.output_schema, capture)
                )
            resumed = request.resume is not None
            prompt = request.prompt
            for attempt in range(MAX_INVALID_SUBMISSIONS + 1):

                async def turn(ctx: Any, resumed: bool = resumed) -> str:
                    session = await prepared.create_session(
                        session_id=session_id,
                        session_cwd=request.workspace.path.resolve()
                        if request.workspace
                        else Path(payload["state_directory"]),
                        is_resumed=resumed,
                    )
                    async with session:
                        context = session.coordinator.get("context")
                        if resumed:
                            loaded = store.load(session_id)
                            if loaded is None:
                                raise ValueError("No transcript exists for this session; start a fresh run.")
                            await context.set_messages(loaded[0])
                        for tool in tools:
                            await session.coordinator.mount("tools", tool, name=tool.name)
                        reply = await session.execute(ctx.prompt)
                        store.save(session_id, await context.get_messages(), metadata={"last_turn": "complete"})
                        return reply

                engine = Engine(
                    turn_handler=turn,
                    protocol_points={
                        "approval": CliApprovalSystem(mode="no"),
                        "display": CliDisplaySystem(stream=sys.stderr, verbosity="quiet"),
                    },
                )
                try:
                    await engine.boot(
                        {
                            "protocolVersion": PROTOCOL_VERSION,
                            "capabilities": server_default_capabilities(),
                            "sessionId": session_id,
                            "resume": resumed,
                        },
                        bundle_override=prepared,
                    )
                    result = await engine.submit_turn(
                        {"sessionId": session_id, "turnId": str(attempt), "prompt": prompt}
                    )
                finally:
                    await engine.shutdown()
                text = result["reply"] or ""
                if request.output_schema is None or submitted is not None:
                    return AgentResult(output=submitted, text=text, session_id=session_id)
                resumed = True
                prompt = "No valid submission received. Call submit with an answer matching its schema."
            return AgentResult(
                error=f"No valid submission after {MAX_INVALID_SUBMISSIONS} retries.", session_id=session_id
            )
    except TimeoutError:
        return AgentResult(
            error=f"The agent did not finish within {request.timeout_seconds} seconds.", session_id=session_id
        )
    except Exception as error:
        return AgentResult(error=f"{type(error).__name__}: {error}", session_id=session_id)


def main() -> None:
    payload = json.load(sys.stdin)
    with contextlib.redirect_stdout(sys.stderr):
        try:
            from amplifier_agent_cli.provider_sources import KNOWN_PROVIDERS, resolve_credential_detailed

            provider = payload["provider"]
            if provider not in KNOWN_PROVIDERS:
                raise ValueError(f"Unknown provider {provider!r}. Choose one of: {', '.join(KNOWN_PROVIDERS)}.")
            resolution = resolve_credential_detailed(provider)
            if not resolution.resolved and not (provider == "ollama" and resolution.source == "default"):
                remedy = (
                    "Complete the provider's OAuth device-code login outside this tool first; "
                    "see https://github.com/microsoft/amplifier-agent/blob/v0.17.0/docs/CONFIGURATION.md. "
                    "Interactive login is disabled here."
                    if provider == "openai-chatgpt"
                    else f"Set {resolution.env_var} to the provider's credential or server endpoint."
                )
                raise ValueError(f"Provider {provider!r} has no usable configuration. {remedy}")
            result = AgentResult() if payload.get("preflight") else asyncio.run(run_agent(payload))
        except Exception as error:
            result = AgentResult(
                error=f"{type(error).__name__}: {error}. Configure credentials with `amplifier-agent auth` or the provider's environment variables.",
                session_id=payload.get("session_id"),
            )
    print(result.model_dump_json())


if __name__ == "__main__":
    main()
