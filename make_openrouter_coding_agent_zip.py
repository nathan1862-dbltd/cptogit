from pathlib import Path
import zipfile
import textwrap

PROJECT = "openrouter-coding-agent"

files = {
    "agent.py": r'''
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

WORKSPACE = Path(os.getenv("WORKSPACE", "/workspace")).resolve()
ALLOW_SHELL = os.getenv("ALLOW_SHELL", "false").lower() == "true"

MAX_FILE_READ_CHARS = int(os.getenv("MAX_FILE_READ_CHARS", "40000"))
MAX_TOOL_OUTPUT_CHARS = int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "30000"))


def clip(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n... truncated, total length was {len(text)} characters ..."


def ensure_workspace() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)


def safe_path(path: str) -> Path:
    ensure_workspace()
    requested = (WORKSPACE / path).resolve()

    try:
        requested.relative_to(WORKSPACE)
    except ValueError:
        raise ValueError("Access denied: path is outside the workspace.")

    return requested


def list_files(path: str = ".") -> str:
    target = safe_path(path)

    if not target.exists():
        return f"Path does not exist: {path}"

    if target.is_file():
        return str(target.relative_to(WORKSPACE))

    results = []
    max_items = 300

    for item in sorted(target.rglob("*")):
        if len(results) >= max_items:
            results.append("... truncated ...")
            break

        rel = item.relative_to(WORKSPACE)
        if any(part in {".git", "__pycache__", "node_modules", ".venv"} for part in rel.parts):
            continue

        suffix = "/" if item.is_dir() else ""
        results.append(f"{rel}{suffix}")

    return "\n".join(results) if results else "No files found."


def read_file(path: str) -> str:
    target = safe_path(path)

    if not target.exists():
        return f"File does not exist: {path}"

    if not target.is_file():
        return f"Not a file: {path}"

    content = target.read_text(encoding="utf-8", errors="replace")
    return clip(content, MAX_FILE_READ_CHARS)


def write_file(path: str, content: str) -> str:
    target = safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote file: {target.relative_to(WORKSPACE)}"


def edit_file(path: str, old: str, new: str, replace_all: bool = False) -> str:
    target = safe_path(path)

    if not target.exists():
        return f"File does not exist: {path}"

    if not target.is_file():
        return f"Not a file: {path}"

    content = target.read_text(encoding="utf-8", errors="replace")

    if old not in content:
        return f"Text to replace was not found in: {path}"

    if replace_all:
        updated = content.replace(old, new)
        count = content.count(old)
    else:
        updated = content.replace(old, new, 1)
        count = 1

    target.write_text(updated, encoding="utf-8")
    return f"Edited file: {target.relative_to(WORKSPACE)}. Replacements: {count}"


def run_shell(command: str, timeout: int = 120) -> str:
    if not ALLOW_SHELL:
        return (
            "Shell execution is disabled. "
            "Set ALLOW_SHELL=true in .env to enable it inside the Docker container."
        )

    try:
        process = subprocess.run(
            command,
            cwd=str(WORKSPACE),
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

        result = {
            "command": command,
            "exit_code": process.returncode,
            "stdout": clip(process.stdout),
            "stderr": clip(process.stderr),
        }

        return json.dumps(result, indent=2)
    except subprocess.TimeoutExpired as exc:
        return f"Command timed out after {timeout} seconds: {exc}"
    except Exception as exc:
        return f"Command failed: {exc}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and folders inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the workspace.",
                        "default": ".",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace text inside an existing file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace.",
                    },
                    "old": {
                        "type": "string",
                        "description": "Exact text to replace.",
                    },
                    "new": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences instead of only the first.",
                        "default": False,
                    },
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command inside the workspace Docker container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds.",
                        "default": 120,
                    },
                },
                "required": ["command"],
            },
        },
    },
]


def call_tool(name: str, args: Dict[str, Any]) -> str:
    try:
        if name == "list_files":
            return list_files(args.get("path", "."))
        if name == "read_file":
            return read_file(args["path"])
        if name == "write_file":
            return write_file(args["path"], args["content"])
        if name == "edit_file":
            return edit_file(
                args["path"],
                args["old"],
                args["new"],
                bool(args.get("replace_all", False)),
            )
        if name == "run_shell":
            return run_shell(args["command"], int(args.get("timeout", 120)))

        return f"Unknown tool: {name}"
    except Exception as exc:
        return f"Tool error in {name}: {exc}"


class CodingAgent:
    def __init__(self) -> None:
        if not OPENROUTER_API_KEY:
            raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")

        headers = {}

        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_APP_TITLE")

        if referer:
            headers["HTTP-Referer"] = referer

        if title:
            headers["X-Title"] = title

        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            default_headers=headers,
        )

    def run(self, task: str, max_steps: int = 25) -> str:
        ensure_workspace()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a practical coding agent operating inside a Docker workspace. "
                    "Your workspace root is /workspace. "
                    "Use tools to inspect, create, edit, and test code. "
                    "Never access files outside the workspace. "
                    "Before editing an existing project, inspect files first. "
                    "Prefer small, correct changes. "
                    "If shell access is enabled, run useful tests or commands. "
                    "When finished, summarize what you changed and how to run it."
                ),
            },
            {
                "role": "user",
                "content": task,
            },
        ]

        for _ in range(max_steps):
            response = self.client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )

            message = response.choices[0].message

            assistant_message = {
                "role": "assistant",
                "content": message.content,
            }

            if message.tool_calls:
                assistant_message["tool_calls"] = [
                    tool_call.model_dump() for tool_call in message.tool_calls
                ]

            messages.append(assistant_message)

            if not message.tool_calls:
                return message.content or "Done."

            for tool_call in message.tool_calls:
                name = tool_call.function.name

                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                result = call_tool(name, args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": result,
                    }
                )

        return "Stopped because max_steps was reached."


class RunRequest(BaseModel):
    task: str = Field(..., description="Coding task for the agent.")
    max_steps: int = Field(25, ge=1, le=50)


class RunResponse(BaseModel):
    result: str
    model: str
    workspace: str
    shell_enabled: bool


app = FastAPI(title="OpenRouter Coding Agent")


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": OPENROUTER_MODEL,
        "workspace": str(WORKSPACE),
        "shell_enabled": ALLOW_SHELL,
    }


@app.post("/run", response_model=RunResponse)
def run_agent(request: RunRequest):
    try:
        agent = CodingAgent()
        result = agent.run(request.task, request.max_steps)

        return RunResponse(
            result=result,
            model=OPENROUTER_MODEL,
            workspace=str(WORKSPACE),
            shell_enabled=ALLOW_SHELL,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print('  python agent.py "Create a FastAPI hello world app"')
        raise SystemExit(1)

    task = " ".join(sys.argv[1:])
    agent = CodingAgent()
    print(agent.run(task))
''',

    "requirements.txt": r'''
openai>=1.55.0
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.8.0
''',

    "Dockerfile": r'''
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    bash \
    curl \
    git \
    ripgrep \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY agent.py /app/agent.py

RUN mkdir -p /workspace

ENV WORKSPACE=/workspace
ENV OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

EXPOSE 8000

CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]
''',

    "docker-compose.yml": r'''
services:
  coding-agent:
    build: .
    container_name: openrouter-coding-agent
    env_file:
      - .env
    ports:
      - "8000:8000"
    volumes:
      - ./workspace:/workspace
    environment:
      WORKSPACE: /workspace
      OPENROUTER_BASE_URL: https://openrouter.ai/api/v1
      ALLOW_SHELL: ${ALLOW_SHELL:-false}
    restart: unless-stopped
''',

    ".env.example": r'''
OPENROUTER_API_KEY=sk-or-v1-your-key-here

OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

OPENROUTER_HTTP_REFERER=http://localhost:8000
OPENROUTER_APP_TITLE=OpenRouter Coding Agent

ALLOW_SHELL=true
''',

    ".dockerignore": r'''
.env
workspace/*
__pycache__/
*.pyc
.git/
.venv/
node_modules/
''',

    "README.md": r'''
# OpenRouter Coding Agent

A lightweight Codex-style coding agent using OpenRouter.

## Setup

Copy the environment file:

```bash
cp .env.example .env