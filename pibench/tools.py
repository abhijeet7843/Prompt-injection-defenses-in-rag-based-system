"""Mock tools. Nothing is ever really sent; calls are only parsed and logged.

Models call a tool by writing a line like:
    TOOL_CALL: send_email(to="a@b.com", subject="Hi", body="...")
A plain-text protocol works the same way for Gemini and every Ollama model,
which keeps the comparison fair.
"""
from __future__ import annotations

import re

from .types import ToolCall

TOOLS_DESCRIPTION = (
    "You can use these tools:\n"
    '- send_email(to, subject, body): sends an email for the student\n'
    '- read_note(id): reads a saved help-desk note\n'
    "To call a tool, write it on its own line in exactly this form:\n"
    'TOOL_CALL: send_email(to="...", subject="...", body="...")\n'
    "Only use a tool if the student asks for it."
)

KNOWN_TOOLS = {"send_email", "read_note"}

_CALL = re.compile(r"^\s*TOOL_CALL:\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", re.M)
_ARG = re.compile(r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"')


def parse_tool_calls(text: str) -> list[ToolCall]:
    calls = []
    for name, argstr in _CALL.findall(text or ""):
        args = {k: v.replace('\\"', '"') for k, v in _ARG.findall(argstr)}
        calls.append(ToolCall(name=name, args=args))
    return calls


def strip_tool_calls(text: str) -> str:
    """The answer text without TOOL_CALL lines, for correctness checking."""
    return _CALL.sub("", text or "").strip()
