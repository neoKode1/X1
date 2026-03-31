"""
Tool registry for ARIA's brain.
Replaces regex-based skill detection with structured JSON function calling.

Tools are registered with name, description, parameter schema, and a callable.
The registry generates tool schemas for LLM prompts and dispatches calls.
"""
from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("x1.tools")


@dataclass
class ToolDef:
    """A registered tool definition."""
    name: str
    description: str
    parameters: dict[str, dict]  # param_name -> {"type": str, "description": str, "required": bool}
    fn: Callable[..., str]


class ToolRegistry:
    """Central registry for all ARIA tools (local skills + MCP + agents)."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(self, name: str, fn: Callable[..., str],
                 description: str | None = None,
                 parameters: dict[str, dict] | None = None) -> None:
        """Register a tool. Auto-extracts description and params from docstring/signature if not provided."""
        if description is None:
            description = (fn.__doc__ or "").strip().split("\n")[0]

        if parameters is None:
            parameters = self._extract_params(fn)

        self._tools[name] = ToolDef(
            name=name, description=description,
            parameters=parameters, fn=fn,
        )
        log.info("Registered tool: %s", name)

    def _extract_params(self, fn: Callable) -> dict[str, dict]:
        """Auto-extract parameters from function signature."""
        params = {}
        sig = inspect.signature(fn)
        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            ptype = "string"
            if param.annotation is int:
                ptype = "integer"
            elif param.annotation is float:
                ptype = "number"
            elif param.annotation is bool:
                ptype = "boolean"
            params[pname] = {
                "type": ptype,
                "description": pname,
                "required": param.default is inspect.Parameter.empty,
            }
        return params

    def call(self, tool_name: str, **kwargs: Any) -> str:
        """Call a registered tool by name."""
        tool = self._tools.get(tool_name)
        if not tool:
            return f"[TOOL ERROR] Unknown tool: {tool_name}"
        try:
            # Coerce types based on schema
            coerced = {}
            for k, v in kwargs.items():
                if k in tool.parameters:
                    ptype = tool.parameters[k].get("type", "string")
                    if ptype == "integer" and not isinstance(v, int):
                        v = int(v)
                    elif ptype == "number" and not isinstance(v, (int, float)):
                        v = float(v)
                    elif ptype == "boolean" and not isinstance(v, bool):
                        v = str(v).lower() in ("true", "1", "yes")
                coerced[k] = v
            return tool.fn(**coerced)
        except Exception as e:
            log.error("Tool %s failed: %s", name, e)
            return f"[TOOL ERROR] {name}: {e}"

    def list_tools(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    def get_schema(self) -> list[dict]:
        """Generate JSON schemas for all tools (for LLM system prompt)."""
        schemas = []
        for tool in self._tools.values():
            props = {}
            required = []
            for pname, pdef in tool.parameters.items():
                props[pname] = {"type": pdef["type"], "description": pdef["description"]}
                if pdef.get("required"):
                    required.append(pname)
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": {"type": "object", "properties": props, "required": required},
            })
        return schemas

    def get_prompt_block(self) -> str:
        """Generate a tool description block for the system prompt."""
        if not self._tools:
            return ""
        schemas = self.get_schema()
        lines = ["TOOLS: You can call tools by responding with a JSON block like this:",
                 '```tool_call', '{"name": "tool_name", "args": {"param": "value"}}', '```',
                 "", "Available tools:"]
        for s in schemas:
            params_desc = ", ".join(
                f'{p} ({d["type"]})' for p, d in s["parameters"]["properties"].items()
            )
            lines.append(f'- {s["name"]}: {s["description"]}  params: {params_desc}')
        lines.append("")
        lines.append("Only use tool_call blocks when you need to execute a tool. Respond normally for conversation.")
        return "\n".join(lines)

    def clear(self) -> None:
        self._tools.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

