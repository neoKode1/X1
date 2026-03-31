"""
MCP Manager — connects MCP servers to ARIA's tool registry.

Manages multiple MCP server connections and registers their tools
so the Brain can call them like any other skill.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from .mcp_client import MCPClient, MCPTool
from .tools import ToolRegistry

log = logging.getLogger("x1.mcp_manager")


@dataclass
class ServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: list[str] | None = None
    env: dict[str, str] | None = None


class MCPManager:
    """Manages MCP server connections and integrates tools into a ToolRegistry."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._clients: dict[str, MCPClient] = {}
        self._mcp_tools: dict[str, MCPTool] = {}  # tool_name -> MCPTool

    async def connect_server(self, config: ServerConfig) -> list[str]:
        """Connect to an MCP server and register its tools. Returns tool names."""
        client = MCPClient(
            command=config.command,
            args=config.args or [],
            server_name=config.name,
            env=config.env,
        )
        await client.connect()
        self._clients[config.name] = client

        tools = await client.list_tools()
        registered = []
        for tool in tools:
            # Prefix with server name to avoid collisions
            full_name = f"mcp.{config.name}.{tool.name}"
            self._mcp_tools[full_name] = tool

            # Build parameter schema from MCP inputSchema
            params = {}
            props = tool.input_schema.get("properties", {})
            required = tool.input_schema.get("required", [])
            for pname, pdef in props.items():
                params[pname] = {
                    "type": pdef.get("type", "string"),
                    "description": pdef.get("description", pname),
                    "required": pname in required,
                }

            # Register as a tool that calls through MCP
            self._registry.register(
                name=full_name,
                fn=self._make_caller(config.name, tool.name),
                description=f"[MCP:{config.name}] {tool.description}",
                parameters=params,
            )
            registered.append(full_name)

        log.info("Registered %d tools from MCP server '%s'", len(registered), config.name)
        return registered

    def _make_caller(self, server_name: str, tool_name: str):
        """Create a sync wrapper that calls an MCP tool."""
        def _call(**kwargs: Any) -> str:
            client = self._clients.get(server_name)
            if not client:
                return f"[MCP ERROR] Server '{server_name}' not connected"
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an async context — schedule and wait
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(
                            asyncio.run, client.call_tool(tool_name, **kwargs)
                        )
                        return future.result(timeout=30)
                else:
                    return loop.run_until_complete(
                        client.call_tool(tool_name, **kwargs)
                    )
            except Exception as e:
                return f"[MCP ERROR] {tool_name}: {e}"
        _call.__doc__ = f"MCP tool: {tool_name} via {server_name}"
        return _call

    async def disconnect_all(self) -> None:
        """Disconnect all MCP servers."""
        for name, client in list(self._clients.items()):
            await client.disconnect()
            log.info("Disconnected MCP server: %s", name)
        self._clients.clear()
        # Remove MCP tools from registry
        for tool_name in list(self._mcp_tools.keys()):
            if tool_name in self._registry._tools:
                del self._registry._tools[tool_name]
        self._mcp_tools.clear()

    @property
    def connected_servers(self) -> list[str]:
        return list(self._clients.keys())

    @property
    def mcp_tool_count(self) -> int:
        return len(self._mcp_tools)

