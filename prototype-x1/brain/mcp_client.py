"""
MCP (Model Context Protocol) client for ARIA.
Connects to MCP servers via stdio transport, discovers tools, and calls them.
Follows the MCP spec: JSON-RPC 2.0 over stdin/stdout.

Usage:
    client = MCPClient("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
    await client.connect()
    tools = await client.list_tools()
    result = await client.call_tool("read_file", path="/tmp/foo.txt")
    await client.disconnect()
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("x1.mcp")

class MCPError(Exception):
    """Error returned by an MCP server."""
    def __init__(self, error_data: dict) -> None:
        self.code = error_data.get("code", -1)
        self.message = error_data.get("message", "Unknown MCP error")
        super().__init__(f"MCP error {self.code}: {self.message}")


_MSG_ID = 0


def _next_id() -> int:
    global _MSG_ID
    _MSG_ID += 1
    return _MSG_ID


@dataclass
class MCPTool:
    """A tool discovered from an MCP server."""
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)
    server_name: str = ""


class MCPClient:
    """Async MCP client — stdio transport."""

    def __init__(self, command: str, args: list[str] | None = None,
                 server_name: str = "", env: dict[str, str] | None = None) -> None:
        self.command = command
        self.args = args or []
        self.server_name = server_name or command
        self.env = env
        self._process: asyncio.subprocess.Process | None = None
        self._tools: list[MCPTool] = []
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None
        self._connected = False

    async def connect(self) -> None:
        """Launch the MCP server process and perform initialization handshake."""
        log.info("Connecting to MCP server: %s %s", self.command, self.args)
        self._process = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
        )
        self._reader_task = asyncio.create_task(self._read_loop())

        # Initialize handshake
        result = await self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "aria-x1", "version": "0.1.0"},
        })
        log.info("MCP server initialized: %s", result.get("serverInfo", {}))

        # Send initialized notification (no response expected)
        await self._notify("notifications/initialized", {})
        self._connected = True

    async def _read_loop(self) -> None:
        """Read JSON-RPC messages from server stdout."""
        assert self._process and self._process.stdout
        while True:
            line = await self._process.stdout.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                log.debug("Non-JSON from MCP server: %s", line[:200])
                continue

            msg_id = msg.get("id")
            if msg_id is not None and msg_id in self._pending:
                fut = self._pending.pop(msg_id)
                if "error" in msg:
                    fut.set_exception(MCPError(msg["error"]))
                else:
                    fut.set_result(msg.get("result", {}))

    async def _request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and wait for response."""
        msg_id = _next_id()
        msg = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params}
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = fut
        self._send(msg)
        return await asyncio.wait_for(fut, timeout=30.0)

    async def _notify(self, method: str, params: dict) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        self._send(msg)

    def _send(self, msg: dict) -> None:
        """Write a JSON-RPC message to server stdin."""
        assert self._process and self._process.stdin
        data = json.dumps(msg) + "\n"
        self._process.stdin.write(data.encode())

    async def list_tools(self) -> list[MCPTool]:
        """Discover tools from the MCP server."""
        result = await self._request("tools/list", {})
        self._tools = [
            MCPTool(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
                server_name=self.server_name,
            )
            for t in result.get("tools", [])
        ]
        log.info("Discovered %d tools from %s", len(self._tools), self.server_name)
        return self._tools

    async def call_tool(self, tool_name: str, **kwargs: Any) -> str:
        """Call a tool on the MCP server."""
        result = await self._request("tools/call", {
            "name": tool_name, "arguments": kwargs,
        })
        # MCP returns content as list of {type, text} blocks
        content = result.get("content", [])
        parts = [c.get("text", "") for c in content if c.get("type") == "text"]
        return "\n".join(parts) if parts else json.dumps(result)

    async def disconnect(self) -> None:
        """Shut down the MCP server."""
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                self._process.kill()
            self._process = None
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None
        self._connected = False
        log.info("Disconnected from MCP server: %s", self.server_name)

