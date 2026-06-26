from __future__ import annotations
import json
from typing import Any, Callable, Dict, List, Optional


class MCPTool:
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[..., Any],
        server_name: str = "",
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler
        self.server_name = server_name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "server":      self.server_name,
            "name":        self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


class MCPResponse:
    def __init__(self, content: List[Dict[str, Any]], is_error: bool = False):
        self.content = content
        self.is_error = is_error

    @classmethod
    def text(cls, text: str) -> "MCPResponse":
        return cls([{"type": "text", "text": text}])

    @classmethod
    def json_data(cls, data: Any) -> "MCPResponse":
        return cls([{"type": "text", "text": json.dumps(data, indent=2)}])

    @classmethod
    def error(cls, message: str) -> "MCPResponse":
        return cls([{"type": "error", "text": message}], is_error=True)

    def get_text(self) -> str:
        return "\n".join(
            b["text"] for b in self.content if b.get("type") in ("text", "error")
        )

    def get_data(self) -> Any:
        text = self.get_text()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text


class BaseMCPServer:
    def __init__(self, server_name: str, server_description: str):
        self.server_name = server_name
        self.server_description = server_description
        self._tools: Dict[str, MCPTool] = {}
        self._register_tools()

    def _register_tools(self) -> None:
        for attr_name in dir(self):
            attr = getattr(self, attr_name, None)
            if callable(attr) and hasattr(attr, "_mcp_tool_meta"):
                meta = attr._mcp_tool_meta
                tool = MCPTool(
                    name=meta["name"],
                    description=meta["description"],
                    input_schema=meta["input_schema"],
                    handler=attr,
                    server_name=self.server_name,
                )
                self._tools[tool.name] = tool

    def list_tools(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self._tools.values()]

    def describe(self) -> Dict[str, Any]:
        return {
            "server":      self.server_name,
            "description": self.server_description,
            "tools":       self.list_tools(),
        }

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResponse:
        if tool_name not in self._tools:
            return MCPResponse.error(
                f"Tool '{tool_name}' not found on server '{self.server_name}'. "
                f"Available: {list(self._tools.keys())}"
            )
        try:
            result = self._tools[tool_name].handler(**arguments)
            if isinstance(result, MCPResponse):
                return result
            return MCPResponse.json_data(result)
        except TypeError as exc:

            return MCPResponse.error(
                f"Tool '{tool_name}' called with bad arguments: {exc}. "
                f"Expected schema: {self._tools[tool_name].input_schema}"
            )
        except Exception as exc:
            return MCPResponse.error(
                f"Tool '{tool_name}' raised: {type(exc).__name__}: {exc}"
            )


def register_tool(
    name: str,
    description: str,
    input_schema: Dict[str, Any],
) -> Callable:
    def decorator(fn: Callable) -> Callable:
        fn._mcp_tool_meta = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
        }
        return fn
    return decorator


class MCPDispatcher:

    def __init__(self):
        self._servers: Dict[str, BaseMCPServer] = {}

    def register_server(self, server: BaseMCPServer) -> None:
        import sys
        self._servers[server.server_name] = server

        print(
            f"[MCPDispatcher] Registered server '{server.server_name}' "
            f"with {len(server._tools)} tool(s): {list(server._tools.keys())}",
            file=sys.stderr,
        )

    def call(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any] | None = None,
    ) -> MCPResponse:
        if server_name not in self._servers:
            return MCPResponse.error(
                f"Server '{server_name}' not registered. "
                f"Available: {list(self._servers.keys())}"
            )
        return self._servers[server_name].call_tool(tool_name, arguments or {})

    def list_all_tools(self) -> List[Dict[str, Any]]:
        """Flat list of every tool across every server, with server tags."""
        out: List[Dict[str, Any]] = []
        for srv in self._servers.values():
            out.extend(srv.list_tools())
        return out

    def describe_all(self) -> List[Dict[str, Any]]:
        return [srv.describe() for srv in self._servers.values()]

    def describe_tool(
        self, server_name: str, tool_name: str,
    ) -> Optional[Dict[str, Any]]:
        srv = self._servers.get(server_name)
        if not srv:
            return None
        tool = srv._tools.get(tool_name)
        return tool.to_dict() if tool else None

    def get_server_names(self) -> List[str]:
        return list(self._servers.keys())

    def find_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        for srv in self._servers.values():
            if tool_name in srv._tools:
                return srv._tools[tool_name].to_dict()
        return None
