"""MCP integration contracts."""

from cayu.mcp._jsonrpc import (
    DEFAULT_MCP_CLIENT_NAME,
    DEFAULT_MCP_CLIENT_VERSION,
    DEFAULT_MCP_REQUEST_TIMEOUT_S,
    MCP_PROTOCOL_VERSION,
    McpProtocolError,
)
from cayu.mcp.base import (
    McpClient,
    McpInitializeResult,
    McpResourceDefinition,
    McpResourceResult,
    McpServerSpec,
    McpSession,
    McpToolDefinition,
    McpToolResult,
)
from cayu.mcp.http import (
    DEFAULT_HTTP_MCP_CONNECT_TIMEOUT_S,
    DEFAULT_HTTP_MCP_TIMEOUT_S,
    HttpMcpClient,
    HttpMcpSession,
)
from cayu.mcp.stdio import (
    DEFAULT_MCP_CANCELLATION_NOTIFICATION_TIMEOUT_S,
    DEFAULT_MCP_GRACEFUL_SHUTDOWN_TIMEOUT_S,
    DEFAULT_MCP_WRITE_TIMEOUT_S,
    StdioMcpClient,
    StdioMcpSession,
)
from cayu.mcp.tools import (
    McpToolAdapter,
    McpToolset,
    connect_mcp_toolset,
    mcp_cayu_tool_name,
    mcp_tool_manifest_hash,
    mcp_tool_manifest_identity,
    mcp_tool_manifest_server_hash,
    mcp_tool_manifest_tools,
)

__all__ = [
    "DEFAULT_HTTP_MCP_CONNECT_TIMEOUT_S",
    "DEFAULT_HTTP_MCP_TIMEOUT_S",
    "DEFAULT_MCP_CANCELLATION_NOTIFICATION_TIMEOUT_S",
    "DEFAULT_MCP_CLIENT_NAME",
    "DEFAULT_MCP_CLIENT_VERSION",
    "DEFAULT_MCP_GRACEFUL_SHUTDOWN_TIMEOUT_S",
    "DEFAULT_MCP_REQUEST_TIMEOUT_S",
    "DEFAULT_MCP_WRITE_TIMEOUT_S",
    "MCP_PROTOCOL_VERSION",
    "HttpMcpClient",
    "HttpMcpSession",
    "McpClient",
    "McpInitializeResult",
    "McpProtocolError",
    "McpResourceDefinition",
    "McpResourceResult",
    "McpServerSpec",
    "McpSession",
    "McpToolAdapter",
    "McpToolDefinition",
    "McpToolResult",
    "McpToolset",
    "StdioMcpClient",
    "StdioMcpSession",
    "connect_mcp_toolset",
    "mcp_cayu_tool_name",
    "mcp_tool_manifest_hash",
    "mcp_tool_manifest_identity",
    "mcp_tool_manifest_server_hash",
    "mcp_tool_manifest_tools",
]
