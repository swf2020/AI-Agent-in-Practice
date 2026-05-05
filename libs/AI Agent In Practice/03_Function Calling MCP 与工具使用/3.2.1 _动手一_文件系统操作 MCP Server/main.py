#!/usr/bin/env python3
"""主入口：文件系统 MCP Server"""

import sys

# 确保 .env 文件中的 API Key 被加载
from dotenv import load_dotenv
load_dotenv()

from filesystem_server import mcp, ALLOWED_ROOT

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="文件系统 MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="传输方式：stdio（Claude Desktop）或 http（调试模式，默认端口 8000）",
    )
    parser.add_argument("--port", type=int, default=8000, help="HTTP 模式端口号")
    args = parser.parse_args()

    print("文件系统 MCP Server 启动", file=sys.stderr)
    print(f"   根目录：{ALLOWED_ROOT}", file=sys.stderr)
    print(f"   传输方式：{args.transport}", file=sys.stderr)

    if args.transport == "http":
        mcp.run(transport="streamable-http", port=args.port)
    else:
        mcp.run(transport="stdio")
