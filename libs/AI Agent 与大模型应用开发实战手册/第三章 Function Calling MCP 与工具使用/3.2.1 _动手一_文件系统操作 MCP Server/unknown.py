# ── 主入口 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
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
    
    print(f"🚀 文件系统 MCP Server 启动", file=sys.stderr)
    print(f"   根目录：{ALLOWED_ROOT}", file=sys.stderr)
    print(f"   传输方式：{args.transport}", file=sys.stderr)
    
    if args.transport == "http":
        # HTTP 模式：用于本地调试，可以直接用 curl 或 MCP Inspector 测试
        mcp.run(transport="streamable-http", port=args.port)
    else:
        # stdio 模式：Claude Desktop 通过子进程 stdin/stdout 通信
        mcp.run(transport="stdio")