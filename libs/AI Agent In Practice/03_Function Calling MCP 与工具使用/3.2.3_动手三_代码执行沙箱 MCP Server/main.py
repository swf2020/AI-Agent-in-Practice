"""
代码执行沙箱 MCP Server — 主入口
运行：python main.py  启动 MCP Server
"""
from sandbox_server import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
