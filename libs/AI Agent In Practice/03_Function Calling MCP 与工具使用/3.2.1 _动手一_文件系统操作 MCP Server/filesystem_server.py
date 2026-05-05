import os
import stat
import fnmatch
from datetime import datetime
from pathlib import Path

from fastmcp import FastMCP

# 文件大小上限：避免 Claude 把一个 2GB 的日志文件整个读进上下文
MAX_FILE_SIZE_BYTES = int(os.environ.get("MCP_MAX_FILE_SIZE", 1024 * 1024))  # 默认 1MB

def _get_allowed_root() -> Path:
    """
    动态获取允许的根目录。
    优先从环境变量读取，让部署时灵活配置；默认限制在当前工作目录。
    动态获取确保环境变量在运行时生效，而不是在模块加载时就固定。
    """
    return Path(
        os.environ.get("MCP_ALLOWED_ROOT", Path.cwd())
    ).expanduser().resolve()

def _safe_path(raw: str) -> Path:
    """
    将用户传入的路径解析为绝对路径，并校验是否在白名单根目录内。

    使用 Path.resolve() 而非简单的字符串前缀匹配，是为了防止路径穿越攻击：
    ../../../etc/passwd 经过 resolve() 后会暴露真实绝对路径，从而被拦截。
    """
    allowed_root = _get_allowed_root()
    resolved = Path(raw).expanduser().resolve()
    # is_relative_to 是 Python 3.9+ 的方法，确保路径在白名单内
    if not resolved.is_relative_to(allowed_root):
        raise PermissionError(
            f"路径 '{raw}' 超出允许的根目录 '{allowed_root}'。"
            f"请设置环境变量 MCP_ALLOWED_ROOT 扩大访问范围。"
        )
    return resolved


# ── Server 初始化 ──────────────────────────────────────────────────────────────
def _get_server_instructions() -> str:
    """动态生成 Server 级别的系统提示。"""
    return (
        f"你可以操作本机文件系统，根目录限定为：{_get_allowed_root()}。"
        f"单文件最大读取 {MAX_FILE_SIZE_BYTES // 1024}KB。"
        "写操作会直接修改磁盘文件，请在执行前向用户确认。"
    )

mcp = FastMCP(
    name="filesystem-server",
    # instructions 是 Server 级别的系统提示，会注入给调用方的 LLM。
    # 清晰说明能力边界，有助于 Claude 做出更准确的工具选择。
    instructions=_get_server_instructions(),
)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

@mcp.tool()
def read_file(path: str) -> str:
    """读取指定文件的内容。"""
    p = _safe_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
    file_size = p.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"文件大小 ({file_size // 1024}KB) 超过限制 ({MAX_FILE_SIZE_BYTES // 1024}KB)"
        )
    return p.read_text(encoding="utf-8")


@mcp.tool()
def write_file(path: str, content: str) -> dict:
    """创建或覆盖文件。"""
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return {"status": "success", "path": str(p), "bytes_written": len(content.encode("utf-8"))}


def _build_tree(dir_path: Path, max_depth: int, current_depth: int = 0) -> list:
    """递归构建目录树结构。"""
    if current_depth >= max_depth:
        return []
    children = []
    try:
        entries = sorted(dir_path.iterdir(), key=lambda x: (not x.is_file(), x.name))
    except PermissionError:
        return ["[Permission Denied]"]
    for entry in entries:
        if entry.is_file():
            children.append({"name": entry.name, "type": "file", "size": entry.stat().st_size})
        elif entry.is_dir():
            children.append({
                "name": entry.name,
                "type": "directory",
                "children": _build_tree(entry, max_depth, current_depth + 1),
            })
    return children


@mcp.tool()
def list_directory(path: str = ".", max_depth: int = 1) -> dict:
    """列出目录内容，支持递归深度控制。"""
    p = _safe_path(path)
    if not p.is_dir():
        raise NotADirectoryError(f"不是目录: {path}")
    children = _build_tree(p, max_depth)
    return {
        "root": str(p),
        "children": children,
        "summary": f"目录 {p} 共 {len(children)} 个条目",
    }


@mcp.tool()
def search_files(
    query: str,
    directory: str = ".",
    file_pattern: str = "*",
    max_results: int = 20,
) -> dict:
    """在指定目录中搜索文件名或内容包含 query 的文件。"""
    p = _safe_path(directory)
    if not p.is_dir():
        raise NotADirectoryError(f"不是目录: {directory}")
    matched = []
    total_matches = 0
    for root, dirs, files in os.walk(p):
        root_path = Path(root)
        if not root_path.is_relative_to(_get_allowed_root()):
            dirs.clear()
            continue
        for fname in files:
            if not fnmatch.fnmatch(fname, file_pattern):
                continue
            if query.lower() in fname.lower():
                matched.append(str(root_path / fname))
                total_matches += 1
            if len(matched) >= max_results:
                break
        if len(matched) >= max_results:
            break
    return {
        "matched_files": len(matched),
        "total_matches": total_matches,
        "files": matched,
    }


@mcp.tool()
def get_file_info(path: str) -> dict:
    """获取文件的元信息（大小、修改时间、权限等）。"""
    p = _safe_path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    st = p.stat()
    return {
        "path": str(p),
        "name": p.name,
        "size_bytes": st.st_size,
        "size_human": f"{st.st_size / 1024:.1f}KB" if st.st_size < 1024 * 1024 else f"{st.st_size / (1024 * 1024):.1f}MB",
        "modified": datetime.fromtimestamp(st.st_mtime).isoformat(),
        "created": datetime.fromtimestamp(st.st_ctime).isoformat(),
        "is_file": p.is_file(),
        "is_dir": p.is_dir(),
        "permissions": stat.filemode(st.st_mode),
    }


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

    print("文件系统 MCP Server 启动", file=sys.stderr)
    print(f"   根目录：{_get_allowed_root()}", file=sys.stderr)
    print(f"   传输方式：{args.transport}", file=sys.stderr)

    if args.transport == "http":
        # HTTP 模式：用于本地调试，可以直接用 curl 或 MCP Inspector 测试
        mcp.run(transport="streamable-http", port=args.port)
    else:
        # stdio 模式：Claude Desktop 通过子进程 stdin/stdout 通信
        mcp.run(transport="stdio")
