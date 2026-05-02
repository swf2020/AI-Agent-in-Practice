@mcp.tool()
def get_execution_history(last_n: int = 10) -> dict[str, Any]:
    """
    获取本次会话的代码执行历史记录。
    
    Args:
        last_n: 返回最近 N 条记录（默认 10，最大 50）
    
    Returns:
        {
          "total": int,          # 本次会话总执行次数
          "session_start": str,  # 会话开始时间
          "records": list,       # 执行记录列表（倒序，最新在前）
          "installed_packages": list,
        }
    
    用途：Claude 可用此工具回顾之前执行了什么、哪些成功/失败，
    避免重复执行相同代码或重蹈已失败的覆辙。
    """
    last_n = max(1, min(last_n, 50))
    recent = _session.history[-last_n:][::-1]  # 倒序，最新在前
    
    return {
        "total": len(_session.history),
        "session_start": _session.created_at,
        "installed_packages": _session.installed_packages,
        "records": [
            {
                "timestamp": r.timestamp,
                "success": r.success,
                "duration_ms": r.duration_ms,
                "code_preview": r.code[:100] + ("..." if len(r.code) > 100 else ""),
                "stdout_preview": r.stdout[:200] + ("..." if len(r.stdout) > 200 else ""),
                "error": r.error,
            }
            for r in recent
        ],
    }


@mcp.tool()
def reset_session() -> dict[str, Any]:
    """
    清空执行历史和已安装包记录，重置会话状态。
    
    注意：此操作不会卸载已安装的 pip 包（卸载有副作用），
    仅清空会话记录。如需真正隔离，请重启 MCP Server。
    """
    global _session
    old_count = len(_session.history)
    _session = SessionState()   # 重建 SessionState，旧引用自动 GC
    
    return {
        "success": True,
        "message": f"会话已重置。清除了 {old_count} 条执行记录。",
        "note": "pip 包不会被卸载，已安装的包在本进程内仍可使用。",
    }


# ── 服务入口 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🔒 代码执行沙箱 MCP Server 启动", flush=True)
    print(f"   Python: {sys.executable}", flush=True)
    print(f"   安全模式: 静态分析 + 子进程隔离 + 环境变量隔离", flush=True)
    mcp.run(transport="stdio")