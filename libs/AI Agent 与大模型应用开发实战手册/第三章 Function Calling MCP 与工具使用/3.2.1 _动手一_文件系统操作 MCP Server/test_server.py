# 直接调用 Server 函数进行单元测试，不走 MCP 协议
# 适合 CI 环境或 Colab 调试

import sys, os
os.environ["MCP_ALLOWED_ROOT"] = os.path.expanduser("~/Documents")

# 把 Server 文件当模块导入
sys.path.insert(0, ".")
from filesystem_server import read_file, write_file, list_directory, search_files, get_file_info

# ── 测试 1：列目录 ────────────────────────────────────────────────────────────
print("=== list_directory ===")
result = list_directory(".", max_depth=1)
print(f"根目录：{result.get('root')}")
print(f"摘要：{result.get('summary', result.get('children', []))[:1]}")

# ── 测试 2：写文件 ────────────────────────────────────────────────────────────
print("\n=== write_file ===")
r = write_file("test_output.md", "# 测试文件\n\n由 MCP Server 创建。\n")
print(r)

# ── 测试 3：读文件 ────────────────────────────────────────────────────────────
print("\n=== read_file ===")
content = read_file("test_output.md")
print(content[:200])

# ── 测试 4：文件元信息 ────────────────────────────────────────────────────────
print("\n=== get_file_info ===")
info = get_file_info("test_output.md")
print(info)

# ── 测试 5：搜索文件 ──────────────────────────────────────────────────────────
print("\n=== search_files ===")
r = search_files("MCP", directory=".", file_pattern="*.md", max_results=5)
print(f"匹配文件数：{r['matched_files']}，总命中：{r['total_matches']}")

# ── 清理 ─────────────────────────────────────────────────────────────────────
import os
if os.path.exists("test_output.md"):
    os.remove("test_output.md")
    print("\n✅ 测试完成，临时文件已清理")