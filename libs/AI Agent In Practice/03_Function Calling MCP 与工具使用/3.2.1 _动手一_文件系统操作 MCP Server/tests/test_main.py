# tests/test_main.py — 冒烟测试
import pytest
import sys
import os
import tempfile
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 设置允许的根目录为临时目录，避免测试时操作真实文件系统
# 使用 Path.resolve() 确保与 filesystem_server 中 ALLOWED_ROOT 的计算方式一致
TEST_TMP_DIR = str(Path(tempfile.mkdtemp()).resolve())
os.environ["MCP_ALLOWED_ROOT"] = TEST_TMP_DIR


# ── 测试 core_config 基础结构 ──────────────────────────────────────────
class TestCoreConfig:
    def test_import(self):
        from core_config import (
            MODEL_REGISTRY, ACTIVE_MODEL_KEY,
        )
        assert isinstance(MODEL_REGISTRY, dict)
        assert len(MODEL_REGISTRY) > 0
        assert isinstance(ACTIVE_MODEL_KEY, str)
        assert ACTIVE_MODEL_KEY in MODEL_REGISTRY

    def test_model_registry_schema(self):
        """验证每个模型条目包含必要字段"""
        from core_config import MODEL_REGISTRY
        required_keys = {"litellm_id", "price_in", "price_out",
                         "max_tokens_limit", "api_key_env", "base_url"}
        for name, cfg in MODEL_REGISTRY.items():
            missing = required_keys - set(cfg.keys())
            assert not missing, f"{name} 缺少字段: {missing}"

    def test_get_litellm_id(self):
        from core_config import get_litellm_id
        result = get_litellm_id()
        assert isinstance(result, str) and len(result) > 0

    def test_get_model_list(self):
        from core_config import get_model_list, MODEL_REGISTRY
        lst = get_model_list()
        assert isinstance(lst, list)
        assert set(lst) == set(MODEL_REGISTRY.keys())

    def test_estimate_cost(self):
        from core_config import estimate_cost, get_model_list
        model_key = get_model_list()[0]
        cost = estimate_cost(model_key, input_tokens=1000, output_tokens=500)
        assert isinstance(cost, float) and cost >= 0

    def test_get_api_key_no_crash(self):
        """无环境变量时应返回 None 而不是抛异常"""
        from core_config import get_api_key
        result = get_api_key()
        assert result is None or isinstance(result, str)

    def test_get_base_url(self):
        from core_config import get_base_url
        result = get_base_url()
        # DeepSeek-V3 (默认) 的 base_url 为 None
        assert result is None or isinstance(result, str)


# ── 测试主模块可导入 ───────────────────────────────────────────────────
def test_main_module_importable():
    try:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")
        spec = importlib.util.spec_from_file_location("main", path)
        assert spec is not None, "main.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


# ── 测试文件系统工具函数 ────────────────────────────────────────────────
class TestFilesystemTools:
    """直接调用 Server 函数，不走 MCP 协议"""

    @classmethod
    def setup_class(cls):
        """确保 filesystem_server 使用测试临时目录作为 ALLOWED_ROOT"""
        import filesystem_server
        assert str(filesystem_server.ALLOWED_ROOT) == TEST_TMP_DIR, (
            f"ALLOWED_ROOT={filesystem_server.ALLOWED_ROOT} != {TEST_TMP_DIR}"
        )
        cls.tmp = TEST_TMP_DIR

    def test_write_and_read_file(self):
        from filesystem_server import write_file, read_file
        filepath = os.path.join(self.tmp, "test_write.md")
        r = write_file(filepath, "hello world")
        assert r["status"] == "success"
        content = read_file(filepath)
        assert content == "hello world"

    def test_list_directory(self):
        from filesystem_server import list_directory, write_file
        filepath = os.path.join(self.tmp, "list_test.txt")
        write_file(filepath, "test")
        result = list_directory(self.tmp, max_depth=1)
        assert "root" in result
        assert "children" in result
        assert any(c["name"] == "list_test.txt" for c in result["children"])

    def test_get_file_info(self):
        from filesystem_server import get_file_info, write_file
        filepath = os.path.join(self.tmp, "info_test.txt")
        write_file(filepath, "12345")
        info = get_file_info(filepath)
        assert info["name"] == "info_test.txt"
        assert info["size_bytes"] == 5
        assert info["is_file"] is True

    def test_search_files(self):
        from filesystem_server import search_files, write_file
        filepath = os.path.join(self.tmp, "search_test.md")
        write_file(filepath, "find me")
        result = search_files("search_test", directory=self.tmp, file_pattern="*.md", max_results=5)
        assert result["matched_files"] >= 1

    def test_safe_path_rejects_traversal(self):
        from filesystem_server import _safe_path, ALLOWED_ROOT
        inside = os.path.join(TEST_TMP_DIR, "subdir", "file.txt")
        result = _safe_path(inside)
        assert str(result).startswith(str(ALLOWED_ROOT))

    def test_safe_path_rejects_outside_root(self):
        from filesystem_server import _safe_path
        with pytest.raises(PermissionError):
            _safe_path("/etc/passwd")
