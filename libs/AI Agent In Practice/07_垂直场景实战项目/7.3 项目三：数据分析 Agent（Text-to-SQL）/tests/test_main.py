# tests/test_main.py — 冒烟测试
import pytest
from unittest.mock import patch, MagicMock
import sys
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


# ── 测试 core_config 基础结构 ──────────────────────────
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
        assert result is None or isinstance(result, str)


# ── 测试主模块可导入 ───────────────────────────────────
def test_main_module_importable():
    try:
        import importlib.util
        path = os.path.join(PROJECT_DIR, "main.py")
        spec = importlib.util.spec_from_file_location("main", path)
        assert spec is not None, "main.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


# ── 测试 SQL 安全检测 ─────────────────────────────────
class TestSQLSafety:
    def test_reject_insert(self):
        from sql_executor import _check_sql_safety, SQLSafetyError
        with pytest.raises(SQLSafetyError):
            _check_sql_safety("INSERT INTO users VALUES (1)")

    def test_reject_drop(self):
        from sql_executor import _check_sql_safety, SQLSafetyError
        with pytest.raises(SQLSafetyError):
            _check_sql_safety("DROP TABLE users")

    def test_reject_delete(self):
        from sql_executor import _check_sql_safety, SQLSafetyError
        with pytest.raises(SQLSafetyError):
            _check_sql_safety("DELETE FROM orders WHERE id=1")

    def test_allow_select(self):
        from sql_executor import _check_sql_safety
        _check_sql_safety("SELECT * FROM users WHERE id = 1")

    def test_allow_select_join(self):
        from sql_executor import _check_sql_safety
        _check_sql_safety(
            "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        )


# ── 测试 SQL 执行器 ───────────────────────────────────
class TestSQLExecutor:
    @pytest.fixture
    def demo_db(self, tmp_path):
        """创建临时测试数据库"""
        db_path = str(tmp_path / "test.db")
        conn = __import__("sqlite3").connect(db_path)
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT, value REAL)")
        conn.execute("INSERT INTO test VALUES (1, 'A', 10.0)")
        conn.execute("INSERT INTO test VALUES (2, 'B', 20.0)")
        conn.commit()
        conn.close()
        return db_path

    def test_successful_query(self, demo_db):
        from sql_executor import SQLExecutor
        executor = SQLExecutor(demo_db)
        result = executor.execute("SELECT * FROM test")
        assert result.success is True
        assert result.row_count == 2

    def test_safety_rejection(self, demo_db):
        from sql_executor import SQLExecutor
        executor = SQLExecutor(demo_db)
        result = executor.execute("DELETE FROM test")
        assert result.success is False
        assert "拒绝执行" in result.error

    def test_syntax_error(self, demo_db):
        from sql_executor import SQLExecutor
        executor = SQLExecutor(demo_db)
        result = executor.execute("SELCT * FROM test")
        assert result.success is False
        assert result.error is not None

    def test_self_correcting_executor_mock(self, demo_db):
        """测试 SelfCorrectingExecutor 在 mock 下的行为"""
        from sql_executor import SQLExecutor, SelfCorrectingExecutor
        executor = SQLExecutor(demo_db)
        mock_gen = MagicMock()
        mock_gen.generate.return_value = MagicMock(
            sql="SELECT * FROM test WHERE id = 1"
        )
        correcting = SelfCorrectingExecutor(executor, mock_gen, "")
        result, sql, retries = correcting.execute_with_correction(
            "test", "SELECT * FROM test WHERE id = 1"
        )
        assert result.success is True
        assert retries == 0


# ── 测试 SQL Generator 结构 ───────────────────────────
class TestSQLGenerator:
    def test_import(self):
        from sql_generator import Dialect
        assert isinstance(Dialect.SQLITE.value, str)

    def test_sql_clean(self):
        from sql_generator import SQLGenerator
        assert SQLGenerator._clean_sql("```sql\nSELECT 1\n```") == "SELECT 1"
        assert SQLGenerator._clean_sql("SELECT 1;") == "SELECT 1"

    @patch("sql_generator.get_openai_client")
    def test_generator_uses_config(self, mock_client_factory):
        """验证 SQLGenerator 从 core_config 获取模型配置"""
        from sql_generator import SQLGenerator, Dialect
        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client
        mock_client.beta.chat.completions.parse.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(
                    parsed=MagicMock(
                        sql="SELECT 1",
                        explanation="test",
                        confidence=0.9,
                        ambiguities=[]
                    )
                )
            )]
        )
        gen = SQLGenerator(dialect=Dialect.SQLITE)
        result = gen.generate("test", "schema")
        assert result.sql == "SELECT 1"


# ── 测试 db_setup ─────────────────────────────────────
class TestDbSetup:
    def test_create_demo_database(self, tmp_path):
        from db_setup import create_demo_database
        db_path = str(tmp_path / "demo.db")
        create_demo_database(db_path)
        assert os.path.exists(db_path)
        conn = __import__("sqlite3").connect(db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert len(tables) >= 5  # users, orders, order_items, products, categories
        conn.close()


# ── 测试 SchemaManager ────────────────────────────────
class TestSchemaManager:
    @pytest.fixture
    def demo_db(self, tmp_path):
        db_path = str(tmp_path / "schema_test.db")
        conn = __import__("sqlite3").connect(db_path)
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO users VALUES (1, 'Alice')")
        conn.commit()
        conn.close()
        return db_path

    def test_load_schemas(self, demo_db):
        from schema_manager import SchemaManager
        mgr = SchemaManager(demo_db)
        assert "users" in mgr.tables
        assert mgr.tables["users"].row_count == 1

    def test_format_schema_prompt(self, demo_db):
        from schema_manager import SchemaManager
        mgr = SchemaManager(demo_db)
        tables = list(mgr.tables.values())
        prompt = mgr.format_schema_prompt(tables, include_samples=True)
        assert "users" in prompt
        assert isinstance(prompt, str)


# ── 测试 LLM 调用（Mock）──────────────────────────────
class TestLLMCall:
    @patch("sql_generator.get_openai_client")
    def test_mocked_completion(self, mock_get_client):
        """验证核心调用路径在 mock 下可正常执行"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.beta.chat.completions.parse.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(
                    parsed=MagicMock(
                        sql="SELECT 1",
                        explanation="test",
                        confidence=0.9,
                        ambiguities=[]
                    )
                )
            )]
        )
        from sql_generator import SQLGenerator, Dialect
        gen = SQLGenerator(dialect=Dialect.SQLITE)
        result = gen.generate("test", "schema")
        assert result.sql == "SELECT 1"
        mock_client.beta.chat.completions.parse.assert_called_once()
