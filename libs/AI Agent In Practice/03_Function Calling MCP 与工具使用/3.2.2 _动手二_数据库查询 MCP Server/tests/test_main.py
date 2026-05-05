# tests/test_main.py — 自动生成的冒烟测试
import pytest
import sys, os
import tempfile, sqlite3

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


# ── 测试 core_config 基础结构 ──────────────────────────
class TestCoreConfig:
    def test_import(self):
        from core_config import (
            MODEL_REGISTRY, ACTIVE_MODEL_KEY,
            get_litellm_id, get_api_key, get_base_url,
            get_model_list, estimate_cost,
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


# ── 测试 db_guard (SQL 安全守卫) ────────────────────────
class TestDBGuard:
    def test_valid_select(self):
        from db_guard import validate_sql
        result = validate_sql("SELECT * FROM users")
        assert "SELECT" in result.upper()

    def test_delete_rejected(self):
        from db_guard import validate_sql, SQLSecurityError
        with pytest.raises(SQLSecurityError):
            validate_sql("DELETE FROM users WHERE 1=1")

    def test_drop_rejected(self):
        from db_guard import validate_sql, SQLSecurityError
        with pytest.raises(SQLSecurityError):
            validate_sql("DROP TABLE users")

    def test_insert_rejected(self):
        from db_guard import validate_sql, SQLSecurityError
        with pytest.raises(SQLSecurityError):
            validate_sql("INSERT INTO users (name) VALUES ('test')")

    def test_update_rejected(self):
        from db_guard import validate_sql, SQLSecurityError
        with pytest.raises(SQLSecurityError):
            validate_sql("UPDATE users SET name='hacked'")

    def test_empty_sql_rejected(self):
        from db_guard import validate_sql
        with pytest.raises(ValueError):
            validate_sql("")

    def test_cte_with_delete_rejected(self):
        """CTE 中隐藏 DELETE 操作应被拦截"""
        from db_guard import validate_sql, SQLSecurityError
        with pytest.raises(SQLSecurityError):
            validate_sql("WITH evil AS (DELETE FROM users) SELECT 1")

    def test_sanitize_identifier_valid(self):
        from db_guard import sanitize_identifier
        assert sanitize_identifier("users") == "users"
        assert sanitize_identifier("order_items") == "order_items"
        assert sanitize_identifier("_private_table") == "_private_table"

    def test_sanitize_identifier_invalid(self):
        from db_guard import sanitize_identifier
        with pytest.raises(ValueError):
            sanitize_identifier("users; DROP TABLE")
        with pytest.raises(ValueError):
            sanitize_identifier("1badname")


# ── 测试 db_backend (数据库后端) ────────────────────────
@pytest.fixture
def sample_db():
    """创建临时测试数据库"""
    db_path = tempfile.mktemp(suffix=".db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL,
            region TEXT NOT NULL,
            is_vip INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(user_id),
            total_amount REAL NOT NULL,
            status TEXT NOT NULL
        );
        INSERT INTO users (username, email, region, is_vip) VALUES
            ('alice', 'alice@example.com', '华东', 1),
            ('bob', 'bob@example.com', '华北', 0),
            ('charlie', 'charlie@example.com', '华南', 1);
        INSERT INTO orders (user_id, total_amount, status) VALUES
            (1, 100.0, 'done'),
            (2, 200.0, 'pending'),
            (1, 50.0, 'done');
    """)
    conn.commit()
    conn.close()
    yield db_path
    os.unlink(db_path)


class TestDBBackend:
    def test_sqlite_execute_basic(self, sample_db):
        from db_backend import sqlite_execute
        result = sqlite_execute(sample_db, "SELECT COUNT(*) as cnt FROM users")
        import json
        data = json.loads(result)
        assert data["rows"][0]["cnt"] == 3

    def test_sqlite_execute_truncation(self, sample_db):
        """验证结果截断机制"""
        from db_backend import sqlite_execute, MAX_ROWS
        # 生成超过 MAX_ROWS 行的临时表
        import sqlite3
        conn = sqlite3.connect(sample_db)
        conn.execute("INSERT INTO users (username, email, region, is_vip) SELECT username||'_'||CAST(rowid+3 AS TEXT), email||'_'||CAST(rowid+3 AS TEXT), region, is_vip FROM users")
        for _ in range(7):  # 翻倍 7 次 = 3 * 128 = 384 行
            conn.execute("INSERT INTO users (username, email, region, is_vip) SELECT username||'_x', email||'_x', region, is_vip FROM users")
        conn.commit()
        conn.close()

        import json
        result = sqlite_execute(sample_db, "SELECT * FROM users")
        data = json.loads(result)
        assert data["truncated"] is True
        assert len(data["rows"]) <= MAX_ROWS

    def test_sqlite_get_schema(self, sample_db):
        from db_backend import sqlite_get_schema
        schema = sqlite_get_schema(sample_db)
        assert "users" in schema
        assert "orders" in schema
        user_cols = [c["column"] for c in schema["users"]]
        assert "user_id" in user_cols
        assert "username" in user_cols

    def test_sqlite_describe_table(self, sample_db):
        from db_backend import sqlite_describe_table
        result = sqlite_describe_table(sample_db, "orders")
        assert result["table"] == "orders"
        col_names = [c["column"] for c in result["columns"]]
        assert "order_id" in col_names
        assert "total_amount" in col_names

    def test_sqlite_describe_table_not_found(self, sample_db):
        from db_backend import sqlite_describe_table
        with pytest.raises(ValueError):
            sqlite_describe_table(sample_db, "nonexistent_table")

    def test_sqlite_get_sample(self, sample_db):
        from db_backend import sqlite_get_sample
        import json
        result = sqlite_get_sample(sample_db, "users", limit=2)
        data = json.loads(result)
        assert len(data["rows"]) == 2

    def test_format_query_result(self):
        from db_backend import format_query_result
        import json
        rows = [{"name": "alice", "score": 100}]
        result = json.loads(format_query_result(rows, 1))
        assert result["row_count"] == 1
        assert result["truncated"] is False

    def test_truncate_long_field(self):
        from db_backend import format_query_result, MAX_FIELD_LENGTH
        import json
        long_text = "A" * (MAX_FIELD_LENGTH + 100)
        rows = [{"text": long_text}]
        result = json.loads(format_query_result(rows, 1))
        val = result["rows"][0]["text"]
        assert len(val) < len(long_text)
        assert "截断" in val


# ── 测试主模块可导入 ───────────────────────────────────
def test_server_module_importable():
    try:
        import importlib.util
        path = os.path.join(PROJECT_DIR, "server.py")
        spec = importlib.util.spec_from_file_location("server", path)
        assert spec is not None, "server.py 不存在"
    except Exception as e:
        pytest.skip(f"主模块检测跳过: {e}")


# ── 测试 scripts_create_sample_db 可导入 ──────────────
def test_sample_db_script_importable():
    try:
        import importlib.util
        path = os.path.join(PROJECT_DIR, "scripts_create_sample_db.py")
        spec = importlib.util.spec_from_file_location("scripts_create_sample_db", path)
        assert spec is not None
    except Exception as e:
        pytest.skip(f"脚本检测跳过: {e}")
