"""
生成示例电商数据库，包含：用户、商品、订单、订单明细 四张表
字段设计参考真实业务场景，包含外键关系和字段注释
"""
import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path("data/sample.db")
DB_PATH.parent.mkdir(exist_ok=True)

def create_database():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 启用外键约束（SQLite 默认关闭）
    cur.execute("PRAGMA foreign_keys = ON")

    # ── 建表 ──────────────────────────────────────────────
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL UNIQUE,
            email       TEXT    NOT NULL UNIQUE,
            region      TEXT    NOT NULL,               -- 大区：华东/华北/华南/海外
            created_at  TEXT    NOT NULL,
            is_vip      INTEGER NOT NULL DEFAULT 0      -- 0普通 1VIP
        );

        CREATE TABLE IF NOT EXISTS products (
            product_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT    NOT NULL,
            category     TEXT    NOT NULL,              -- 数码/服装/食品/家居
            price        REAL    NOT NULL,
            stock        INTEGER NOT NULL DEFAULT 0,
            supplier_id  INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS orders (
            order_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL REFERENCES users(user_id),
            status       TEXT    NOT NULL,              -- pending/paid/shipped/done/cancelled
            total_amount REAL    NOT NULL,
            created_at   TEXT    NOT NULL,
            shipped_at   TEXT
        );

        CREATE TABLE IF NOT EXISTS order_items (
            item_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id   INTEGER NOT NULL REFERENCES orders(order_id),
            product_id INTEGER NOT NULL REFERENCES products(product_id),
            quantity   INTEGER NOT NULL,
            unit_price REAL    NOT NULL                 -- 下单时快照价格，避免价格变更影响历史记录
        );
    """)

    # ── 写入种子数据 ──────────────────────────────────────
    regions = ["华东", "华北", "华南", "海外"]
    categories = ["数码", "服装", "食品", "家居"]
    statuses = ["pending", "paid", "shipped", "done", "cancelled"]

    # 50 名用户
    users = [
        (f"user_{i:03d}", f"user{i}@example.com",
         random.choice(regions),
         (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
         random.randint(0, 1))
        for i in range(1, 51)
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO users (username,email,region,created_at,is_vip) VALUES (?,?,?,?,?)",
        users
    )

    # 100 件商品
    products = [
        (f"商品_{i:03d}", random.choice(categories),
         round(random.uniform(9.9, 9999.0), 2),
         random.randint(0, 500),
         random.randint(1, 10))
        for i in range(1, 101)
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO products (name,category,price,stock,supplier_id) VALUES (?,?,?,?,?)",
        products
    )

    # 200 笔订单 + 明细
    for i in range(1, 201):
        user_id = random.randint(1, 50)
        status = random.choice(statuses)
        created_at = (datetime.now() - timedelta(days=random.randint(1, 180))).isoformat()
        shipped_at = (datetime.now() - timedelta(days=random.randint(0, 10))).isoformat() \
            if status in ("shipped", "done") else None

        # 每笔订单 1-5 件商品
        items = []
        total = 0.0
        for _ in range(random.randint(1, 5)):
            product_id = random.randint(1, 100)
            qty = random.randint(1, 3)
            price = round(random.uniform(9.9, 999.0), 2)
            items.append((product_id, qty, price))
            total += qty * price

        cur.execute(
            "INSERT INTO orders (user_id,status,total_amount,created_at,shipped_at) VALUES (?,?,?,?,?)",
            (user_id, status, round(total, 2), created_at, shipped_at)
        )
        order_id = cur.lastrowid
        cur.executemany(
            "INSERT INTO order_items (order_id,product_id,quantity,unit_price) VALUES (?,?,?,?)",
            [(order_id, *item) for item in items]
        )

    conn.commit()
    conn.close()
    print(f"✅ 数据库已生成：{DB_PATH.resolve()}")
    print("   表：users(50行) / products(100行) / orders(200行) / order_items(~600行)")

if __name__ == "__main__":
    create_database()