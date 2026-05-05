import sqlite3
import random
from datetime import datetime, timedelta


def create_demo_database(db_path: str = "ecommerce.db") -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS categories (
        category_id   INTEGER PRIMARY KEY,
        name          TEXT NOT NULL,
        parent_id     INTEGER,
        created_at    TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS products (
        product_id    INTEGER PRIMARY KEY,
        name          TEXT NOT NULL,
        category_id   INTEGER REFERENCES categories(category_id),
        price         REAL NOT NULL,
        cost          REAL NOT NULL,
        stock         INTEGER DEFAULT 0,
        created_at    TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS users (
        user_id       INTEGER PRIMARY KEY,
        username      TEXT NOT NULL UNIQUE,
        email         TEXT NOT NULL UNIQUE,
        city          TEXT,
        register_date TEXT DEFAULT (datetime('now')),
        vip_level     INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS orders (
        order_id      INTEGER PRIMARY KEY,
        user_id       INTEGER REFERENCES users(user_id),
        status        TEXT NOT NULL,
        total_amount  REAL NOT NULL,
        created_at    TEXT NOT NULL,
        paid_at       TEXT
    );

    CREATE TABLE IF NOT EXISTS order_items (
        item_id       INTEGER PRIMARY KEY,
        order_id      INTEGER REFERENCES orders(order_id),
        product_id    INTEGER REFERENCES products(product_id),
        quantity      INTEGER NOT NULL,
        unit_price    REAL NOT NULL,
        discount      REAL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS user_events (
        event_id      INTEGER PRIMARY KEY,
        user_id       INTEGER REFERENCES users(user_id),
        product_id    INTEGER REFERENCES products(product_id),
        event_type    TEXT NOT NULL,
        event_time    TEXT NOT NULL
    );
    """)

    categories_data = [
        (1, "电子产品", None), (2, "服装", None), (3, "食品", None),
        (4, "手机", 1), (5, "笔记本", 1), (6, "耳机", 1),
        (7, "男装", 2), (8, "女装", 2),
    ]
    cur.executemany(
        "INSERT OR IGNORE INTO categories VALUES (?,?,?,datetime('now'))",
        categories_data,
    )

    random.seed(42)
    products_data = []
    for i in range(1, 51):
        cat_id = random.choice([4, 5, 6, 7, 8])
        price = round(random.uniform(50, 8000), 2)
        products_data.append((
            i, f"商品_{i:03d}", cat_id,
            price, round(price * 0.6, 2),
            random.randint(0, 500),
        ))
    cur.executemany(
        "INSERT OR IGNORE INTO products(product_id,name,category_id,price,cost,stock) VALUES (?,?,?,?,?,?)",
        products_data,
    )

    cities = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉"]
    users_data = [(i, f"user_{i}", f"user_{i}@example.com",
                   random.choice(cities), random.randint(0, 3))
                  for i in range(1, 201)]
    cur.executemany(
        "INSERT OR IGNORE INTO users(user_id,username,email,city,vip_level) VALUES (?,?,?,?,?)",
        users_data,
    )

    base_date = datetime(2024, 1, 1)
    order_id = 1
    item_id = 1
    statuses = ["completed", "completed", "completed", "cancelled", "shipped"]
    for _ in range(800):
        user_id = random.randint(1, 200)
        created_at = base_date + timedelta(days=random.randint(0, 364),
                                           hours=random.randint(0, 23))
        status = random.choice(statuses)
        items_count = random.randint(1, 4)
        total = 0.0

        cur.execute(
            "INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?,?)",
            (order_id, user_id, status, 0,
             created_at.isoformat(), created_at.isoformat() if status != "pending" else None),
        )

        for _ in range(items_count):
            prod = random.choice(products_data)
            qty = random.randint(1, 3)
            price = prod[3]
            disc = round(price * random.uniform(0, 0.15), 2)
            total += (price - disc) * qty
            cur.execute(
                "INSERT OR IGNORE INTO order_items VALUES (?,?,?,?,?,?)",
                (item_id, order_id, prod[0], qty, price, disc),
            )
            item_id += 1

        cur.execute("UPDATE orders SET total_amount=? WHERE order_id=?",
                    (round(total, 2), order_id))
        order_id += 1

    conn.commit()
    conn.close()
    print(f"✅ 演示数据库已创建：{db_path}（{order_id-1} 笔订单）")


if __name__ == "__main__":
    create_demo_database()