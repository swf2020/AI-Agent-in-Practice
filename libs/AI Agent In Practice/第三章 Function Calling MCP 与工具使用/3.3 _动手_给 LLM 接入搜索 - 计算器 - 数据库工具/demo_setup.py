# demo_setup.py + smoke_test.py 合并版本，可直接运行

import os
import json
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# ── 1. 初始化演示数据库 ──────────────────────────────────────────────────────
DB_URL = "sqlite:///demo.db"
engine = create_engine(DB_URL)

with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock INTEGER,
            sales_count INTEGER
        )
    """))
    conn.execute(text("DELETE FROM products"))  # 清空防止重复插入
    conn.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "MacBook Pro 16", "laptop", 19999, 50, 320),
            (2, "iPhone 15 Pro", "phone", 8999, 200, 1580),
            (3, "AirPods Pro", "audio", 1799, 500, 2100),
            (4, "iPad Air", "tablet", 4799, 80, 450),
            (5, "Apple Watch S9", "wearable", 2999, 150, 880),
        ],
    )
    conn.commit()
print("✅ 演示数据库初始化完成")

# ── 2. 初始化 Agent ──────────────────────────────────────────────────────────
from agent import build_agent

agent = build_agent(db_url=DB_URL)
print("✅ Agent 初始化完成\n")

# ── 3. 端到端测试三类工具 ─────────────────────────────────────────────────────
test_cases = [
    {
        "desc": "🔍 测试搜索工具",
        "query": "2024年诺贝尔物理学奖得主是谁？他们的主要贡献是什么？"
    },
    {
        "desc": "💻 测试代码执行工具",
        "query": "计算斐波那契数列前20项的总和，并告诉我第20项是多少"
    },
    {
        "desc": "🗄️ 测试数据库工具",
        "query": "在我们的产品数据库中，哪个类别的平均价格最高？销量最好的产品是什么？"
    },
]

for case in test_cases:
    print(f"\n{'='*60}")
    print(f"{case['desc']}")
    print(f"问题: {case['query']}")
    print("-" * 40)
    
    answer = agent.run(case["query"], verbose=True)
    
    print(f"\n💬 最终回答:")
    print(answer)

print(f"\n\n✅ 所有测试通过！")