import sqlite3
import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


@dataclass
class TableSchema:
    table_name: str
    ddl: str
    columns: list[dict]
    sample_rows: list[dict]
    row_count: int
    description: str = ""
    embedding: list[float] = field(default_factory=list)


class SchemaManager:
    LARGE_SCHEMA_THRESHOLD = 20

    def __init__(self, db_path: str, embed_model: str = "text-embedding-3-small"):
        self.db_path = db_path
        self.embed_model = embed_model
        self.tables: dict[str, TableSchema] = {}
        self._load_schemas()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_schemas(self) -> None:
        conn = self._get_connection()
        cur = conn.cursor()

        tables = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()

        for (table_name,) in tables:
            ddl_row = cur.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,)
            ).fetchone()
            ddl = ddl_row[0] if ddl_row else ""

            col_rows = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
            columns = [
                {
                    "name": row["name"],
                    "type": row["type"],
                    "nullable": not row["notnull"],
                    "is_pk": bool(row["pk"]),
                }
                for row in col_rows
            ]

            row_count = cur.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

            sample = cur.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchall()
            col_names = [desc[0] for desc in cur.description]
            sample_rows = [dict(zip(col_names, row)) for row in sample]

            col_descriptions = []
            for col in columns:
                import re
                pattern = rf"{col['name']}\s+\S+[^,\n]*--\s*(.+)"
                match = re.search(pattern, ddl, re.IGNORECASE)
                comment = match.group(1).strip() if match else ""
                col_descriptions.append(
                    f"{col['name']}（{col['type']}）{'：' + comment if comment else ''}"
                )

            description = (
                f"表名：{table_name}，"
                f"行数约 {row_count}。"
                f"字段：{', '.join(col_descriptions)}"
            )

            self.tables[table_name] = TableSchema(
                table_name=table_name,
                ddl=ddl,
                columns=columns,
                sample_rows=sample_rows,
                row_count=row_count,
                description=description,
            )

        conn.close()
        print(f"✅ 加载 {len(self.tables)} 张表的 Schema")

    def build_embeddings(self) -> None:
        cache_path = f"{self.db_path}.schema_embeddings.json"
        schema_hash = hashlib.md5(
            "".join(t.description for t in self.tables.values()).encode()
        ).hexdigest()

        import os
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cache = json.load(f)
            if cache.get("hash") == schema_hash:
                for table_name, emb in cache["embeddings"].items():
                    if table_name in self.tables:
                        self.tables[table_name].embedding = emb
                print(f"✅ 从缓存加载 Embedding（{len(self.tables)} 张表）")
                return

        descriptions = [t.description for t in self.tables.values()]
        table_names = list(self.tables.keys())

        response = client.embeddings.create(
            model=self.embed_model,
            input=descriptions,
        )
        for i, item in enumerate(response.data):
            self.tables[table_names[i]].embedding = item.embedding

        with open(cache_path, "w") as f:
            json.dump({
                "hash": schema_hash,
                "embeddings": {n: self.tables[n].embedding for n in table_names}
            }, f)
        print(f"✅ 构建并缓存 {len(self.tables)} 张表的 Embedding")

    def retrieve_relevant_tables(
        self, query: str, top_k: int = 5
    ) -> list[TableSchema]:
        if len(self.tables) < self.LARGE_SCHEMA_THRESHOLD:
            return list(self.tables.values())

        if not next(iter(self.tables.values())).embedding:
            self.build_embeddings()

        query_emb = np.array(
            client.embeddings.create(model=self.embed_model, input=[query])
            .data[0].embedding
        )

        scores = []
        for table in self.tables.values():
            table_emb = np.array(table.embedding)
            cos_sim = np.dot(query_emb, table_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(table_emb) + 1e-9
            )
            scores.append((cos_sim, table))

        scores.sort(key=lambda x: x[0], reverse=True)
        selected = [t for _, t in scores[:top_k]]
        print(f"📋 检索到相关表：{[t.table_name for t in selected]}")
        return selected

    def format_schema_prompt(
        self, tables: list[TableSchema], include_samples: bool = True
    ) -> str:
        enc = tiktoken.get_encoding("cl100k_base")
        parts = []

        for table in tables:
            part = f"### 表：{table.table_name}（约 {table.row_count} 行）\n"
            part += f"```sql\n{table.ddl}\n```\n"

            if include_samples and table.sample_rows:
                part += "\n**样例数据（前3行）：**\n"
                for row in table.sample_rows[:3]:
                    row_str = ", ".join(
                        f"{k}={repr(v)}" for k, v in list(row.items())[:6]
                    )
                    part += f"  {row_str}\n"

            parts.append(part)

        full_prompt = "\n".join(parts)
        token_count = len(enc.encode(full_prompt))
        print(f"📊 Schema Prompt：{len(tables)} 张表，约 {token_count} Token")
        return full_prompt