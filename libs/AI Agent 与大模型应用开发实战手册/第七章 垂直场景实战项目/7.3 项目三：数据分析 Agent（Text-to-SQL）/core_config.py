from dotenv import load_dotenv
import os

load_dotenv()

# OpenAI 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Anthropic 配置（可选）
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# 数据库配置
DB_PATH = os.getenv("DB_PATH", "ecommerce.db")

# Schema 配置
LARGE_SCHEMA_THRESHOLD = 20
EMBED_MODEL = "text-embedding-3-small"

# SQL 执行配置
QUERY_TIMEOUT_SECONDS = 10
MAX_RETRIES = 3

# 图表输出目录
CHARTS_DIR = "charts"