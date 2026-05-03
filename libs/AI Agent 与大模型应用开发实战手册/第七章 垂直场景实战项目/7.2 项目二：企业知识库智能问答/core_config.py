from dotenv import load_dotenv
import os

load_dotenv()

# Qdrant 配置
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "enterprise_kb")

# OpenAI 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Embedding 模型配置
EMBED_MODEL = "BAAI/bge-m3"
VECTOR_DIM = 1024

# 检索配置
TOP_K_PER_SOURCE = 20
FINAL_TOP_N = 5
RRF_K = 60
CONFIDENCE_THRESHOLD = 0.0

# 分块配置
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64