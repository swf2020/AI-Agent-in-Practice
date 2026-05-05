# conftest.py — pytest hooks for this project
import os

# Set dummy API keys so project modules can be imported during test collection
os.environ.setdefault("TAVILY_API_KEY", "test_dummy_key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test_dummy_key")
os.environ.setdefault("OPENAI_API_KEY", "test_dummy_key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test_dummy_key")
os.environ.setdefault("DASHSCOPE_API_KEY", "test_dummy_key")
