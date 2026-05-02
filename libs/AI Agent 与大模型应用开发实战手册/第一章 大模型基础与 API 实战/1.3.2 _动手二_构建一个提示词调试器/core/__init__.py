"""
prompt-debugger 核心包

模块说明：
  caller.py  - LLM 并发调用层
  history.py - 实验历史持久化层
"""
from dotenv import load_dotenv

# 包加载时立即读取 .env，确保后续所有模块都能拿到环境变量
load_dotenv()