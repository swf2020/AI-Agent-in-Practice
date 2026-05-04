"""
prompt-debugger 核心包

模块说明：
  caller.py  - LLM 并发调用层
  history.py - 实验历史持久化层
"""
import os
import sys
from dotenv import load_dotenv

# 确保项目根目录在 sys.path 中，使 core_config.py 可被 core/ 下模块导入
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 包加载时立即读取 .env，确保后续所有模块都能拿到环境变量
load_dotenv()