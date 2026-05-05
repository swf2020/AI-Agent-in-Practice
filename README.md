# AI Agent 与大模型应用开发实战手册

> 🚀 **从入门到精通的 AI Agent 实战指南**

一本关于 AI Agent 与大模型应用开发的实战手册，涵盖从基础概念到生产级部署的完整知识体系。

---

## 🎯 这本书是什么？

**一句话概括**：面向工程师的 AI Agent 实战教程，通过 15+ 个动手项目带你从零构建可上线的 AI 产品。

**核心特点**：
- ✅ 15+ 可运行实战项目（代码可直接执行）
- ✅ 10 章 + 附录体系化内容（基础 → 进阶 → 生产级 → 面试）
- ✅ 覆盖主流框架（LangGraph、AutoGen、CrewAI、TradingAgents）
- ✅ 生产级落地经验（监控、成本控制、安全）

**适合人群**：
- 有 Python 基础，想快速做出 AI 产品的开发者
- 想系统学习 AI Agent 架构的工程师
- 准备面试 AI Agent 相关岗位的求职者

---

## 📚 核心内容

| 章节 | 内容 |
|------|------|
| 第 0 章 | 写给读者：学习路线图 |
| 第 1-2 章 | 大模型基础：LLM API 调用、微调、Prompt Engineering、RAG |
| 第 3-4 章 | Agent 核心：Function Calling、MCP、ReAct、Planning、记忆系统 |
| 第 5 章 | Multi-Agent：多 Agent 协作模式、AutoGen、CrewAI |
| 第 6 章 | 生产级落地：流式输出、成本控制、缓存、安全、可观测性 |
| 第 7 章 | 实战项目：AI 选股分析师、知识库问答、Text-to-SQL、工作流 Agent |
| 第 8 章 | 技术演进：大模型发展史、Agent 演进路线图、未来预测 |
| 第 9 章 | 面试题真题：原理/工程/设计/视野/开放题 |

---

## 🚀 快速开始

```bash
# 克隆项目
git clone https://github.com/swf2020/AI-Agent-in-Practice.git
cd AI-Agent-in-Practice

# 进入实战项目目录
cd libs/AI\ Agent\ In\ Practice/07_垂直场景实战项目/7.1\ 项目一：AI\ 选股分析师（基于\ TradingAgents）/

# 运行示例（需要配置 .env 中的 API Key）
python main.py 1

# 本地预览文档
pip install mkdocs mkdocs-material
mkdocs serve
```

---

## 📂 目录结构

```
AI-Agent-in-Practice/
├── docs/                    # 教程文档（CC BY-NC-SA 4.0）
│   └── AI Agent In Practice/
│       ├── 00_ ~ 10_/       # 第零章 ~ 附录（按自然顺序）
│       └── index.md         # 内容索引
├── libs/                    # 代码示例（MIT License）
│   └── AI Agent In Practice/
│       ├── 01_ ~ 07_/       # 各章节代码（按自然顺序）
│       └── 7.1 ~ 7.4/       # 实战项目完整代码
├── README.md                # 本文件
├── CONTRIBUTING.md          # 贡献指南
├── CHANGELOG.md            # 更新日志
└── LICENSE                 # 混合许可证
```

---

## 🛠️ 技术栈

| 分类 | 技术 |
|------|------|
| LLM API | OpenAI GPT-4o、Claude 3.5、Gemini、DeepSeek、通义千问 |
| Agent 框架 | LangGraph、AutoGen、CrewAI、TradingAgents |
| 向量数据库 | FAISS、Milvus、Chroma |
| RAG | LangChain、RAGAS 评估 |
| MCP | MCP SDK 文件系统/数据库/代码执行 |
| 生产工具 | Redis、ARQ、FastAPI、LiteLLM |

---

## 📊 项目统计

| 指标 | 数量 |
|------|------|
| 总章节数 | 10 章 + 附录 |
| 文档文件数 | 63+ 个 |
| 动手实验 | 15+ 个 |
| 实战项目 | 4 个完整项目 |

---

## 📝 贡献指南

欢迎贡献代码、文档修正或新增内容！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与。

---

## 📄 许可证

本项目采用混合许可证：
- `/libs/` - MIT License
- `/docs/` - CC BY-NC-SA 4.0
- 根目录 - Apache License 2.0

详情请阅读 [LICENSE](LICENSE)。
