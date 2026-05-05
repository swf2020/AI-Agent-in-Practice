# AI Agent 与大模型应用开发实战手册

> 🚀 **从入门到精通的 AI Agent 实战指南**

一本面向工程师的 AI Agent 实战手册，通过理论精讲 + 动手实验 + 面试真题的三位一体模式，带你从零构建可上线的 AI 产品。

---

## 🎯 这本书是什么？

**一句话概括**：通过 6–12 周系统学习，掌握从 LLM API 调用到生产级 Agent 部署的完整技能链。

**核心特点**：
- ✅ **双轨学习路径**：工程实战通道（6–8周）/ 求职面试通道（10–12周）
- ✅ **15+ 动手实验**：每个核心知识点配套可运行代码
- ✅ **4 个完整项目**：AI 选股分析师、企业知识库问答、Text-to-SQL、自动化工作流
- ✅ **生产级经验**：监控、成本控制、安全、云端部署全流程
- ✅ **面试题集**：200+ 道真实面试题，覆盖原理/工程/设计/视野

**适合人群**：
- 有 Python 基础，想快速做出 AI 产品的开发者
- 在职工程师，想转型 AI 应用方向
- 准备 AI Agent 相关岗位面试的求职者
- 参加黑客松或快速验证 MVP 的创业者

---

## 🗺️ 学习路线图

### 通道一：工程实战快速通道（6–8 周）
优先动手实验，快速产出可运行项目
- **Week 1**：LLM API 调用 → Prompt 工程 → 统一调用层封装
- **Week 2**：RAG 基础 → 从零搭建本地知识库问答系统
- **Week 3**：Advanced RAG → 工具调用接入
- **Week 4**：ReAct 范式 → LangGraph 有状态 Agent
- **Week 5**：Multi-Agent → 代码生成+审查双 Agent 系统
- **Week 6**：生产级监控 → 垂直项目实战
- **Week 7–8**：Docker + 云端部署全流程

### 通道二：求职面试深度通道（10–12 周）
理论精读 + 动手实验 + 面试题集三线并进
- **Week 1–2**：LLM 核心概念 + LoRA 原理 + Prompt/LLM 面试题
- **Week 3–4**：RAG 全流程 + RAGAS 评估 + RAG 面试题
- **Week 5**：Function Calling/MCP + 工具调用面试题
- **Week 6–7**：Agent 架构 + Multi-Agent + Agent 设计面试题
- **Week 8**：生产级落地 + 工程面试题
- **Week 9**：4 个垂直项目实战
- **Week 10**：技术演进史 + 视野类面试题
- **Week 11–12**：模拟面试 + 系统设计题

---

## 📚 核心内容

| 模块 | 主题 | 核心知识点 |
|------|------|------------|
| **00** | 写给读者 | 学习路线图、环境搭建、资料使用指南 |
| **01** | 大模型基础 | Token、采样策略、微调（LoRA/QLoRA）、Prompt Engineering |
| **02** | RAG | 向量数据库、检索策略、Advanced RAG、RAGAS 评估 |
| **03** | 工具使用 | Function Calling、MCP 协议、搜索/计算器/数据库工具 |
| **04** | Agent 核心 | ReAct、Planning（ToT/GoT）、记忆系统、LangGraph |
| **05** | Multi-Agent | 协作模式、AutoGen、CrewAI、代码审查双 Agent |
| **06** | 生产级落地 | 流式输出、成本控制、缓存、安全、可观测性 |
| **07** | 实战项目 | AI 选股分析师、知识库问答、Text-to-SQL、工作流 Agent、云端部署 |
| **08** | 技术演进 | 大模型发展史、Agent 演进路线、关键技术专题 |
| **09** | 面试真题 | 原理题、工程题、设计题、视野题、开放题 |
| **10** | 附录 | 框架对比、学习资源、术语表 |

---

## 🚀 快速开始

```bash
# 克隆项目
git clone https://github.com/swf2020/AI-Agent-in-Practice.git
cd AI-Agent-in-Practice

# 本地预览文档（推荐）
pip install mkdocs mkdocs-material
mkdocs serve
# 访问 http://127.0.0.1:8000/AI-Agent-in-Practice/

# 运行实战项目示例
cd libs/AI\ Agent\ In\ Practice/第七章\ 垂直场景实战项目/7.1\ 项目一：AI\ 选股分析师（基于\ TradingAgents）/
python main.py 1  # 运行实验一：美股分析
```

---

## 📂 目录结构

```
AI-Agent-in-Practice/
├── docs/                    # 教程文档（CC BY-NC-SA 4.0）
│   ├── index.md             # 文档首页
│   └── AI Agent In Practice/
│       ├── 00_写给读者/       # 学习路线图
│       ├── 01_大模型基础与 API 实战/
│       ├── 02_RAG（检索增强生成）/
│       ├── 03_Function Calling MCP 与工具使用/
│       ├── 04_AI Agent 核心架构/
│       ├── 05_Multi-Agent 系统/
│       ├── 06_生产级落地关键技术/
│       ├── 07_垂直场景实战项目/
│       ├── 08_AI 大模型与 Agent 技术演进全景/
│       ├── 09_面试题真题/
│       └── 10_附录/
├── libs/                    # 代码示例（MIT License）
│   └── AI Agent In Practice/
│       ├── 第一章 ~ 第六章/   # 各章节配套代码
│       └── 第七章 垂直场景实战项目/
│           ├── 7.1 AI 选股分析师/
│           ├── 7.2 企业知识库问答/
│           ├── 7.3 数据分析 Agent/
│           └── 7.4 自动化工作流 Agent/
├── README.md                # 项目入口
├── CONTRIBUTING.md          # 贡献指南
├── CHANGELOG.md            # 更新日志
├── LICENSE                 # 根目录许可证（Apache 2.0）
└── mkdocs.yml              # MkDocs 配置
```

---

## 🛠️ 技术栈

| 分类 | 技术 |
|------|------|
| **LLM API** | OpenAI GPT-4o、Claude 3.5、Gemini、DeepSeek、通义千问 |
| **Agent 框架** | LangGraph、AutoGen、CrewAI、TradingAgents |
| **向量数据库** | FAISS、Milvus、Chroma、Qdrant |
| **RAG** | LangChain、RAGAS、BGE-Reranker |
| **MCP** | MCP SDK、文件系统/数据库/代码执行沙箱 |
| **生产工具** | Redis、ARQ、FastAPI、LiteLLM、LangFuse |
| **部署** | Docker、AWS Lambda、阿里云函数计算、GitHub Actions |

---

## 📊 项目统计

| 指标 | 数量 |
|------|------|
| 总章节数 | 10 章 + 附录 |
| 文档文件数 | 63+ 个 |
| 动手实验 | 15+ 个 |
| 实战项目 | 4 个完整项目 + 云端部署 |
| 面试题 | 200+ 道 |
| 代码示例 | 100+ 个 |

---

## 📝 贡献指南

欢迎贡献代码、文档修正或新增内容！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与。

---

## 📄 许可证

本项目采用混合许可证：
- `/docs/` - **CC BY-NC-SA 4.0**（文档内容）
- `/libs/` - **MIT License**（代码示例）
- 根目录 - **Apache License 2.0**（配置文件）

详情请阅读 [LICENSE](LICENSE)。

---

## 🔗 相关链接

- 📖 **在线文档**：https://swf2020.github.io/AI-Agent-in-Practice/
- 📦 **代码仓库**：https://github.com/swf2020/AI-Agent-in-Practice
- 💬 **讨论区**：欢迎提交 Issue 和 PR
