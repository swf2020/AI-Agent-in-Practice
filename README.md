# AI Agent 与大模型应用开发实战手册

> 🚀 **从入门到精通的 AI Agent 实战指南**

这是一本关于 AI Agent 与大模型应用开发的实战手册，涵盖从基础概念到生产级部署的完整知识体系。

---

## 📚 目录结构

```
docs/
├── index.md                    # 学习路线图与内容索引
└── AI Agent 与大模型应用开发实战手册/
    ├── 第零章 写给读者/
    │   └── 学习路线图.md
    ├── 第一章 大模型基础与 API 实战/
    │   ├── 1.1 LLM核心概念.md
    │   ├── 1.2.1 微调技术全景与选型决策.md
    │   ├── 1.2.2 LoRA 原理精讲.md
    │   ├── 1.2.3 QLoRA 原理精讲.md
    │   ├── 1.2.4 数据工程：微调数据的准备与质量控制.md
    │   ├── 1.2.5 【动手一】用 QLoRA 微调 Qwen2.5-7B 指令模型.md
    │   ├── 1.2.6 【动手二】微调效果评估与对比实验.md
    │   ├── 1.2.7 【动手三】基于 Unsloth 的高效微调加速实战.md
    │   ├── 1.3 Prompt Engineering 实战.md
    │   ├── 1.3.1【动手一】流式输出 + 实时思维链可视化.md
    │   ├── 1.3.2 【动手二】构建一个提示词调试器.md
    │   ├── 1.3.3【动手三】多语言翻译质量评估器.md
    │   ├── 1.3.4 【动手四】自动化 Prompt 优化器（DSPy 入门）.md
    │   └── 1.4 【动手】统一封装多模型调用层.md
    ├── 第二章 RAG（检索增强生成）/
    │   ├── 2.1 RAG 架构全景.md
    │   ├── 2.2 Embedding 与向量数据库.md
    │   ├── 2.3 检索策略（稠密-稀疏-混合）.md
    │   ├── 2.4 【动手】从零搭建本地知识库问答系统.md
    │   ├── 2.5 【动手】Advanced RAG：重排序 + 查询改写.md
    │   └── 2.6 RAG 评估体系（RAGAS 框架）.md
    ├── 第三章 Function Calling MCP 与工具使用/
    │   ├── 3.1 Function Calling 原理与协议.md
    │   ├── 3.2 MCP 协议详解.md
    │   ├── 3.2.1 【动手一】文件系统操作 MCP Server.md
    │   ├── 3.2.2 【动手二】数据库查询 MCP Server.md
    │   ├── 3.2.3【动手三】代码执行沙箱 MCP Server.md
    │   ├── 3.3 【动手】给 LLM 接入搜索-计算器-数据库工具.md
    │   └── 3.4 工具可靠性与错误处理.md
    ├── 第四章 AI Agent 核心架构/
    │   ├── 4.1 Agent 定义：感知-规划-行动循环.md
    │   ├── 4.2 ReAct 范式实战.md
    │   ├── 4.3 Planning 策略（ToT-GoT-Plan-and-Execute）.md
    │   ├── 4.4 记忆系统设计.md
    │   └── 4.5 【动手】用 LangGraph 构建有状态 Agent.md
    ├── 第五章 Multi-Agent 系统/
    │   ├── 5.1 多 Agent 协作模式（层级-对等-流水线）.md
    │   ├── 5.2 AutoGen 框架实战.md
    │   ├── 5.3 CrewAI 角色分工实战.md
    │   └── 5.4 【动手】搭建代码生成+审查的双 Agent 系统.md
    ├── 第六章 生产级落地关键技术/
    │   ├── 6.1 流式输出与用户体验.md
    │   ├── 6.2 成本控制与 Token 优化.md
    │   ├── 6.3 缓存策略（Prompt Cache-Semantic Cache）.md
    │   ├── 6.4 安全与对齐（Prompt Injection 防御）.md
    │   ├── 6.5 可观测性（LangSmith-LangFuse）.md
    │   └── 6.6 【动手】构建带监控的生产级 Agent 服务.md
    ├── 第七章 垂直场景实战项目/
    │   ├── 7.1 项目一：AI 选股分析师（基于 TradingAgents）.md
    │   ├── 7.1.2 环境搭建与数据接入.md
    │   ├── 7.1.4 动手实验.md
    │   ├── 7.2 项目二：企业知识库智能问答.md
    │   ├── 7.2.5 评估与上线.md
    │   ├── 7.3 项目三：数据分析 Agent（Text-to-SQL）.md
    │   └── 7.4 项目四：自动化工作流 Agent.md
    └── 第八章 AI 大模型与 Agent 技术演进全景/
        ├── 8.1.1 前深度学习时代（2000-2017）.md
        ├── 8.1.2 Transformer 革命（2017-2019）.md
        ├── 8.1.3 规模定律与涌现能力（2020-2022）.md
        ├── 8.1.4 指令对齐与 RLHF 时代（2022-2023）.md
        ├── 8.1.5 开源生态爆发与长上下文竞赛（2023-2024）.md
        ├── 8.1.6 推理模型范式（2024-2025）.md
        ├── 8.2.1 前 LLM Agent 时代（约 1960s-2022）.md
        ├── 8.2.2 LLM Agent 萌芽期（2022-2023）.md
        ├── 8.2.3 框架与协议的规范化（2023-2024）.md
        ├── 8.2.4 Computer Use 与 GUI Agent（2024）.md
        ├── 8.2.5 Agentic AI 走向生产（2024-2025）.md
        ├── 8.3.1 上下文长度：从 512 Token 到无限流式记忆.md
        ├── 8.3.2 RAG 技术演进：关键字检索到知识图谱增强.md
        ├── 8.3.3 多模态：从 CLIP 到全模态统一模型.md
        ├── 8.3.4 高效推理：从量化压缩到推理加速芯片生态.md
        ├── 8.3.5 对齐技术：从 RLHF 到 DPO 到宪法 AI.md
        ├── 8.3.6 微调技术：从全参数到极低成本适配.md
        ├── 8.5.1 模型能力演进预判（2025-2030）.md
        ├── 8.5.2 Agent 架构演进预判.md
        ├── 8.5.3 技术瓶颈与开放性挑战.md
        ├── 8.5.4 基础设施与生态演进预判.md
        └── 8.5.5 社会影响与监管格局.md
```

---

## 📖 内容简介

### 第零章 写给读者
- 学习路线图（配技能树图）
- 两种学习通道：工程实战快速通道 vs 求职面试深度通道
- 环境搭建指南

### 第一章 大模型基础与 API 实战
- LLM 核心概念（Token、Temperature、上下文窗口）
- 大模型微调实战（LoRA / QLoRA）
- Prompt Engineering 实战
- 主流 API 对比接入（OpenAI / Claude / Gemini / DeepSeek / 通义千问）

### 第二章 RAG（检索增强生成）
- RAG 架构全景与演进路线
- Embedding 与向量数据库选型
- 检索策略（稠密/稀疏/混合）
- RAGAS 评估体系

### 第三章 Function Calling / MCP 与工具使用
- Function Calling 原理与协议
- MCP 协议详解与动手实践
- 工具可靠性与错误处理

### 第四章 AI Agent 核心架构
- Agent 定义：感知-规划-行动循环
- ReAct 范式实战
- Planning 策略（ToT / GoT / Plan-and-Execute）
- 记忆系统设计（短期/长期/外部记忆）

### 第五章 Multi-Agent 系统
- 多 Agent 协作模式（层级 / 对等 / 流水线）
- AutoGen 框架实战
- CrewAI 角色分工实战

### 第六章 生产级落地关键技术
- 流式输出与用户体验
- 成本控制与 Token 优化
- 缓存策略（Prompt Cache / Semantic Cache）
- 安全与对齐（Prompt Injection 防御）
- 可观测性（LangSmith / LangFuse）

### 第七章 垂直场景实战项目
- **项目一**：AI 选股分析师（基于 TradingAgents）
- **项目二**：企业知识库智能问答
- **项目三**：数据分析 Agent（Text-to-SQL）
- **项目四**：自动化工作流 Agent

### 第八章 AI 大模型与 Agent 技术演进全景
- 大语言模型发展史（2000-2025）
- AI Agent 发展史
- 关键技术专题演进（上下文长度、RAG、多模态、高效推理、对齐技术、微调技术）
- 各大厂商技术路线图与战略研判
- 未来演进路线图（2025-2030）

---

## 🎯 学习路径建议

### 工程实战快速通道（6-8 周）
适合有 Python 基础、想尽快做出可运行 AI 产品的开发者。

### 求职面试深度通道（10-12 周）
适合准备冲击大厂 AI 工程师岗位的求职者。

---

## 🚀 快速开始

```bash
# 安装 MkDocs
pip install mkdocs mkdocs-material

# 本地预览
mkdocs serve

# 构建静态站点
mkdocs build
```

---

## 🎤 面试刷题资源（interview/）

`interview/` 目录包含一套完整的 AI Agent 社招面试题集采集与解答系统。

### 📁 目录结构

```
interview/
├── ai-agent-interview-collector/
│   └── SKILL.md                      # Claude Code Skill 定义文件
├── ai_agent_interview_questions_20260502.md   # 面试题集（86题，8大模块）
└── answers/
    ├── module_00_other.md              # 其他/综合类（6题）
    ├── module_01_prompt_llm.md         # Prompt / LLM 原理类（12题）
    ├── module_02_rag.md               # RAG 设计与优化类（26题）
    ├── module_03_tool_calling.md      # 工具调用类（13题）
    ├── module_04_agent_architecture.md # Agent 架构设计类（15题）
    ├── module_05_multi_agent.md       # 多 Agent 设计类（12题）
    ├── module_06_engineering.md       # 工程落地类（12题）
    ├── module_07_deployment.md        # 云端部署类（11题）
    └── module_08_evolution.md         # 技术演进与视野类（10题）
```

### 功能说明

- **题集来源**：牛客/脉脉/小红书/B站/知乎/CSDN 等 14+ 中文平台，采集自近 12 个月真实社招面经
- **分类体系**：按 8 大技术模块 + 综合类组织，共 86 道去重题目
- **解答覆盖**：每题包含"考察点 → 解答思路 → 参考答案 + 加分项"四段式结构，共 117 题解答（7375 行）
- **高频标注**：题集末尾标注近 1 个月出现 3 次以上的热点题目

### SKILL.md 安装与使用

**安装步骤：**

```bash
# 将 skill 复制到你的 Claude Code skills 目录
cp -r interview/ai-agent-interview-collector ~/.claude/skills/ai-agent-interview-collector
```

**使用方式：**

安装后，在 Claude Code 对话中说以下任意一句即可触发：

- "更新面试题集"
- "刷新 AI Agent 面试题"
- "收集最新面试题"
- "跑一次面试题采集"

Skill 会自动执行：从中文互联网搜索近 12 个月的真实面试题 → 清洗去重 → 按 8 大模块分类 → 生成 Markdown 题集 → 并行调用大模型为每题生成解答思路与参考答案 → 保存到本地。

**自定义输出路径：**

```
跑一次面试题采集，输出到 ~/my-projects/interview-questions/
```

---

## 📊 统计信息

| 项目 | 数量 |
|------|------|
| 总章节数 | 8 章 |
| 文档文件数 | 63 个 |
| 动手实验 | 15+ 个 |
| 面试题集（interview/） | 86 题 + 117 题解答 |

---

## 📝 贡献指南

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

Apache License
