# 附录 B：学习资源导航

---

## 一、必读论文清单

以下按主题分类，列出本书涉及的核论文，附 arXiv 链接与一句话摘要。

### 1. Transformer 与基础架构

| 论文 | arXiv | 摘要 |
|------|-------|------|
| **Attention Is All You Need** (Vaswani et al., 2017) | [1706.03762](https://arxiv.org/abs/1706.03762) | 提出 Transformer 架构，以 Self-Attention 完全替代 RNN/CNN，奠定现代大模型基础 |
| **BERT** (Devlin et al., 2018) | [1810.04805](https://arxiv.org/abs/1810.04805) | 双向掩码语言模型，开创"预训练-微调"范式，在 11 项 NLP 任务上创 SOTA |
| **GPT-3** (Brown et al., 2020) | [2005.14165](https://arxiv.org/abs/2005.14165) | 175B 参数语言模型，首次展示少样本学习能力，开启大模型时代 |
| **Scaling Laws** (Kaplan et al., 2020) | [2001.08361](https://arxiv.org/abs/2001.08361) | 发现模型性能与参数量、数据量、算力之间存在幂律关系 |
| **Chinchilla** (Hoffmann et al., 2022) | [2203.15556](https://arxiv.org/abs/2203.15556) | 修正 Scaling Laws，证明最优训练需参数量与 token 数等比例增长 |
| **GPT-4 技术报告** (OpenAI, 2023) | [2303.08774](https://arxiv.org/abs/2303.08774) | 多模态能力提升与安全评估框架，定义行业新基准 |
| **LLaMA** (Touvron et al., 2023) | [2302.13971](https://arxiv.org/abs/2302.13971) | Meta 开源基座模型系列，用更少数据达到与闭源模型相当的性能 |
| **Mistral** (Jiang et al., 2023) | [2310.06825](https://arxiv.org/abs/2310.06825) | 引入 GQA 和滑动窗口 Attention，8B 模型超越 Llama-2 13B |
| **Mixtral 8x7B** (Jiang et al., 2024) | [2401.04088](https://arxiv.org/abs/2401.04088) | 稀疏 MoE 架构，46.7B 参数但每次仅激活 12.9B，效率与质量兼优 |
| **RoPE** (Su et al., 2021) | [2104.09864](https://arxiv.org/abs/2104.09864) | 旋转位置编码，将绝对位置转化为相对位置旋转变换，外推能力强 |
| **ALiBi** (Press et al., 2021) | [2108.12409](https://arxiv.org/abs/2108.12409) | 在注意力分数上加距离线性偏置，实现训练短序列推理长序列 |

### 2. 微调与对齐

| 论文 | arXiv | 摘要 |
|------|-------|------|
| **LoRA** (Hu et al., 2021) | [2106.09685](https://arxiv.org/abs/2106.09685) | 低秩自适应微调，将可训练参数量降至全量的 0.01%，效果接近全量微调 |
| **QLoRA** (Dettmers et al., 2023) | [2305.14314](https://arxiv.org/abs/2305.14314) | NF4 量化 + LoRA，65B 模型可在单张 A100 上微调 |
| **LLM.int8()** (Dettmers et al., 2022) | [2208.07339](https://arxiv.org/abs/2208.07339) | 混合精度 INT8 量化，使大模型推理显存减半且精度损失极小 |
| **GPTQ** (Frantar et al., 2022) | [2210.17323](https://arxiv.org/abs/2210.17323) | 训练后 4-bit 量化方法，逐层贪心量化保持接近 FP16 的效果 |
| **AWQ** (Lin et al., 2023) | [2306.00978](https://arxiv.org/abs/2306.00978) | 激活感知的 4-bit 量化，保护重要权重通道，量化精度优于 GPTQ |
| **InstructGPT** (Ouyang et al., 2022) | [2203.02155](https://arxiv.org/abs/2203.02155) | RLHF 三阶段训练流程，使 GPT-3 成为可对话的 ChatGPT 原型 |
| **Constitutional AI** (Bai et al., 2022) | [2212.08073](https://arxiv.org/abs/2212.08073) | 用原则（Constitution）替代人工偏好标注，实现无害性对齐 |
| **DPO** (Rafailov et al., 2023) | [2305.18290](https://arxiv.org/abs/2305.18290) | 直接偏好优化，将 RLHF 的 RL 优化转化为分类损失，无需 Reward Model |
| **GRPO** (Shao et al., 2024) | [2402.03300](https://arxiv.org/abs/2402.03300) | 组相对策略优化，省去 Critic 模型，通过组内归一化估计优势函数 |
| **Weak-to-Strong** (Burns et al., 2023) | [2312.09390](https://arxiv.org/abs/2312.09390) | 用弱模型监督强模型的对齐方法，探索超级对齐的技术路径 |

### 3. Agent 与工具调用

| 论文 | arXiv | 摘要 |
|------|-------|------|
| **ReAct** (Yao et al., 2022) | [2210.03629](https://arxiv.org/abs/2210.03629) | 推理与行动交替的 Agent 范式，在多种任务上超越纯推理或纯行动 |
| **ToT** (Yao et al., 2023) | [2305.10601](https://arxiv.org/abs/2305.10601) | 思维树，在 CoT 基础上加入 BFS/DFS 搜索与回溯，提升复杂推理 |
| **GoT** (Besta et al., 2024) | [2308.09687](https://arxiv.org/abs/2308.09687) | 思维图，将推理过程建模为图结构，支持合并、分支等高级操作 |
| **Plan-and-Execute** (Wang et al., 2023) | [2305.04091](https://arxiv.org/abs/2305.04091) | 先规划全局计划再逐步执行，规划与执行分离提升任务完成率 |
| **Toolformer** (Schick et al., 2023) | [2302.04761](https://arxiv.org/abs/2302.04761) | 模型自学习何时调用外部工具，用少量示例即可教会 API 调用 |
| **AutoGen** (Wu et al., 2023) | [2308.08155](https://arxiv.org/abs/2308.08155) | 多 Agent 对话框架，通过可对话的 Agent 构建灵活的工作流 |
| **Emergent Abilities** (Wei et al., 2022) | [2206.07682](https://arxiv.org/abs/2206.07682) | 讨论大模型在规模增长后"突然获得"的能力，引发学界广泛讨论 |

### 4. RAG 与检索

| 论文 | arXiv | 摘要 |
|------|-------|------|
| **RAG** (Lewis et al., 2020) | [2005.11401](https://arxiv.org/abs/2005.11401) | 检索增强生成开山之作，将检索文档与生成模型联合训练 |
| **DPR** (Karpukhin et al., 2020) | [2004.04906](https://arxiv.org/abs/2004.04906) | 稠密段落检索，用双编码器替代 BM25，显著提升开放域 QA 效果 |
| **RAG Survey** (Gao et al., 2023) | [2312.10997](https://arxiv.org/abs/2312.10997) | RAG 技术综述，定义 Naive → Advanced → Modular 三代架构演进 |
| **GraphRAG** (Edge et al., 2024) | [2404.16130](https://arxiv.org/abs/2404.16130) | 微软知识图谱增强 RAG，用图结构捕捉实体关系，提升复杂推理问答 |
| **Lost in the Middle** (Liu et al., 2023) | [2307.03172](https://arxiv.org/abs/2307.03172) | LLM 倾向于利用上下文的开头和结尾信息，中间内容容易被忽略 |

### 5. 多模态

| 论文 | arXiv | 摘要 |
|------|-------|------|
| **CLIP** (Radford et al., 2021) | [2103.00020](https://arxiv.org/abs/2103.00020) | 对比语言-图像预训练，通过图文对对比学习实现零样本分类 |
| **LLaVA** (Liu et al., 2023) | [2304.08485](https://arxiv.org/abs/2304.08485) | 将视觉编码器通过 MLP 投影到 LLM token 空间，实现视觉指令微调 |
| **Flamingo** (Alayrac et al., 2022) | [2204.14198](https://arxiv.org/abs/2204.14198) | 跨模态注意力桥接视觉与语言，支持交错图文输入的对话 |
| **Gemini 1.5** (Gemini Team, 2024) | [2403.05530](https://arxiv.org/abs/2403.05530) | 百万 token 上下文多模态模型，支持视频/音频/文本/图像混合输入 |

### 6. 推理效率与部署

| 论文 | arXiv | 摘要 |
|------|-------|------|
| **FlashAttention** (Dao et al., 2022) | [2205.14135](https://arxiv.org/abs/2205.14135) | IO 感知的 Attention 实现，通过分块计算减少 HBM 访问，提速 3-4x |
| **vLLM / PagedAttention** (Kwon et al., 2023) | [2309.06180](https://arxiv.org/abs/2309.06180) | 分页注意力，将 KV Cache 视为虚拟内存管理，吞吐量提升 2-4x |
| **Mamba** (Gu & Dao, 2023) | [2312.00752](https://arxiv.org/abs/2312.00752) | 选择性状态空间模型，O(n) 复杂度挑战 Transformer 权威 |
| **Speculative Decoding** (Leviathan et al., 2022) | [2211.17192](https://arxiv.org/abs/2211.17192) | 用小模型生成草稿、大模型验证，在不损失质量的前提下加速推理 |
| **BitNet** (Wang et al., 2023) | [2310.11453](https://arxiv.org/abs/2310.11453) | 1-bit Transformer，将权重量化为 {-1, 0, 1}，大幅降低计算和存储 |
| **Ring Attention** (Liu et al., 2023) | [2310.01889](https://arxiv.org/abs/2310.01889) | 分布式超长上下文 Attention，通过环形通信实现万亿 token 处理 |
| **SGLang** (Zheng et al., 2023) | [2312.07104](https://arxiv.org/abs/2312.07104) | 结构化的 LLM 程序运行时，提供高效的推理服务框架 |
| **DeepSeek-R1** (DeepSeek-AI, 2025) | [2501.12948](https://arxiv.org/abs/2501.12948) | 开源推理模型，纯 GRPO 训练即达到 o1 级别推理能力 |

---

## 二、优质开源项目推荐

### Agent 框架

| 项目 | GitHub | Stars | 适用场景 |
|------|--------|:-----:|---------|
| **LangChain** | [github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain) | ~136K | LLM 应用开发的事实标准，Chain/Agent/Memory 抽象 |
| **LangGraph** | [github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | ~15K | 有状态 Agent 工作流，循环/分支/人工审批 |
| **LlamaIndex** | [github.com/run-llama/llama_index](https://github.com/run-llama/llama_index) | ~46K | RAG 专项框架，文档索引与检索优化 |
| **AutoGen** | [github.com/microsoft/autogen](https://github.com/microsoft/autogen) | ~57K | 多 Agent 对话协作，代码执行沙箱 |
| **CrewAI** | [github.com/crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | ~50K | 角色分工 Multi-Agent 协作，API 简单直观 |
| **Dify** | [github.com/langgenius/dify](https://github.com/langgenius/dify) | ~120K+ | 可视化 LLM 应用平台，企业级一站式服务 |
| **Flowise** | [github.com/FlowiseAI/Flowise](https://github.com/FlowiseAI/Flowise) | ~52K | 拖拽式 LLM 应用构建，零代码入门 |

### 模型推理

| 项目 | GitHub | Stars | 适用场景 |
|------|--------|:-----:|---------|
| **vLLM** | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) | ~50K+ | 高吞吐量 LLM 推理服务，PagedAttention |
| **Ollama** | [github.com/ollama/ollama](https://github.com/ollama/ollama) | ~120K+ | 本地模型一键部署，GGUF 格式推理 |
| **llama.cpp** | [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) | ~75K+ | CPU/GPU 混合推理，边缘设备部署 |
| **TensorRT-LLM** | [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | ~20K+ | NVIDIA GPU 极致推理优化 |
| **SGLang** | [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) | ~25K+ | 结构化 LLM 程序运行时 |

### 向量数据库

| 项目 | GitHub | Stars | 适用场景 |
|------|--------|:-----:|---------|
| **Qdrant** | [github.com/qdrant/qdrant](https://github.com/qdrant/qdrant) | ~30K+ | Rust 编写，高性能向量数据库，支持过滤 |
| **Milvus** | [github.com/milvus-io/milvus](https://github.com/milvus-io/milvus) | ~30K+ | 分布式向量数据库，云原生架构 |
| **Chroma** | [github.com/chroma-core/chroma](https://github.com/chroma-core/chroma) | ~15K+ | 轻量嵌入式向量库，适合快速原型 |
| **PGVector** | [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector) | ~12K+ | PostgreSQL 向量扩展，已有数据库直接复用 |

### 微调与训练

| 项目 | GitHub | Stars | 适用场景 |
|------|--------|:-----:|---------|
| **Unsloth** | [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth) | ~30K+ | 2x 训练速度 + 显存减半，一键微调模板 |
| **PEFT** | [github.com/huggingface/peft](https://github.com/huggingface/peft) | ~20K+ | HuggingFace 参数高效微调库，LoRA/QLoRA 等 |
| **TRL** | [github.com/huggingface/trl](https://github.com/huggingface/trl) | ~15K+ | Transformer 强化学习库，SFT/DPO/PPO |
| **LLaMA-Factory** | [github.com/hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | ~50K+ | 可视化微调工具，支持 100+ 模型 |
| **DeepSpeed** | [github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) | ~35K+ | 分布式训练优化，ZeRO 显存优化 |

### 可观测性与评估

| 项目 | GitHub | Stars | 适用场景 |
|------|--------|:-----:|---------|
| **LangFuse** | [github.com/langfuse/langfuse](https://github.com/langfuse/langfuse) | ~10K+ | 开源 LLM 可观测平台，全链路 Trace |
| **RAGAS** | [github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas) | ~10K+ | RAG 系统自动评估，四大核心指标 |
| **LangSmith** | [smith.langchain.com](https://smith.langchain.com) | - | LangChain 官方可观测服务（SaaS） |

### 其他实用工具

| 项目 | GitHub | Stars | 适用场景 |
|------|--------|:-----:|---------|
| **LiteLLM** | [github.com/BerriAI/litellm](https://github.com/BerriAI/litellm) | ~20K+ | 统一 LLM 调用接口，多 Provider Fallback |
| **DSPy** | [github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) | ~25K+ | 声明式 Prompt 优化，自动搜索最优 Few-shot |
| **markitdown** | [github.com/microsoft/markitdown](https://github.com/microsoft/markitdown) | ~20K+ | 微软文档解析工具，PDF/Word/PPT 转 Markdown |
| **GPTCache** | [github.com/zilliztech/GPTCache](https://github.com/zilliztech/GPTCache) | ~10K+ | 语义缓存层，相似查询直接返回 |
| **E2B** | [github.com/e2b-dev/e2b](https://github.com/e2b-dev/e2b) | ~10K+ | 云端代码沙箱，安全执行 Agent 生成的代码 |

---

## 三、持续跟进渠道

### Newsletter

| 名称 | 作者 | 特点 | 订阅地址 |
|------|------|------|---------|
| **The Batch** | Andrew Ng / DeepLearning.AI | 每周 AI 行业综述，适合把握大局 | [deeplearning.ai/the-batch](https://www.deeplearning.ai/the-batch/) |
| **Import AI** | Jack Clark | 深度技术分析，涵盖论文解读和行业洞察 | [importai.substack.com](https://importai.substack.com/) |
| **Lilian Weng's Blog** | Lilian Weng (OpenAI) | 架构演进的深度长文，每篇都是 mini 教程 | [lilianweng.github.io](https://lilianweng.github.io/) |
| **Sebastian Raschka's Blog** | Sebastian Raschka | ML 教程 + 最新论文解读，代码示例丰富 | [sebastianraschka.com](https://sebastianraschka.com/) |
| **Hugging Face Newsletter** | Hugging Face | 开源模型和社区动态，每月一期 | [huggingface.co/newsletter](https://huggingface.co/newsletter) |

### X / Twitter 账号

| 账号 | 领域 | 关注理由 |
|------|------|---------|
| [@AndrewYNg](https://x.com/AndrewYNg) | AI 教育 | 行业趋势判断，课程发布 |
| [@lilianweng](https://x.com/lilianweng) | AI 研究 | OpenAI 研究博客，深度架构分析 |
| [@JimFan](https://x.com/drJimFan) | AI 研究 | NVIDIA 科学家，Agent/世界模型方向 |
| [@hardmaru](https://x.com/hardmaru) | AI 研究 | Google Brain 研究员，生成模型与 Agent |
| [@amasad](https://x.com/amasad) | 开发者工具 | Replit CEO，AI 编程工具趋势 |
| [@LangChainAI](https://x.com/LangChainAI) | 框架动态 | LangChain 官方，产品更新 |
| [@anthropic](https://x.com/anthropic) | 厂商动态 | Anthropic 官方，Claude 发布与安全 |
| [@OpenAI](https://x.com/OpenAI) | 厂商动态 | OpenAI 官方，产品与模型发布 |
| [@GoogleDeepMind](https://x.com/GoogleDeepMind) | 厂商动态 | Google DeepMind 官方，Gemini 与研究 |
| [@DeepSeekAI](https://x.com/DeepSeekAI) | 开源模型 | DeepSeek 官方，模型与技术更新 |

### Discord / 社区

| 社区 | 平台 | 特点 |
|------|------|------|
| **LangChain Discord** | Discord | LangChain/LangGraph 用户社区，问题解答 |
| **LlamaIndex Discord** | Discord | RAG 场景交流，文档处理最佳实践 |
| **Hugging Face Discord** | Discord | 开源模型社区，模型发布 + 技术交流 |
| **Datawhale** | 微信/知乎/GitHub | 中文 AI 学习社区，教程 + 开源项目 |
| **稀土掘金 AI 频道** | 掘金 | 中文开发者技术社区，实战经验分享 |

### 论文跟踪

| 渠道 | 用途 |
|------|------|
| [arXiv cs.CL](https://arxiv.org/list/cs.CL/recent) | 计算语言学最新论文 |
| [arXiv cs.AI](https://arxiv.org/list/cs.AI/recent) | AI 最新论文 |
| [Hugging Face Papers](https://huggingface.co/papers) | 每日 AI 论文摘要 + 社区讨论 |
| [Papers With Code](https://paperswithcode.com/) | 论文 + 代码关联，追踪实现 |
