"""
测试数据集：3个领域 × 10句 = 30条英文原文
每条包含：原文、参考译文、人工评分（忠实度/流畅度/术语，1-5）
"""
from dataclasses import dataclass


@dataclass
class TestItem:
    id: str
    domain: str                    # tech / legal / casual
    source: str                    # 英文原文
    reference: str                 # 人工参考译文
    human_scores: dict[str, float] # {"faithfulness": 4.5, "fluency": 4.0, "terminology": 4.5}


TEST_SET: list[TestItem] = [
    # ── 科技新闻领域 ──────────────────────────────────────────────
    TestItem(
        id="tech_01",
        domain="tech",
        source="The transformer architecture revolutionized natural language processing by replacing recurrent neural networks with self-attention mechanisms.",
        reference="Transformer架构通过用自注意力机制取代循环神经网络，彻底革新了自然语言处理领域。",
        human_scores={"faithfulness": 5.0, "fluency": 5.0, "terminology": 5.0},
    ),
    TestItem(
        id="tech_02",
        domain="tech",
        source="Retrieval-Augmented Generation combines the parametric knowledge of language models with non-parametric retrieval from external knowledge bases.",
        reference="检索增强生成将语言模型的参数化知识与从外部知识库进行的非参数化检索相结合。",
        human_scores={"faithfulness": 5.0, "fluency": 5.0, "terminology": 5.0},
    ),
    TestItem(
        id="tech_03",
        domain="tech",
        source="Quantization reduces model size by representing weights in lower precision formats, trading some accuracy for significant memory savings.",
        reference="量化通过以低精度格式表示权重来缩小模型体积，以少量精度损失换取显著的内存节省。",
        human_scores={"faithfulness": 5.0, "fluency": 5.0, "terminology": 5.0},
    ),
    TestItem(
        id="tech_04",
        domain="tech",
        source="The key-value cache stores intermediate attention computations, allowing the model to avoid redundant calculations during autoregressive generation.",
        reference="键值缓存存储中间注意力计算结果，使模型在自回归生成过程中避免冗余计算。",
        human_scores={"faithfulness": 5.0, "fluency": 5.0, "terminology": 5.0},
    ),
    TestItem(
        id="tech_05",
        domain="tech",
        source="Fine-tuning on domain-specific data allows pre-trained models to adapt their general capabilities to specialized tasks without training from scratch.",
        reference="在领域专用数据上进行微调，使预训练模型无需从头训练即可将通用能力适配至专项任务。",
        human_scores={"faithfulness": 5.0, "fluency": 5.0, "terminology": 5.0},
    ),
    # ── 法律条款领域 ──────────────────────────────────────────────
    TestItem(
        id="legal_01",
        domain="legal",
        source="The licensee shall not sublicense, sell, resell, transfer, assign, or otherwise commercially exploit or make available to any third party the Software.",
        reference="被许可方不得将本软件再许可、出售、转售、转让、转让或以其他方式进行商业利用，或向任何第三方提供本软件。",
        human_scores={"faithfulness": 4.5, "fluency": 3.5, "terminology": 4.5},
    ),
    TestItem(
        id="legal_02",
        domain="legal",
        source="This Agreement constitutes the entire agreement between the parties with respect to the subject matter hereof and supersedes all prior agreements.",
        reference="本协议构成双方就本协议主题事项达成的完整协议，并取代双方此前就该主题事项签订的所有协议。",
        human_scores={"faithfulness": 5.0, "fluency": 4.0, "terminology": 5.0},
    ),
    TestItem(
        id="legal_03",
        domain="legal",
        source="In no event shall either party be liable to the other for any indirect, incidental, special, exemplary, or consequential damages.",
        reference="在任何情况下，任何一方均不对另一方承担任何间接、附带、特殊、示范性或后果性损害赔偿责任。",
        human_scores={"faithfulness": 5.0, "fluency": 4.5, "terminology": 5.0},
    ),
    TestItem(
        id="legal_04",
        domain="legal",
        source="The arbitration shall be conducted in accordance with the rules of the International Chamber of Commerce then in effect.",
        reference="仲裁应依据国际商会届时有效的仲裁规则进行。",
        human_scores={"faithfulness": 5.0, "fluency": 5.0, "terminology": 5.0},
    ),
    TestItem(
        id="legal_05",
        domain="legal",
        source="Each party represents and warrants that it has full power and authority to enter into and perform its obligations under this Agreement.",
        reference="各方声明并保证其拥有订立本协议并履行其在本协议项下义务的完全权力和授权。",
        human_scores={"faithfulness": 5.0, "fluency": 4.5, "terminology": 5.0},
    ),
    # ── 日常口语领域 ──────────────────────────────────────────────
    TestItem(
        id="casual_01",
        domain="casual",
        source="I've been swamped with work lately and haven't had a chance to catch up with anyone.",
        reference="最近工作忙得焦头烂额，都没机会跟大家联系一下。",
        human_scores={"faithfulness": 4.5, "fluency": 5.0, "terminology": 4.0},
    ),
    TestItem(
        id="casual_02",
        domain="casual",
        source="That new coffee shop on the corner is a total vibe — you should definitely check it out.",
        reference="街角那家新开的咖啡店氛围超棒，你一定要去看看。",
        human_scores={"faithfulness": 4.5, "fluency": 5.0, "terminology": 4.0},
    ),
    TestItem(
        id="casual_03",
        domain="casual",
        source="She's been killing it at the gym — I've never seen anyone get results that fast.",
        reference="她最近在健身房简直开挂了——我从没见过有人进步这么快。",
        human_scores={"faithfulness": 4.5, "fluency": 5.0, "terminology": 4.0},
    ),
    TestItem(
        id="casual_04",
        domain="casual",
        source="Let's grab some takeout tonight, I really don't feel like cooking after that meeting.",
        reference="今晚叫外卖吧，开完那个会我真的不想做饭了。",
        human_scores={"faithfulness": 5.0, "fluency": 5.0, "terminology": 4.0},
    ),
    TestItem(
        id="casual_05",
        domain="casual",
        source="He somehow managed to ace the exam without studying at all — classic him.",
        reference="他愣是没复习就把考试过了——太他了。",
        human_scores={"faithfulness": 4.0, "fluency": 4.5, "terminology": 4.0},
    ),
]