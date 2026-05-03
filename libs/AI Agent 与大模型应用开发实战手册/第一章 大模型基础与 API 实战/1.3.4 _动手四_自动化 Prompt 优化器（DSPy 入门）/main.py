#!/usr/bin/env python3
"""
DSPy 自动化 Prompt 优化器 - 端到端完整示例
运行：python main.py
"""
import os
import random
import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot
from dotenv import load_dotenv
from typing import Literal

load_dotenv()


def configure_lm():
    """配置 LLM"""
    lm = dspy.LM(
        model="deepseek/deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0,
        cache=True,
    )
    dspy.configure(lm=lm)
    print("✅ LLM 配置完成：DeepSeek-V3")


class SentimentSignature(dspy.Signature):
    """分析中文用户评论的情感倾向"""
    text: str = dspy.InputField(desc="用户评论原文")
    reason: str = dspy.OutputField(desc="判断情感的核心依据，1-2句话")
    sentiment: Literal["正面", "负面", "中性"] = dspy.OutputField()


class SentimentClassifier(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classify = dspy.ChainOfThought(SentimentSignature)
    
    def forward(self, text: str) -> dspy.Prediction:
        return self.classify(text=text)


def accuracy_metric(example, prediction, trace=None):
    """评估指标：准确率"""
    return prediction.sentiment == example.sentiment


def main():
    """主函数"""
    configure_lm()
    
    raw_data = [
        {"text": "包装很好，物流超快！", "sentiment": "正面"},
        {"text": "质量太差了，买了就坏", "sentiment": "负面"},
        {"text": "东西还可以，没有特别惊喜", "sentiment": "中性"},
        {"text": "颜值超高，闺蜜都问哪买的", "sentiment": "正面"},
        {"text": "和描述严重不符，申请退款", "sentiment": "负面"},
        {"text": "价格一般，质量也一般", "sentiment": "中性"},
        {"text": "客服态度很好，五星好评", "sentiment": "正面"},
        {"text": "味道刺鼻，通风好几天还有味", "sentiment": "负面"},
        {"text": "功能就那样，没什么特别的", "sentiment": "中性"},
        {"text": "第一次买，超级好用，会回购", "sentiment": "正面"},
    ]
    
    random.seed(42)
    random.shuffle(raw_data)
    examples = [dspy.Example(**d).with_inputs("text") for d in raw_data]
    devset = examples[:3]
    trainset = examples[3:]
    
    print("=" * 50)
    print("Step 1: 运行 Baseline（无优化）")
    baseline = SentimentClassifier()
    
    result = baseline(text="这个产品真的超级棒，强烈推荐！")
    print(f"  输入：这个产品真的超级棒，强烈推荐！")
    print(f"  推理：{result.reason}")
    print(f"  预测：{result.sentiment}")
    
    evaluator = Evaluate(devset=devset, metric=accuracy_metric, num_threads=2, display_progress=False)
    baseline_score = evaluator(baseline)
    print(f"\n📊 Baseline 准确率: {float(baseline_score):.1f}%")
    
    print("\n" + "=" * 50)
    print("Step 2: BootstrapFewShot 优化")
    optimizer = BootstrapFewShot(metric=accuracy_metric, max_bootstrapped_demos=2)
    optimized = optimizer.compile(SentimentClassifier(), trainset=trainset)
    optimized_score = evaluator(optimized)
    print(f"📊 优化后准确率: {float(optimized_score):.1f}%")
    print(f"📈 提升: +{float(optimized_score) - float(baseline_score):.1f}%")
    
    optimized.save("best_sentiment_classifier.json")
    print("\n✅ 最优模型已保存")
    
    loaded = SentimentClassifier()
    loaded.load("best_sentiment_classifier.json")
    test_result = loaded(text="收到货后发现是坏的，气死了")
    print(f"\n🔄 加载验证 - 预测：{test_result.sentiment}（预期：负面）")


if __name__ == "__main__":
    main()
