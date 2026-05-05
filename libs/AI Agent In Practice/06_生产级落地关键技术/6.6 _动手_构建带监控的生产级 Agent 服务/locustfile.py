from __future__ import annotations
import time
import random
from locust import HttpUser, task, between, events


# 测试用的问题集，覆盖不同复杂度
QUESTIONS = [
    "现在几点了？",                          # 简单：单工具调用
    "计算 (123 * 456 + 789) / 3 的结果",   # 中等：计算器工具
    "搜索一下最新的 AI Agent 进展，并计算如果每天学习 2 小时，30 天能学多少小时", # 复杂：多工具
]


class AgentUser(HttpUser):
    """模拟真实用户行为：提交任务 → 轮询结果。"""

    # 每个虚拟用户在两次请求之间等待 1-3 秒，模拟真实用户节奏
    wait_time = between(1, 3)

    @task(3)
    def test_sync_chat(self):
        """测试同步接口（权重 3：高频）。"""
        question = random.choice(QUESTIONS[:2])  # 同步接口只测简单问题
        with self.client.post(
            "/chat",
            json={
                "message": question,
                "session_id": f"locust-{self.user_id}",
                "user_id": f"user-{self.user_id}",
            },
            timeout=60,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"状态码 {response.status_code}: {response.text[:200]}")

    @task(1)
    def test_async_task(self):
        """测试异步接口（权重 1：低频）：提交任务 → 轮询结果。"""
        question = random.choice(QUESTIONS)

        # Step 1: 提交任务
        submit_resp = self.client.post(
            "/task",
            json={
                "message": question,
                "session_id": f"locust-async-{self.user_id}",
            },
            timeout=10,
        )

        if submit_resp.status_code != 200:
            return

        task_id = submit_resp.json().get("task_id")
        if not task_id:
            return

        # Step 2: 轮询结果（最多等 120s）
        max_polls = 40
        for _ in range(max_polls):
            time.sleep(3)
            poll_resp = self.client.get(f"/task/{task_id}", timeout=10)
            if poll_resp.status_code == 200:
                data = poll_resp.json()
                if data["status"] in ("success", "failed"):
                    break


@events.quitting.add_listener
def on_locust_quit(environment, **kwargs):
    """压测结束时打印容量规划建议。"""
    stats = environment.runner.stats.total
    print("\n" + "="*60)
    print("📊 压测结论与容量规划建议")
    print("="*60)
    print(f"总请求数: {stats.num_requests}")
    print(f"失败数: {stats.num_failures}")
    print(f"失败率: {stats.fail_ratio:.2%}")
    print(f"RPS (avg): {stats.current_rps:.1f}")
    print(f"P50 延迟: {stats.get_response_time_percentile(0.50):.0f}ms")
    print(f"P90 延迟: {stats.get_response_time_percentile(0.90):.0f}ms")
    print(f"P99 延迟: {stats.get_response_time_percentile(0.99):.0f}ms")

    # 容量规划：根据压测结果推算生产所需实例数
    target_rps = 100  # 生产目标 QPS
    current_rps = max(stats.current_rps, 0.1)
    scale_factor = target_rps / current_rps
    print(f"\n若目标 QPS={target_rps}，当前压测 RPS={current_rps:.1f}")
    print(f"建议实例数倍数: {scale_factor:.1f}x（含 20% buffer 则 {scale_factor*1.2:.1f}x）")