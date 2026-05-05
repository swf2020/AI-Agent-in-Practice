# smoke_test.py — 端到端冒烟测试，验证完整链路
import asyncio
import httpx
import os

BASE_URL = "http://localhost:8000"

# 禁用代理，避免连接本地服务时走代理
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)


async def smoke_test():
    async with httpx.AsyncClient(timeout=120) as client:
        print("1️⃣ 健康检查...")
        resp = await client.get(f"{BASE_URL}/health")
        assert resp.status_code == 200, f"健康检查失败: {resp.text}"
        print(f"   ✅ {resp.json()}")

        print("\n2️⃣ 同步对话接口...")
        resp = await client.post(f"{BASE_URL}/chat", json={
            "message": "现在几点了？",
            "session_id": "smoke-test-001",
            "user_id": "tester",
        })
        assert resp.status_code == 200, f"同步接口失败: {resp.text}"
        data = resp.json()
        print(f"   ✅ 回答: {data['output'][:50]}...")
        print(f"   ✅ 耗时: {data['duration_ms']}ms")

        print("\n3️⃣ 异步任务接口...")
        resp = await client.post(f"{BASE_URL}/task", json={
            "message": "计算 42 * 1337 + 100 的结果",
            "session_id": "smoke-test-002",
        })
        assert resp.status_code == 200, f"提交任务失败: {resp.text}"
        task_id = resp.json()["task_id"]
        print(f"   ✅ 任务 ID: {task_id}")

        print("   ⏳ 轮询任务结果...")
        for attempt in range(30):
            await asyncio.sleep(3)
            poll_resp = await client.get(f"{BASE_URL}/task/{task_id}")
            result = poll_resp.json()
            status = result["status"]
            print(f"   第 {attempt+1} 次轮询，状态: {status}")
            if status == "success":
                print(f"   ✅ 任务完成: {result['result'][:80]}...")
                print(f"   ✅ 耗时: {result['duration_ms']}ms")
                break
            elif status == "failed":
                raise AssertionError(f"任务失败: {result['error']}")
        else:
            raise AssertionError("任务超时（90s），请检查 ARQ Worker 是否正常运行")

        print("\n4️⃣ Prometheus 指标验证...")
        resp = await client.get(f"{BASE_URL}/metrics")
        assert "agent_requests_total" in resp.text, "Prometheus 指标未找到"
        print("   ✅ 指标上报正常")

        print("\n🎉 所有冒烟测试通过！打开 http://localhost:3000 查看 Grafana Dashboard")


asyncio.run(smoke_test())