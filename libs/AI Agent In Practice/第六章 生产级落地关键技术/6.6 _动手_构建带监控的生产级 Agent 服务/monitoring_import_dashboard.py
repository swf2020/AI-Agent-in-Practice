"""
自动向 Grafana 导入 Agent 服务监控 Dashboard。
运行：python monitoring/import_dashboard.py
"""
import json
import httpx

GRAFANA_URL = "http://localhost:3000"
AUTH = ("admin", "admin123")

# 核心 Dashboard 配置（精简版，聚焦最重要的 4 个指标）
dashboard_config = {
    "dashboard": {
        "title": "Agent Service Overview",
        "refresh": "30s",
        "panels": [
            {
                "title": "请求 QPS（按接口）",
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                "targets": [{
                    "expr": 'rate(agent_requests_total[1m])',
                    "legendFormat": "{{endpoint}} - {{status}}",
                }],
            },
            {
                "title": "P50/P90/P99 延迟（秒）",
                "type": "graph",
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                "targets": [
                    {
                        "expr": 'histogram_quantile(0.50, rate(agent_request_duration_seconds_bucket[5m]))',
                        "legendFormat": "P50",
                    },
                    {
                        "expr": 'histogram_quantile(0.90, rate(agent_request_duration_seconds_bucket[5m]))',
                        "legendFormat": "P90",
                    },
                    {
                        "expr": 'histogram_quantile(0.99, rate(agent_request_duration_seconds_bucket[5m]))',
                        "legendFormat": "P99",
                    },
                ],
            },
            {
                "title": "错误率（5xx）",
                "type": "singlestat",
                "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8},
                "targets": [{
                    "expr": 'rate(agent_requests_total{status=~"5.."}[5m]) / rate(agent_requests_total[5m]) * 100',
                    "legendFormat": "错误率 %",
                }],
            },
            {
                "title": "异步任务入队总量",
                "type": "singlestat",
                "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8},
                "targets": [{
                    "expr": 'agent_tasks_enqueued_total',
                    "legendFormat": "累计入队",
                }],
            },
        ],
        "schemaVersion": 39,
    },
    "overwrite": True,
    "folderId": 0,
}

resp = httpx.post(
    f"{GRAFANA_URL}/api/dashboards/db",
    json=dashboard_config,
    auth=AUTH,
    timeout=10,
)
print(f"Dashboard 导入结果：{resp.status_code} - {resp.json()}")