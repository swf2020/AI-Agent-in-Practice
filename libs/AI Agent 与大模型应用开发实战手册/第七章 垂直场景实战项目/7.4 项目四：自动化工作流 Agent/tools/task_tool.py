from __future__ import annotations
import json
import httpx
from langchain_core.tools import tool

from config import settings


@tool
def notion_create_task(
    title: str,
    description: str,
    assignee: str = "",
    due_date: str = "",
    priority: str = "medium",
) -> str:
    """在 Notion 数据库中创建一个新任务页面"""
    properties: dict = {
        "Name": {"title": [{"text": {"content": title}}]},
        "Description": {"rich_text": [{"text": {"content": description}}]},
        "Priority": {"select": {"name": priority.capitalize()}},
        "Status": {"select": {"name": "To Do"}},
    }
    if due_date:
        properties["Due Date"] = {"date": {"start": due_date}}
    if assignee:
        properties["Assignee"] = {"rich_text": [{"text": {"content": assignee}}]}

    with httpx.Client() as client:
        response = client.post(
            "https://api.notion.com/v1/pages",
            headers={
                "Authorization": f"Bearer {settings.notion_api_key}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json",
            },
            json={
                "parent": {"database_id": settings.notion_database_id},
                "properties": properties,
            },
            timeout=15,
        )
        response.raise_for_status()
        page_id = response.json()["id"]
        return page_id


@tool
def jira_create_issue(
    title: str,
    description: str,
    assignee: str = "",
    due_date: str = "",
    priority: str = "Medium",
) -> str:
    """在 Jira 项目中创建一个新 Issue"""
    fields: dict = {
        "project": {"key": settings.jira_project_key},
        "summary": title,
        "description": {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": description}],
                }
            ],
        },
        "issuetype": {"name": "Task"},
        "priority": {"name": priority.capitalize()},
    }
    if due_date:
        fields["duedate"] = due_date

    with httpx.Client() as client:
        response = client.post(
            f"{settings.jira_base_url}/rest/api/3/issue",
            auth=(settings.jira_email, settings.jira_api_token),
            json={"fields": fields},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()["key"]