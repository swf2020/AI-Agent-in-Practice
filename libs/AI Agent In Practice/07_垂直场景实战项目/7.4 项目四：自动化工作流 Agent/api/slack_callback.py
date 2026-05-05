from __future__ import annotations
import hashlib
import hmac
import json
import time
from fastapi import Request, HTTPException
from langgraph.types import Command

from agent.workflow_graph import build_workflow_graph
from config import settings


async def slack_interactions_handler(request: Request):
    await _verify_slack_signature(request)

    form_data = await request.form()
    payload = json.loads(form_data.get("payload", "{}"))

    if payload.get("type") != "block_actions":
        return {"ok": True}

    action = payload["actions"][0]
    action_id: str = action["action_id"]
    operator = payload["user"]["name"]

    if action_id.startswith("approve_"):
        run_id = action_id[len("approve_"):]
        approved = True
    elif action_id.startswith("reject_"):
        run_id = action_id[len("reject_"):]
        approved = False
    else:
        return {"ok": True}

    graph = build_workflow_graph(settings.redis_url)
    config = {"configurable": {"thread_id": run_id}}

    graph.invoke(
        Command(resume={"approved": approved, "operator": operator}),
        config=config,
    )

    return {"ok": True}


async def _verify_slack_signature(request: Request) -> None:
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")
    body = await request.body()

    if abs(time.time() - int(timestamp)) > 300:
        raise HTTPException(status_code=403, detail="Request too old")

    sig_basestring = f"v0:{timestamp}:{body.decode()}"
    expected_sig = "v0=" + hmac.new(
        settings.slack_signing_secret.encode(),
        sig_basestring.encode(),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected_sig, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")
