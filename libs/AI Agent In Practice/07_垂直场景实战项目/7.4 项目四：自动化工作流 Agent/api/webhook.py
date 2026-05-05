from fastapi import Request, HTTPException
import base64
import json


async def gmail_push_webhook(request: Request):
    body = await request.json()

    pubsub_data = body.get("message", {}).get("data", "")
    if not pubsub_data:
        raise HTTPException(status_code=400, detail="No Pub/Sub data")

    _notification = json.loads(base64.urlsafe_b64decode(pubsub_data + "=="))

    from scheduler.tasks import poll_gmail_and_dispatch
    poll_gmail_and_dispatch.delay()

    return {"status": "ok"}
