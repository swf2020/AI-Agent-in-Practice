from fastapi import FastAPI, Request
from api.webhook import gmail_push_webhook
from api.slack_callback import slack_interactions_handler

app = FastAPI(title="Workflow Agent API")

app.post("/webhook/gmail")(gmail_push_webhook)
app.post("/slack/interactions")(slack_interactions_handler)


@app.get("/health")
async def health():
    return {"status": "ok"}
