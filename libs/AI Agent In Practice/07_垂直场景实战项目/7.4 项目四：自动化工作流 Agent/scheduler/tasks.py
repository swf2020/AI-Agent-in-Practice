import logging
from celery import Celery
from celery.schedules import crontab
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

from agent.workflow_graph import build_workflow_graph
from models import WorkflowState
from config import settings

logger = logging.getLogger(__name__)

app = Celery(
    "workflow",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_soft_time_limit=600,
    task_time_limit=660,
)

app.conf.beat_schedule = {
    "poll-gmail-every-5min": {
        "task": "scheduler.tasks.poll_gmail_and_dispatch",
        "schedule": crontab(minute="*/5"),
    }
}


def _fetch_unprocessed_emails() -> list[str]:
    creds = Credentials(
        token=None,
        refresh_token=settings.gmail_refresh_token,
        client_id=settings.gmail_client_id,
        client_secret=settings.gmail_client_secret,
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/gmail.readonly"],
    )
    creds.refresh(Request())
    service = build("gmail", "v1", credentials=creds)

    result = service.users().messages().list(
        userId="me",
        q="is:unread -label:WORKFLOW_PROCESSED",
        maxResults=10,
    ).execute()

    return [m["id"] for m in result.get("messages", [])]


@app.task(
    name="scheduler.tasks.process_single_email",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def process_single_email(self, email_id: str) -> dict:
    try:
        graph = build_workflow_graph(settings.redis_url)
        initial_state = WorkflowState(email_id=email_id)

        config = {"configurable": {"thread_id": email_id}}

        result = graph.invoke(initial_state, config=config)
        logger.info("邮件 %s 处理完成: %s", email_id, result)
        return {"status": "done", "email_id": email_id}

    except Exception as exc:
        logger.error("邮件 %s 处理失败: %s", email_id, exc, exc_info=True)
        raise self.retry(exc=exc)


@app.task(name="scheduler.tasks.poll_gmail_and_dispatch")
def poll_gmail_and_dispatch() -> dict:
    email_ids = _fetch_unprocessed_emails()
    for email_id in email_ids:
        process_single_email.delay(email_id)
    logger.info("本次轮询共分发 %d 封邮件", len(email_ids))
    return {"dispatched": len(email_ids)}
