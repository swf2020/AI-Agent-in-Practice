# conftest.py — set env vars before any module-level Settings() is evaluated
import os
import sys

# Ensure project root is on sys.path before first import
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

REQUIRED_ENV_VARS = {
    "ANTHROPIC_API_KEY": "sk-test-key",
    "GMAIL_CLIENT_ID": "test-client-id",
    "GMAIL_CLIENT_SECRET": "test-client-secret",
    "GMAIL_REFRESH_TOKEN": "test-refresh-token",
    "GMAIL_USER_EMAIL": "test@test.com",
    "SLACK_BOT_TOKEN": "xoxb-test-token",
    "SLACK_SIGNING_SECRET": "test-signing-secret",
    "SLACK_APPROVAL_CHANNEL": "C123",
    "SLACK_NOTIFY_CHANNEL": "C456",
    "REDIS_URL": "redis://localhost:6379/0",
}

for key, value in REQUIRED_ENV_VARS.items():
    os.environ.setdefault(key, value)
