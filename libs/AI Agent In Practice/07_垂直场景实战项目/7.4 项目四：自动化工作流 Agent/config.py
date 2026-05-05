from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str = ""

    gmail_client_id: str = ""
    gmail_client_secret: str = ""
    gmail_refresh_token: str = ""
    gmail_user_email: str = ""

    slack_bot_token: str = ""
    slack_signing_secret: str = ""
    slack_approval_channel: str = ""
    slack_notify_channel: str = ""

    notion_api_key: str = ""
    notion_database_id: str = ""

    jira_base_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    jira_project_key: str = ""

    redis_url: str = "redis://localhost:6379/0"


settings = Settings()
