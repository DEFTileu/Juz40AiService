import logging
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Корень проекта — две папки вверх от app/config.py
_PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API защита — обязательный токен для всех эндпойнтов
    api_key: SecretStr

    # LLM Provider
    llm_provider: str = "gemini"

    # Gemini — обязательный, нет дефолта → ошибка при старте если не задан
    gemini_api_key: SecretStr

    # OpenAI — опциональный
    openai_api_key: str = ""

    # ChromaDB — обязательные с разумными дефолтами
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "juz40_codebase"

    # Пути к коду — хотя бы один обязателен
    backend_code_path: str = ""
    frontend_code_path: str = ""

    # Параметры RAG агента — опциональные
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k_results: int = 5
    llm_temperature: float = 0.0

    # GitLab
    gitlab_url: str = "https://gitlab.com"
    gitlab_token: SecretStr = SecretStr("")
    gitlab_project_id: str = ""
    gitlab_default_branch: str = "develop"
    gitlab_index_branch: str = "master"

    # Jira
    jira_url: str = ""
    jira_email: str = ""
    jira_token: SecretStr = SecretStr("")
    jira_project_key: str = "DV"
    jira_board_id: int = 0                          # для получения активного спринта
    jira_sprint_field_id: str = "customfield_10020" # поле спринта в Jira

    # Agentic executor
    agent_max_iterations: int = 15
    agent_verbose: bool = True

    # Маппинг TG username → Jira accountId
    # Формат: "username1:jira_id_1,username2:jira_id_2"
    telegram_jira_mapping: str = ""

    # Confluence
    confluence_url: str = ""
    confluence_email: str = ""
    confluence_token: SecretStr = SecretStr("")
    confluence_space_key: str = ""

    # Docs ChromaDB индекс (этап 3)
    docs_collection: str = "juz40_docs"

    # Страницы Confluence с этими подстроками в названии пропускаются
    confluence_exclude_titles: str = "тест кейс,test case,тест-кейс,тестовые случаи"

    @field_validator("backend_code_path", "frontend_code_path")
    @classmethod
    def path_not_empty(cls, v: str, info) -> str:
        return v.strip()

    @field_validator("chunk_size", "chunk_overlap", "top_k_results")
    @classmethod
    def positive_int(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"{info.field_name.upper()} должен быть положительным числом, получено: {v}")
        return v

    @field_validator("llm_temperature")
    @classmethod
    def temperature_range(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"LLM_TEMPERATURE должен быть от 0.0 до 2.0, получено: {v}")
        return v


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    logger.info(
        "Settings loaded: provider=%s, chroma=%s:%d, collection=%s",
        settings.llm_provider,
        settings.chroma_host,
        settings.chroma_port,
        settings.chroma_collection,
    )
    return settings
