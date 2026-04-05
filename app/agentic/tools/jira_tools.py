import json
import logging

import requests
from langchain.tools import BaseTool
from requests.auth import HTTPBasicAuth

from app.config import get_settings

logger = logging.getLogger(__name__)


def _jira_auth() -> HTTPBasicAuth:
    s = get_settings()
    return HTTPBasicAuth(s.jira_email, s.jira_token.get_secret_value())


def _jira_base_url() -> str:
    return f"{get_settings().jira_url}/rest/api/3"


def _jira_headers() -> dict[str, str]:
    return {"Content-Type": "application/json", "Accept": "application/json"}


def _get_active_sprint_id() -> int | None:
    """Возвращает ID активного спринта для board из settings. None если не найден или не настроен."""
    s = get_settings()
    if not s.jira_board_id:
        return None

    url = f"{s.jira_url}/rest/agile/1.0/board/{s.jira_board_id}/sprint"
    try:
        response = requests.get(
            url,
            headers=_jira_headers(),
            auth=_jira_auth(),
            params={"state": "active"},
            timeout=10,
        )
        if not response.ok:
            logger.warning("[jira] не удалось получить активный спринт: HTTP %d", response.status_code)
            return None
        values = response.json().get("values", [])
        if values:
            sprint_id: int = values[0]["id"]
            logger.info("[jira] активный спринт: id=%d name=%s", sprint_id, values[0].get("name"))
            return sprint_id
    except Exception as exc:
        logger.warning("[jira] ошибка получения спринта: %s", exc)
    return None


def _adf(text: str) -> dict:
    """Оборачивает plain text в Atlassian Document Format."""
    return {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [{"type": "text", "text": text}],
            }
        ],
    }


class JiraCreateTaskTool(BaseTool):
    name: str = "jira_create_task"
    description: str = (
        "Создаёт Bug таск в Jira проекте DV. Название автоматически получает префикс [AI]. "
        'Input JSON: {"summary": "NPE in AuthService line 45", '
        '"description": "описание проблемы", '
        '"assignee_account_id": "712020:xxx" или null}'
    )

    def _run(self, tool_input: str) -> str:
        try:
            data = json.loads(tool_input)
            summary: str = data["summary"].strip()
            description: str = data.get("description", "").strip()
            assignee_id: str | None = data.get("assignee_account_id") or None
        except (json.JSONDecodeError, KeyError) as exc:
            return f"Ошибка входных данных: {exc}. Ожидается JSON с полями: summary, description, assignee_account_id."

        settings = get_settings()
        full_summary = f"[AI] {summary}"

        logger.info("[jira_create_task] summary='%s' assignee=%s", full_summary, assignee_id)

        fields: dict = {
            "project": {"key": settings.jira_project_key},
            "issuetype": {"name": "Bug"},
            "summary": full_summary,
            "description": _adf(description) if description else _adf("Создано AI агентом."),
        }
        if assignee_id:
            fields["assignee"] = {"accountId": assignee_id}

        sprint_id = _get_active_sprint_id()
        if sprint_id:
            fields[settings.jira_sprint_field_id] = {"id": sprint_id}

        try:
            response = requests.post(
                f"{_jira_base_url()}/issue",
                headers=_jira_headers(),
                auth=_jira_auth(),
                json={"fields": fields},
                timeout=15,
            )
        except requests.RequestException as exc:
            logger.error("[jira_create_task] request error: %s", exc)
            return f"Ошибка создания таска: {exc}"

        if not response.ok:
            logger.error(
                "[jira_create_task] status=%d body=%s",
                response.status_code,
                response.text[:300],
            )
            return f"Ошибка создания таска: HTTP {response.status_code} — {response.text[:200]}"

        result = response.json()
        key: str = result["key"]
        url = f"{settings.jira_url}/browse/{key}"

        logger.info("[jira_create_task] created: %s %s", key, url)
        return f"Таск создан: {key} | {url}"


class JiraGetTaskTool(BaseTool):
    name: str = "jira_get_task"
    description: str = (
        "Получает информацию о Jira таске по ключу. "
        "Input: ключ таска например DV-456"
    )

    def _run(self, tool_input: str) -> str:
        key = tool_input.strip().upper()
        logger.info("[jira_get_task] key=%s", key)

        try:
            response = requests.get(
                f"{_jira_base_url()}/issue/{key}",
                headers=_jira_headers(),
                auth=_jira_auth(),
                timeout=15,
            )
        except requests.RequestException as exc:
            logger.error("[jira_get_task] request error: %s", exc)
            return f"Ошибка получения таска: {exc}"

        if response.status_code == 404:
            return f"Таск не найден: {key}"

        if not response.ok:
            logger.error(
                "[jira_get_task] status=%d body=%s",
                response.status_code,
                response.text[:200],
            )
            return f"Ошибка получения таска: HTTP {response.status_code}"

        issue = response.json()
        fields = issue.get("fields", {})

        summary: str = fields.get("summary", "—")
        status: str = fields.get("status", {}).get("name", "—")
        assignee_field = fields.get("assignee") or {}
        assignee: str = assignee_field.get("displayName", "не назначен")
        url = f"{get_settings().jira_url}/browse/{key}"

        logger.info("[jira_get_task] ok: %s status=%s", key, status)

        return (
            f"Таск {key}\n"
            f"  summary:  {summary}\n"
            f"  status:   {status}\n"
            f"  assignee: {assignee}\n"
            f"  url:      {url}"
        )
