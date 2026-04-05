import json
import logging

import requests
from langchain.tools import BaseTool

from app.agentic.tools.gitlab_read import _gitlab_base_url, _gitlab_headers

logger = logging.getLogger(__name__)

_TARGET_BRANCH = "develop"


class GitLabCreateMRTool(BaseTool):
    name: str = "gitlab_create_mr"
    description: str = (
        "Создаёт Merge Request в ветку develop. "
        'Input JSON: {"source_branch": "fix/DV-456-auth-npe", "jira_key": "DV-456"}'
    )

    def _run(self, tool_input: str) -> str:
        try:
            data = json.loads(tool_input)
            source_branch: str = data["source_branch"].strip()
            jira_key: str = data["jira_key"].strip()
        except (json.JSONDecodeError, KeyError) as exc:
            return f"Ошибка входных данных: {exc}. Ожидается JSON с полями: source_branch, jira_key."

        logger.info(
            "[WRITE][gitlab_create_mr] %s → %s jira_key=%s",
            source_branch,
            _TARGET_BRANCH,
            jira_key,
        )

        url = f"{_gitlab_base_url()}/merge_requests"
        payload = {
            "source_branch": source_branch,
            "target_branch": _TARGET_BRANCH,
            "title": jira_key,
            "description": f"{jira_key}\n🤖 AI Created",
        }

        try:
            response = requests.post(
                url,
                headers=_gitlab_headers(),
                json=payload,
                timeout=15,
            )
        except requests.RequestException as exc:
            logger.error("[WRITE][gitlab_create_mr] request error: %s", exc)
            return f"Ошибка создания MR: {exc}"

        if not response.ok:
            logger.error(
                "[WRITE][gitlab_create_mr] status=%d body=%s",
                response.status_code,
                response.text[:200],
            )
            return f"Ошибка создания MR: HTTP {response.status_code}"

        mr = response.json()
        iid: int = mr["iid"]
        web_url: str = mr["web_url"]

        logger.info("[WRITE][gitlab_create_mr] created: !%d %s", iid, web_url)
        return f"MR создан: !{iid} → {_TARGET_BRANCH} | {web_url}"


class GitLabCommentMRTool(BaseTool):
    name: str = "gitlab_comment_mr"
    description: str = (
        "Добавляет комментарий к Merge Request. "
        'Input JSON: {"mr_iid": 142, "comment": "текст комментария"}'
    )

    def _run(self, tool_input: str) -> str:
        try:
            data = json.loads(tool_input)
            mr_iid: int = int(data["mr_iid"])
            comment: str = data["comment"].strip()
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            return f"Ошибка входных данных: {exc}. Ожидается JSON с полями: mr_iid, comment."

        logger.info("[gitlab_comment_mr] mr_iid=%d", mr_iid)

        url = f"{_gitlab_base_url()}/merge_requests/{mr_iid}/notes"

        try:
            response = requests.post(
                url,
                headers=_gitlab_headers(),
                json={"body": comment},
                timeout=15,
            )
        except requests.RequestException as exc:
            logger.error("[gitlab_comment_mr] request error: %s", exc)
            return f"Ошибка добавления комментария: {exc}"

        if not response.ok:
            logger.error(
                "[gitlab_comment_mr] status=%d body=%s",
                response.status_code,
                response.text[:200],
            )
            return f"Ошибка добавления комментария: HTTP {response.status_code}"

        logger.info("[gitlab_comment_mr] ok: !%d", mr_iid)
        return f"Комментарий добавлен к MR !{mr_iid}"


class GitLabGetMRTool(BaseTool):
    name: str = "gitlab_get_mr"
    description: str = (
        "Получает информацию о Merge Request по номеру. "
        "Input: номер MR (число)"
    )

    def _run(self, tool_input: str) -> str:
        try:
            mr_iid = int(tool_input.strip())
        except ValueError as exc:
            return f"Ошибка входных данных: {exc}. Ожидается номер MR (число)."

        logger.info("[gitlab_get_mr] mr_iid=%d", mr_iid)

        url = f"{_gitlab_base_url()}/merge_requests/{mr_iid}"

        try:
            response = requests.get(
                url,
                headers=_gitlab_headers(),
                timeout=15,
            )
        except requests.RequestException as exc:
            logger.error("[gitlab_get_mr] request error: %s", exc)
            return f"Ошибка получения MR: {exc}"

        if response.status_code == 404:
            return f"MR !{mr_iid} не найден"

        if not response.ok:
            logger.error(
                "[gitlab_get_mr] status=%d body=%s",
                response.status_code,
                response.text[:200],
            )
            return f"Ошибка получения MR: HTTP {response.status_code}"

        mr = response.json()
        logger.info("[gitlab_get_mr] ok: !%d state=%s", mr_iid, mr.get("state"))

        return (
            f"MR !{mr_iid}\n"
            f"  title:         {mr.get('title')}\n"
            f"  state:         {mr.get('state')}\n"
            f"  source_branch: {mr.get('source_branch')}\n"
            f"  web_url:       {mr.get('web_url')}"
        )
