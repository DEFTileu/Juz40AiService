import json
import logging
from urllib.parse import quote

import requests
from langchain.tools import BaseTool

from app.agentic.tools.gitlab_read import _gitlab_base_url, _gitlab_headers, _ref

logger = logging.getLogger(__name__)


def _file_exists(file_path: str, branch: str) -> bool:
    """HEAD запрос для проверки существования файла в ветке."""
    encoded = quote(file_path, safe="")
    url = f"{_gitlab_base_url()}/repository/files/{encoded}"
    try:
        response = requests.head(
            url,
            headers=_gitlab_headers(),
            params={"ref": branch},
            timeout=10,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


class GitLabCreateBranchTool(BaseTool):
    name: str = "gitlab_create_branch"
    description: str = (
        "Создаёт новую ветку в GitLab от ветки develop. "
        'Input JSON: {"branch_name": "fix/DV-456-auth-npe"}'
    )

    def _run(self, tool_input: str) -> str:
        try:
            data = json.loads(tool_input)
            branch_name: str = data["branch_name"].strip()
        except (json.JSONDecodeError, KeyError) as exc:
            return f"Ошибка входных данных: {exc}. Ожидается JSON с полем branch_name."

        logger.info("[WRITE][gitlab_create_branch] branch=%s from=%s", branch_name, _ref())

        url = f"{_gitlab_base_url()}/repository/branches"

        try:
            response = requests.post(
                url,
                headers=_gitlab_headers(),
                json={"branch": branch_name, "ref": _ref()},
                timeout=15,
            )
        except requests.RequestException as exc:
            logger.error("[WRITE][gitlab_create_branch] request error: %s", exc)
            return f"Ошибка создания ветки: {exc}"

        if response.status_code == 400:
            # GitLab возвращает 400 если ветка уже существует
            logger.warning("[WRITE][gitlab_create_branch] already exists: %s", branch_name)
            return f"Ветка уже существует: {branch_name}"

        if not response.ok:
            logger.error(
                "[WRITE][gitlab_create_branch] status=%d body=%s",
                response.status_code,
                response.text[:200],
            )
            return f"Ошибка создания ветки: HTTP {response.status_code}"

        logger.info("[WRITE][gitlab_create_branch] created: %s", branch_name)
        return f"Ветка создана: {branch_name} от {_ref()}"


class GitLabCommitTool(BaseTool):
    name: str = "gitlab_commit_file"
    description: str = (
        "Создаёт коммит с изменением файла. "
        "Commit message паттерн: {story}/{ticket}: {description}. "
        "Input JSON: {"
        '"branch": "fix/DV-456-auth-npe", '
        '"file_path": "src/.../AuthService.java", '
        '"content": "полный новый контент файла", '
        '"commit_message": "fix/DV-456: null check in AuthService line 45"'
        "}"
    )

    def _run(self, tool_input: str) -> str:
        try:
            data = json.loads(tool_input)
            branch: str = data["branch"].strip()
            file_path: str = data["file_path"].strip()
            content: str = data["content"]
            commit_message: str = data["commit_message"].strip()
        except (json.JSONDecodeError, KeyError) as exc:
            return (
                f"Ошибка входных данных: {exc}. "
                "Ожидается JSON с полями: branch, file_path, content, commit_message."
            )

        logger.info(
            "[WRITE][gitlab_commit_file] branch=%s file=%s message='%s'",
            branch,
            file_path,
            commit_message,
        )

        action = "update" if _file_exists(file_path, branch) else "create"
        logger.info("[WRITE][gitlab_commit_file] action=%s", action)

        url = f"{_gitlab_base_url()}/repository/commits"
        payload = {
            "branch": branch,
            "commit_message": commit_message,
            "actions": [
                {
                    "action": action,
                    "file_path": file_path,
                    "content": content,
                }
            ],
        }

        try:
            response = requests.post(
                url,
                headers=_gitlab_headers(),
                json=payload,
                timeout=30,
            )
        except requests.RequestException as exc:
            logger.error("[WRITE][gitlab_commit_file] request error: %s", exc)
            return f"Ошибка создания коммита: {exc}"

        if not response.ok:
            logger.error(
                "[WRITE][gitlab_commit_file] status=%d body=%s",
                response.status_code,
                response.text[:200],
            )
            return f"Ошибка создания коммита: HTTP {response.status_code}"

        logger.info("[WRITE][gitlab_commit_file] ok: branch=%s", branch)
        return f"Коммит создан в ветке {branch}: {commit_message}"
