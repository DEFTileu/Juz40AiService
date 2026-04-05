import logging
from urllib.parse import quote

import requests
from langchain.tools import BaseTool

from app.config import get_settings

logger = logging.getLogger(__name__)


def _gitlab_headers() -> dict[str, str]:
    return {"PRIVATE-TOKEN": get_settings().gitlab_token.get_secret_value()}


def _gitlab_base_url() -> str:
    s = get_settings()
    return f"{s.gitlab_url}/api/v4/projects/{s.gitlab_project_id}"


def _ref() -> str:
    return get_settings().gitlab_default_branch


class GitLabReadFileTool(BaseTool):
    name: str = "gitlab_read_file"
    description: str = (
        "Читает содержимое файла из GitLab репозитория. "
        "Input: путь к файлу например src/main/java/kz/juz40/auth/AuthService.java"
    )

    def _run(self, path: str) -> str:
        path = path.strip()
        logger.info("[gitlab_read_file] path=%s", path)

        encoded = quote(path, safe="")
        url = f"{_gitlab_base_url()}/repository/files/{encoded}/raw"

        try:
            response = requests.get(
                url,
                headers=_gitlab_headers(),
                params={"ref": _ref()},
                timeout=15,
            )
        except requests.RequestException as exc:
            logger.error("[gitlab_read_file] request error: %s", exc)
            return f"Ошибка чтения файла: {exc}"

        if response.status_code == 404:
            logger.warning("[gitlab_read_file] not found: %s", path)
            return f"Файл не найден: {path}"

        if not response.ok:
            logger.error("[gitlab_read_file] status=%d path=%s", response.status_code, path)
            return f"Ошибка чтения файла: HTTP {response.status_code}"

        content = response.text
        logger.info("[gitlab_read_file] ok: %s (%d chars)", path, len(content))
        return content


class GitLabListFilesTool(BaseTool):
    name: str = "gitlab_list_files"
    description: str = (
        "Показывает список файлов и папок в директории репозитория. "
        "Input: путь к директории например src/main/java/kz/juz40/auth/"
    )

    def _run(self, path: str) -> str:
        path = path.strip().rstrip("/")
        logger.info("[gitlab_list_files] path=%s", path)

        url = f"{_gitlab_base_url()}/repository/tree"

        try:
            response = requests.get(
                url,
                headers=_gitlab_headers(),
                params={"path": path, "ref": _ref(), "per_page": 100},
                timeout=15,
            )
        except requests.RequestException as exc:
            logger.error("[gitlab_list_files] request error: %s", exc)
            return f"Ошибка получения списка файлов: {exc}"

        if response.status_code == 404:
            logger.warning("[gitlab_list_files] not found: %s", path)
            return f"Директория не найдена: {path}"

        if not response.ok:
            logger.error("[gitlab_list_files] status=%d path=%s", response.status_code, path)
            return f"Ошибка получения списка файлов: HTTP {response.status_code}"

        items: list[dict] = response.json()
        if not items:
            return f"Директория пуста: {path}"

        dirs = sorted(
            item["name"] + "/" for item in items if item["type"] == "tree"
        )
        files = sorted(
            item["name"] for item in items if item["type"] == "blob"
        )

        lines = [f"{path or '/'}:"] + [f"  {d}" for d in dirs] + [f"  {f}" for f in files]
        result = "\n".join(lines)

        logger.info("[gitlab_list_files] ok: %s (%d items)", path, len(items))
        return result
