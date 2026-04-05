import logging
from urllib.parse import quote

import requests
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".java": "java",
    ".ts":   "typescript",
    ".tsx":  "typescript",
    ".sql":  "sql",
    ".md":   "markdown",
}

_INCLUDED_EXTENSIONS: frozenset[str] = frozenset(_EXTENSION_TO_LANGUAGE)

_EXCLUDED_PATH_PARTS: frozenset[str] = frozenset({
    "node_modules", "dist", "build", "target", "__pycache__",
    ".git", ".claude", "worktrees",
})

_EXCLUDED_SUFFIXES: tuple[str, ...] = (
    "Test.java",
    ".test.ts",
    ".test.tsx",
    ".spec.ts",
    ".spec.tsx",
)


def should_index_file(path: str) -> bool:
    """True если файл должен попасть в индекс."""
    parts = path.replace("\\", "/").split("/")

    # Исключаем по директориям
    for part in parts[:-1]:
        if part in _EXCLUDED_PATH_PARTS:
            return False

    # Проверяем расширение
    name = parts[-1]
    dot = name.rfind(".")
    if dot == -1:
        return False
    ext = name[dot:]
    if ext not in _INCLUDED_EXTENSIONS:
        return False

    # Исключаем тесты
    if any(name.endswith(s) for s in _EXCLUDED_SUFFIXES):
        return False

    return True


def load_from_gitlab(branch: str, project_id: str = "") -> list[Document]:
    """
    Читает файлы проекта из GitLab API и возвращает LangChain Documents.

    Args:
        branch:     ветка для чтения (например "master" или "develop")
        project_id: GitLab project ID или namespace/repo. Если пусто —
                    берётся gitlab_project_id из settings.

    Returns:
        Список Document: page_content=содержимое файла,
        metadata={"source": path, "branch": branch, "language": str}
    """
    from app.config import get_settings
    settings = get_settings()

    session = requests.Session()
    session.headers.update({
        "PRIVATE-TOKEN": settings.gitlab_token.get_secret_value(),
    })

    pid = quote(str(project_id or settings.gitlab_project_id), safe="")
    tree_url = f"{settings.gitlab_url}/api/v4/projects/{pid}/repository/tree"
    files_base = f"{settings.gitlab_url}/api/v4/projects/{pid}/repository/files"

    # ── Шаг 1: получаем полный список файлов через пагинацию ──────────────────
    all_items: list[dict] = []
    page = 1

    while True:
        response = session.get(
            tree_url,
            params={"recursive": "true", "ref": branch, "per_page": 100, "page": page},
            timeout=30,
        )
        response.raise_for_status()
        items = response.json()

        if not items:
            break

        all_items.extend(items)
        page += 1

        # GitLab отдаёт X-Next-Page пустым когда страниц больше нет
        if not response.headers.get("X-Next-Page"):
            break

    logger.info("GitLab tree [%s]: всего объектов %d", branch, len(all_items))

    # Оставляем только файлы (blob) которые нужно индексировать
    to_index = [
        item["path"]
        for item in all_items
        if item.get("type") == "blob" and should_index_file(item["path"])
    ]

    logger.info("Файлов для индексации: %d", len(to_index))

    # ── Шаг 2: читаем содержимое каждого файла ────────────────────────────────
    documents: list[Document] = []
    skipped = 0

    for i, path in enumerate(to_index, 1):
        encoded_path = quote(path, safe="")
        raw_url = f"{files_base}/{encoded_path}/raw"

        try:
            resp = session.get(raw_url, params={"ref": branch}, timeout=20)
            resp.raise_for_status()
            content = resp.text
        except requests.RequestException as exc:
            logger.warning("  Не удалось прочитать %s: %s", path, exc)
            skipped += 1
            continue

        if not content.strip():
            logger.debug("  Пустой файл, пропускаем: %s", path)
            skipped += 1
            continue

        dot = path.rfind(".")
        ext = path[dot:] if dot != -1 else ""

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": path,
                    "branch": branch,
                    "language": _EXTENSION_TO_LANGUAGE.get(ext, "unknown"),
                },
            )
        )

        if i % 20 == 0:
            logger.info("  прочитано %d/%d файлов...", i, len(to_index))

    logger.info(
        "Загрузка из GitLab [%s] завершена: документов=%d, пропущено=%d",
        branch,
        len(documents),
        skipped,
    )
    return documents
