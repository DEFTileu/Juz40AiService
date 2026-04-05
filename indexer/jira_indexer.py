import logging

import chromadb
import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from requests.auth import HTTPBasicAuth

from app.config import Settings

logger = logging.getLogger(__name__)

_JQL = "project = DV AND issuetype in (Story, Epic) AND description is not EMPTY ORDER BY created DESC"
_FIELDS = "summary,description,issuetype"
_PAGE_SIZE = 100
_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 150


# ── ADF → plain text ───────────────────────────────────────────────────────────

def parse_adf_to_text(description: dict) -> str:
    """
    Рекурсивно обходит Atlassian Document Format и собирает текстовые ноды.
    Возвращает plain text.
    """
    if not description or not isinstance(description, dict):
        return ""

    parts: list[str] = []

    node_type = description.get("type", "")
    text = description.get("text")
    if text and isinstance(text, str):
        parts.append(text)

    # Разделитель после параграфов и заголовков для читаемости
    for child in description.get("content", []):
        child_text = parse_adf_to_text(child)
        if child_text:
            parts.append(child_text)
        if node_type in ("paragraph", "heading", "bulletList", "orderedList"):
            parts.append("\n")

    return "".join(parts).strip()


# ── Jira API ───────────────────────────────────────────────────────────────────

def get_user_stories(settings: Settings) -> list[dict]:
    """
    Возвращает все Story и Epic из проекта DV с непустым description.
    Каждый элемент: {key, summary, description_text, issuetype}
    """
    session = requests.Session()
    session.auth = HTTPBasicAuth(
        settings.jira_email,
        settings.jira_token.get_secret_value(),
    )
    session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})

    base_url = f"{settings.jira_url}/rest/api/3/search/jql"
    stories: list[dict] = []
    next_page_token: str | None = None
    fetched = 0

    while True:
        body: dict = {
            "jql": _JQL,
            "fields": _FIELDS.split(","),
            "maxResults": _PAGE_SIZE,
        }
        if next_page_token:
            body["nextPageToken"] = next_page_token

        response = session.post(base_url, json=body, timeout=20)
        response.raise_for_status()
        data = response.json()

        issues = data.get("issues", [])
        for issue in issues:
            key: str = issue["key"]
            fields = issue.get("fields", {})
            summary: str = fields.get("summary", "").strip()
            description_raw = fields.get("description")
            issuetype: str = fields.get("issuetype", {}).get("name", "Story")

            description_text = parse_adf_to_text(description_raw) if description_raw else ""
            if not description_text.strip():
                logger.debug("  пропускаем %s — пустое description после парсинга", key)
                continue

            stories.append({
                "key": key,
                "summary": summary,
                "description_text": description_text,
                "issuetype": issuetype,
            })

        fetched += len(issues)
        next_page_token = data.get("nextPageToken")
        logger.info("get_user_stories: загружено %d (на странице %d)", fetched, len(issues))

        if not issues or not next_page_token:
            break

    logger.info("get_user_stories: итого %d записей", len(stories))
    return stories


# ── Главная функция индексации ─────────────────────────────────────────────────

def index_jira_stories(settings: Settings) -> dict[str, int]:
    """
    Загружает Jira Story и Epic в ChromaDB коллекцию docs_collection.

    Добавляет к существующей коллекции (Confluence уже должен быть загружен).
    Не пересоздаёт коллекцию.

    Returns:
        {"stories_indexed": N, "chunks_created": K}
    """
    if not settings.jira_url:
        raise ValueError("JIRA_URL не задан в .env")

    stories = get_user_stories(settings)

    if not stories:
        logger.warning("Нет stories для индексации")
        return {"stories_indexed": 0, "chunks_created": 0}

    documents: list[Document] = []
    for story in stories:
        content = f"{story['summary']}\n\n{story['description_text']}"
        url = f"{settings.jira_url}/browse/{story['key']}"

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": story["key"],
                    "url": url,
                    "type": "jira",
                    "issuetype": story["issuetype"],
                },
            )
        )

    logger.info("Подготовлено документов: %d", len(documents))

    # --- Нарезка на чанки ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Нарезка: %d stories → %d чанков", len(documents), len(chunks))

    # --- Embeddings ---
    from app.indexer.embedder import _build_embeddings
    embeddings = _build_embeddings(settings)

    # --- ChromaDB: добавляем к существующей коллекции ---
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)

    store = Chroma(
        client=client,
        collection_name=settings.docs_collection,
        embedding_function=embeddings,
    )

    # --- Батчевая вставка ---
    batch_size = 50
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        store.add_documents(batch)
        logger.info("  вставлено %d/%d чанков", min(i + batch_size, total), total)

    logger.info(
        "Индексация Jira завершена: stories=%d chunks=%d",
        len(documents),
        total,
    )
    return {"stories_indexed": len(documents), "chunks_created": total}
