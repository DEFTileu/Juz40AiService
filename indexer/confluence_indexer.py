import logging
import re
from html.parser import HTMLParser

import chromadb
import requests
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from requests.auth import HTTPBasicAuth

from app.config import Settings

logger = logging.getLogger(__name__)

EXCLUDE_KEYWORDS = [
    "тест кейс",
    "test case",
    "тест-кейс",
    "тестовые случаи",
    "test cases",
]

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 150


# ── HTML → plain text ──────────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(html: str) -> str:
    """Парсит HTML и возвращает чистый текст без тегов."""
    stripper = _HTMLStripper()
    stripper.feed(html)
    text = stripper.get_text()
    # Сворачиваем множественные пробелы и пустые строки
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()


# ── Фильтрация страниц ─────────────────────────────────────────────────────────

def should_skip_page(title: str) -> bool:
    """True если заголовок страницы содержит любое из EXCLUDE_KEYWORDS (без учёта регистра)."""
    lower = title.lower()
    return any(kw in lower for kw in EXCLUDE_KEYWORDS)


# ── Confluence API ─────────────────────────────────────────────────────────────

def get_all_spaces(session: requests.Session, base_url: str) -> list[dict]:
    """Возвращает список всех пространств {key, name} с пагинацией."""
    spaces: list[dict] = []
    start = 0
    limit = 50

    while True:
        response = session.get(
            f"{base_url}/wiki/rest/api/space",
            params={"limit": limit, "start": start},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        spaces.extend({"key": s["key"], "name": s["name"]} for s in results)

        if data.get("size", 0) < limit:
            break
        start += limit

    logger.info("get_all_spaces: найдено %d пространств", len(spaces))
    return spaces


def get_pages_in_space(
    session: requests.Session,
    base_url: str,
    space_key: str,
) -> list[dict]:
    """
    Возвращает страницы пространства {id, title} с пагинацией.
    Страницы попадающие под EXCLUDE_KEYWORDS пропускаются.
    """
    pages: list[dict] = []
    skipped = 0
    start = 0
    limit = 100

    while True:
        response = session.get(
            f"{base_url}/wiki/rest/api/space/{space_key}/content/page",
            params={"limit": limit, "start": start, "expand": "title"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        for page in results:
            title: str = page.get("title", "")
            if should_skip_page(title):
                logger.debug("  пропускаем [тест кейс]: '%s'", title)
                skipped += 1
            else:
                pages.append({"id": page["id"], "title": title})

        if data.get("size", 0) < limit:
            break
        start += limit

    logger.info(
        "get_pages_in_space [%s]: страниц=%d пропущено=%d",
        space_key,
        len(pages),
        skipped,
    )
    return pages


def get_page_content(
    session: requests.Session,
    base_url: str,
    page_id: str,
) -> str:
    """Возвращает чистый текст страницы Confluence."""
    response = session.get(
        f"{base_url}/wiki/rest/api/content/{page_id}",
        params={"expand": "body.storage"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    html = data.get("body", {}).get("storage", {}).get("value", "")
    return _strip_html(html)


# ── Главная функция индексации ─────────────────────────────────────────────────

def index_confluence(settings: Settings) -> dict[str, int]:
    """
    Читает все страницы Confluence и загружает их в ChromaDB коллекцию docs_collection.

    Идемпотентно: коллекция пересоздаётся при каждом запуске.

    Returns:
        {"pages_indexed": N, "pages_skipped": M, "chunks_created": K}
    """
    base_url = settings.confluence_url.rstrip("/")
    if not base_url:
        raise ValueError("CONFLUENCE_URL не задан в .env")

    session = requests.Session()
    session.auth = HTTPBasicAuth(
        settings.confluence_email,
        settings.confluence_token.get_secret_value(),
    )
    session.headers.update({"Accept": "application/json"})

    # --- Собираем все страницы ---
    spaces = get_all_spaces(session, base_url)

    all_documents: list[Document] = []
    pages_skipped = 0

    _EXCLUDED_SPACES: frozenset[str] = frozenset({"Juz40"})

    for space in spaces:
        space_key: str = space["key"]

        if space_key in _EXCLUDED_SPACES:
            logger.info("Пропускаем пространство (исключено): %s (%s)", space["name"], space_key)
            continue

        logger.info("Обрабатываем пространство: %s (%s)", space["name"], space_key)

        try:
            pages = get_pages_in_space(session, base_url, space_key)
        except requests.RequestException as exc:
            logger.warning("Не удалось получить страницы [%s]: %s", space_key, exc)
            continue

        # Считаем пропущенные — разница между raw и отфильтрованными
        # (get_pages_in_space уже фильтрует, но skipped считается внутри)
        for i, page in enumerate(pages):
            page_id: str = page["id"]
            title: str = page["title"]

            try:
                content = get_page_content(session, base_url, page_id)
            except requests.RequestException as exc:
                logger.warning("  ошибка чтения страницы '%s' (%s): %s", title, page_id, exc)
                pages_skipped += 1
                continue

            if not content.strip():
                logger.debug("  пустая страница: '%s'", title)
                pages_skipped += 1
                continue

            page_url = f"{base_url}/wiki/spaces/{space_key}/pages/{page_id}"

            all_documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": title,
                        "url": page_url,
                        "type": "confluence",
                        "space": space_key,
                    },
                )
            )

            if (i + 1) % 10 == 0:
                logger.info("  обработано %d/%d страниц в [%s]", i + 1, len(pages), space_key)

    logger.info("Всего страниц для индексации: %d", len(all_documents))

    if not all_documents:
        logger.warning("Нет страниц для индексации — коллекция не будет создана")
        return {"pages_indexed": 0, "pages_skipped": pages_skipped, "chunks_created": 0}

    # --- Нарезка на чанки ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_documents)
    logger.info("Нарезка: %d страниц → %d чанков", len(all_documents), len(chunks))

    # --- Embeddings ---
    from app.indexer.embedder import _build_embeddings
    embeddings = _build_embeddings(settings)

    # --- ChromaDB: пересоздаём коллекцию ---
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)

    existing = [c.name for c in client.list_collections()]
    if settings.docs_collection in existing:
        client.delete_collection(settings.docs_collection)
        logger.info("Коллекция '%s' удалена для переиндексации", settings.docs_collection)

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
        "Индексация Confluence завершена: pages=%d skipped=%d chunks=%d",
        len(all_documents),
        pages_skipped,
        total,
    )
    return {
        "pages_indexed": len(all_documents),
        "pages_skipped": pages_skipped,
        "chunks_created": total,
    }
