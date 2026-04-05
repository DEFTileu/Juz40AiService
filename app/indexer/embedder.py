import json
import logging
import time
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings

logger = logging.getLogger(__name__)

_GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-2-preview"
_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

_BATCH_SIZE = 50        # чанков за одну вставку в ChromaDB
_REQUEST_DELAY = 0.7    # секунд между батчами (free tier: 100 req/min)
_MAX_RETRIES = 5        # попыток при 429

# Файл прогресса — рядом с проектом
_CHECKPOINT_PATH = Path(__file__).parent.parent.parent / ".index_checkpoint.json"


# ── Checkpoint ─────────────────────────────────────────────────────────────────

def _save_checkpoint(collection: str, total_chunks: int, last_batch: int) -> None:
    _CHECKPOINT_PATH.write_text(
        json.dumps({
            "collection": collection,
            "total_chunks": total_chunks,
            "last_batch": last_batch,
        }),
        encoding="utf-8",
    )


def _load_checkpoint(collection: str, total_chunks: int) -> int:
    """
    Возвращает номер батча с которого продолжать (0 = начать сначала).
    Checkpoint считается валидным только если совпадают коллекция и кол-во чанков.
    """
    if not _CHECKPOINT_PATH.exists():
        return 0
    try:
        data = json.loads(_CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if data["collection"] == collection and data["total_chunks"] == total_chunks:
            last = data["last_batch"]
            logger.info("Найден checkpoint: продолжаем с батча %d", last + 1)
            return last
    except Exception:
        pass
    return 0


def _clear_checkpoint() -> None:
    if _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()


# ── Embeddings factory ─────────────────────────────────────────────────────────

def _build_embeddings(settings: Settings) -> Embeddings:
    if settings.llm_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        logger.info("Embedding провайдер: OpenAI (%s)", _OPENAI_EMBEDDING_MODEL)
        return OpenAIEmbeddings(
            model=_OPENAI_EMBEDDING_MODEL,
            openai_api_key=settings.openai_api_key,
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    logger.info("Embedding провайдер: Gemini (%s)", _GEMINI_EMBEDDING_MODEL)
    return GoogleGenerativeAIEmbeddings(
        model=_GEMINI_EMBEDDING_MODEL,
        google_api_key=settings.gemini_api_key.get_secret_value(),
    )


# ── Retry ──────────────────────────────────────────────────────────────────────

def _add_with_retry(store: Chroma, batch: list[Document], batch_num: int, total: int) -> None:
    """Вставляет батч в ChromaDB с retry при 429."""
    delay = 5.0
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            store.add_documents(batch)
            logger.info("  Батч %d/%d готов (%d чанков)", batch_num, total, len(batch))
            return
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                if attempt < _MAX_RETRIES:
                    logger.warning(
                        "  Батч %d/%d: rate limit, ждём %.0fs (попытка %d/%d)...",
                        batch_num, total, delay, attempt, _MAX_RETRIES,
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise RuntimeError(
                        f"Батч {batch_num}/{total}: превышен лимит попыток после rate limit"
                    ) from exc
            else:
                raise


# ── Main ───────────────────────────────────────────────────────────────────────

def embed_and_store(documents: list[Document], settings: Settings) -> dict[str, int]:
    """
    Нарезает документы на чанки, создаёт эмбеддинги и сохраняет в ChromaDB.

    Возобновляемо: при падении сохраняет прогресс в .index_checkpoint.json.
    При повторном запуске продолжает с места остановки.
    При успешном завершении checkpoint удаляется автоматически.

    Returns:
        {"files_indexed": int, "chunks_created": int}
    """
    if not documents:
        logger.warning("Список документов пуст — индексация пропущена")
        return {"files_indexed": 0, "chunks_created": 0}

    # --- Нарезка на чанки ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    total_chunks = len(chunks)
    logger.info("Нарезка завершена: %d документов → %d чанков", len(documents), total_chunks)

    # --- Checkpoint: определяем с какого батча продолжать ---
    batches = [chunks[i : i + _BATCH_SIZE] for i in range(0, total_chunks, _BATCH_SIZE)]
    total_batches = len(batches)
    resume_from = _load_checkpoint(settings.chroma_collection, total_chunks)
    is_resume = resume_from > 0

    # --- Инициализация эмбеддингов ---
    embeddings = _build_embeddings(settings)

    # --- Подключение к ChromaDB ---
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)

    if not is_resume:
        # Свежий старт — сбрасываем коллекцию
        existing = [c.name for c in client.list_collections()]
        if settings.chroma_collection in existing:
            client.delete_collection(settings.chroma_collection)
            logger.info("Коллекция '%s' удалена для переиндексации", settings.chroma_collection)
    else:
        logger.info(
            "Возобновление индексации: пропускаем батчи 1–%d, продолжаем с %d/%d",
            resume_from, resume_from + 1, total_batches,
        )

    store = Chroma(
        client=client,
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
    )

    # --- Батчевая вставка с checkpoint ---
    logger.info(
        "Сохраняем %d чанков батчами по %d → %d запросов...",
        total_chunks, _BATCH_SIZE, total_batches,
    )

    for i, batch in enumerate(batches, 1):
        if i <= resume_from:
            continue  # уже сохранено

        _add_with_retry(store, batch, i, total_batches)
        _save_checkpoint(settings.chroma_collection, total_chunks, i)

        if i < total_batches:
            time.sleep(_REQUEST_DELAY)

    # --- Готово: удаляем checkpoint ---
    _clear_checkpoint()

    unique_sources = {doc.metadata.get("source", "") for doc in documents}
    logger.info(
        "Индексация завершена: files=%d, chunks=%d",
        len(unique_sources),
        total_chunks,
    )
    return {"files_indexed": len(unique_sources), "chunks_created": total_chunks}
