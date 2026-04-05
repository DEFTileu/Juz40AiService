import logging

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-2-preview"
_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


def _build_embeddings(settings: Settings) -> Embeddings:
    if settings.llm_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=_OPENAI_EMBEDDING_MODEL,
            openai_api_key=settings.openai_api_key,
        )
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(
        model=_GEMINI_EMBEDDING_MODEL,
        google_api_key=settings.gemini_api_key.get_secret_value(),
    )


class CodeRetriever:
    """Поиск семантически похожих кусков кода в ChromaDB."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

        self._embeddings = _build_embeddings(self._settings)

        self._client = chromadb.HttpClient(
            host=self._settings.chroma_host,
            port=self._settings.chroma_port,
        )

        self._store = Chroma(
            client=self._client,
            collection_name=self._settings.chroma_collection,
            embedding_function=self._embeddings,
        )

        logger.info(
            "CodeRetriever инициализирован: collection=%s, top_k=%d",
            self._settings.chroma_collection,
            self._settings.top_k_results,
        )

    def search(self, question: str) -> list[dict[str, str | float]]:
        """
        Ищет top-K чанков наиболее похожих на вопрос.

        Returns:
            Список словарей с полями:
            - content (str): текст чанка
            - source  (str): путь к исходному файлу
            - score   (float): релевантность (0–1, выше — лучше)
        """
        results = self._store.similarity_search_with_relevance_scores(
            query=question,
            k=self._settings.top_k_results,
        )

        chunks: list[dict[str, str | float]] = []
        for doc, score in results:
            chunks.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "score": round(score, 4),
                }
            )

        logger.debug("search '%s': вернули %d чанков", question[:60], len(chunks))
        return chunks

    def get_collection_size(self) -> int:
        """Возвращает общее количество чанков в коллекции."""
        collection = self._client.get_collection(self._settings.chroma_collection)
        return collection.count()

    def as_langchain_retriever(self):  # type: ignore[return]
        """Возвращает LangChain-совместимый retriever для использования в цепочках."""
        return self._store.as_retriever(
            search_kwargs={"k": self._settings.top_k_results}
        )
