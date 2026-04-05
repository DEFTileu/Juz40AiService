import logging

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.agent.chain import _build_llm
from app.agent.retriever import CodeRetriever, _build_embeddings
from app.biz.prompts import BIZ_PROMPT_TEMPLATE, get_role_hint
from app.config import get_settings

logger = logging.getLogger(__name__)

_DOCS_K = 6
_CODE_K = 3
_SCORE_THRESHOLD = 0.8   # L2-дистанция: меньше → релевантнее
_MIN_DOCS = 2             # если меньше — идём в код


class BizAgent:
    """
    Бизнес-консультант платформы juz40-edu.kz.

    Шаг 1 — поиск в juz40_docs (Confluence + Jira).
    Шаг 2 — если мало результатов, авто-поиск в juz40_codebase.
    Шаг 3 — ответ простым языком, без кода и технических деталей.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings

        # ── Docs retriever (juz40_docs) ────────────────────────────────────────
        embeddings = _build_embeddings(settings)
        client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        self._docs_store = Chroma(
            client=client,
            collection_name=settings.docs_collection,
            embedding_function=embeddings,
        )

        # ── Code retriever (juz40_codebase) — не дублируем логику ─────────────
        self._code_retriever = CodeRetriever(settings=settings)

        # ── LLM — берём провайдера из settings (gemini / openai) ──────────────
        self._llm = _build_llm(settings)

        logger.info(
            "BizAgent инициализирован: docs=%s, code=%s",
            settings.docs_collection,
            settings.chroma_collection,
        )

    # ── Вспомогательные методы ─────────────────────────────────────────────────

    def _format_docs_context(self, docs: list[Document]) -> str:
        parts: list[str] = []
        for doc in docs:
            meta = doc.metadata
            source = meta.get("source", "unknown")
            doc_type = meta.get("type", "")
            url = meta.get("url", "")

            if doc_type == "confluence":
                label = f"[Confluence: {source}]"
            elif doc_type == "jira":
                issuetype = meta.get("issuetype", "Story")
                label = f"[Jira {issuetype}: {source}]"
            else:
                label = f"[{source}]"

            if url:
                label += f" ({url})"

            parts.append(f"{label}\n{doc.page_content}")

        return "\n\n".join(parts)

    def _format_code_context(self, code_results: list[dict]) -> str:
        if not code_results:
            return ""
        parts = ["Из кода платформы (только факты, без деталей реализации):"]
        for chunk in code_results:
            source = chunk.get("source", "unknown")
            content = chunk.get("content", "")
            parts.append(f"[{source}]\n{content}")
        return "\n\n".join(parts)

    def _build_sources(
        self,
        docs: list[Document],
        code_results: list[dict],
    ) -> list[str]:
        seen: set[str] = set()
        sources: list[str] = []

        for doc in docs:
            meta = doc.metadata
            doc_type = meta.get("type", "")
            source = meta.get("source", "unknown")
            url = meta.get("url", "")

            if doc_type == "confluence":
                label = f"Confluence: {source}"
            elif doc_type == "jira":
                label = f"Jira: {source}"
            else:
                label = source

            key = url or label
            if key not in seen:
                seen.add(key)
                sources.append(label)

        for chunk in code_results:
            src = chunk.get("source", "")
            if src and src not in seen:
                seen.add(src)
                # Код — не показываем пользователю как источник напрямую,
                # но логируем факт использования
                logger.debug("code source (не в ответе): %s", src)

        return sources

    # ── Основной метод ─────────────────────────────────────────────────────────

    def ask(self, question: str, role: str = "") -> dict:
        """
        Args:
            question: вопрос пользователя
            role: student | curator | zavuch | methodist | parent | admin | ""

        Returns:
            {"answer": str, "sources": list[str], "used_code_search": bool}
        """
        logger.info("BizAgent.ask: role=%r  question='%s'", role or "—", question[:80])

        # ── Шаг 1: поиск в документации ───────────────────────────────────────
        raw_docs: list[tuple[Document, float]] = self._docs_store.similarity_search_with_score(
            query=question,
            k=_DOCS_K,
        )

        # similarity_search_with_score возвращает L2-дистанцию: меньше = лучше
        filtered_docs: list[Document] = [
            doc for doc, score in raw_docs if score < _SCORE_THRESHOLD
        ]

        logger.info(
            "docs поиск: всего=%d, релевантных (score<%.2f)=%d",
            len(raw_docs),
            _SCORE_THRESHOLD,
            len(filtered_docs),
        )

        # ── Шаг 2: нужен ли поиск по коду? ────────────────────────────────────
        used_code = False
        code_results: list[dict] = []

        if len(filtered_docs) < _MIN_DOCS:
            logger.info(
                "Мало docs (< %d) — включаем поиск по коду",
                _MIN_DOCS,
            )
            code_results = self._code_retriever.search(question)[:_CODE_K]
            used_code = True

        # ── Шаг 3: сборка контекста ────────────────────────────────────────────
        docs_context = self._format_docs_context(filtered_docs)
        code_context = self._format_code_context(code_results) if used_code else ""

        if docs_context and code_context:
            context = f"{docs_context}\n\n{code_context}"
        elif docs_context:
            context = docs_context
        elif code_context:
            context = code_context
        else:
            logger.info("BizAgent: контекст пустой — возвращаем fallback")
            return {
                "answer": (
                    "В документации нет информации по этому вопросу. "
                    "Обратитесь к вашему куратору или команде поддержки."
                ),
                "sources": [],
                "used_code_search": used_code,
            }

        # ── Шаг 4: генерация ответа ────────────────────────────────────────────
        messages = BIZ_PROMPT_TEMPLATE.format_messages(
            context=context,
            question=question,
            role=role.strip(),
            role_hint=get_role_hint(role),
        )

        answer: str = self._llm.invoke(messages).content
        logger.info("BizAgent.ask завершён: used_code=%s", used_code)

        # ── Шаг 5: источники ───────────────────────────────────────────────────
        sources = self._build_sources(filtered_docs, code_results)

        return {
            "answer": answer,
            "sources": sources,
            "used_code_search": used_code,
        }

    def get_collection_size(self) -> int:
        collection = self._docs_store._client.get_collection(
            self._settings.docs_collection
        )
        return collection.count()


# Singleton
biz_agent = BizAgent()
