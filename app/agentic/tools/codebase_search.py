import logging
from pathlib import Path

from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

_PREVIEW_LEN = 300


class CodebaseSearchTool(BaseTool):
    name: str = "codebase_search"
    description: str = (
        "Семантический поиск по проиндексированному коду проекта в ChromaDB. "
        "Используй ПЕРВЫМ когда не знаешь в каком файле искать. "
        "Input: текстовый запрос например 'авторизация JWT' или 'создание пользователя регистрация'"
    )

    def _run(self, query: str) -> str:
        query = query.strip()
        logger.info("[codebase_search] query='%s'", query[:80])

        try:
            from app.agent.retriever import CodeRetriever
            retriever = CodeRetriever()
            chunks = retriever.search(query)
        except Exception as exc:
            logger.error("[codebase_search] ошибка подключения к ChromaDB: %s", exc)
            return (
                f"Не удалось подключиться к ChromaDB: {exc}\n"
                "Убедитесь что ChromaDB запущен (docker compose up -d) и код проиндексирован."
            )

        if not chunks:
            logger.info("[codebase_search] ничего не найдено для '%s'", query[:60])
            return f"По запросу '{query}' ничего не найдено в индексе."

        parts: list[str] = []
        for chunk in chunks:
            filename = Path(str(chunk["source"])).name
            preview = str(chunk["content"])[:_PREVIEW_LEN].strip()
            if len(str(chunk["content"])) > _PREVIEW_LEN:
                preview += "..."
            parts.append(f"📄 {filename}\n{preview}\n---")

        logger.info("[codebase_search] найдено %d чанков для '%s'", len(chunks), query[:60])
        return "\n".join(parts)
