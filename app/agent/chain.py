import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from app.agent.prompts import QA_PROMPT_TEMPLATE
from app.agent.retriever import CodeRetriever
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_GEMINI_CHAT_MODEL = "gemini-2.0-flash"
_OPENAI_CHAT_MODEL = "gpt-4o-mini"


def _build_llm(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        logger.info("LLM провайдер: OpenAI (%s)", _OPENAI_CHAT_MODEL)
        return ChatOpenAI(
            model=_OPENAI_CHAT_MODEL,
            openai_api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            max_tokens=2048,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI
    logger.info("LLM провайдер: Gemini (%s)", _GEMINI_CHAT_MODEL)
    return ChatGoogleGenerativeAI(
        model=_GEMINI_CHAT_MODEL,
        google_api_key=settings.gemini_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
        max_output_tokens=2048,
    )


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


def _deduplicate_sources(docs: list[Document]) -> list[str]:
    """Уникальные имена файлов-источников в порядке первого появления."""
    seen: set[str] = set()
    result: list[str] = []
    for doc in docs:
        path = doc.metadata.get("source", "")
        name = Path(path).name if path else "unknown"
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


class AiAgent:
    """
    RAG-агент: принимает вопрос → ищет контекст → генерирует ответ.

    Использует LCEL (LangChain Expression Language).
    Инициализируется один раз при старте приложения.
    """

    def __init__(self) -> None:
        settings = get_settings()

        llm = _build_llm(settings)
        self._retriever = CodeRetriever(settings=settings)
        lc_retriever = self._retriever.as_langchain_retriever()

        # LCEL цепочка: параллельно получаем контекст и прокидываем вопрос
        self._chain = (
            RunnableParallel(
                context=lc_retriever,
                question=RunnablePassthrough(),
            )
            | RunnableParallel(
                answer=(
                    RunnableLambda(lambda x: {"context": _format_docs(x["context"]), "question": x["question"]})
                    | QA_PROMPT_TEMPLATE
                    | llm
                    | StrOutputParser()
                ),
                source_documents=RunnableLambda(lambda x: x["context"]),
            )
        )

        logger.info(
            "AiAgent инициализирован: provider=%s, temperature=%s",
            settings.llm_provider,
            settings.llm_temperature,
        )

    def ask(self, question: str) -> dict[str, str | list[str]]:
        """
        Задаёт вопрос и возвращает ответ с источниками.

        Returns:
            {"answer": str, "sources": list[str]}
        """
        logger.info("ask: '%s'", question[:80])

        result = self._chain.invoke(question)

        for i, doc in enumerate(result.get("source_documents", []), 1):
            logger.debug(
                "  chunk %d | %s | score=? | preview: %s",
                i,
                doc.metadata.get("source", "?"),
                doc.page_content[:120].replace("\n", " "),
            )

        answer: str = result["answer"]
        sources = _deduplicate_sources(result["source_documents"])

        logger.info("ask завершён: sources=%s", sources)
        return {"answer": answer, "sources": sources}


# Singleton — инициализируется один раз при импорте модуля
agent = AiAgent()
