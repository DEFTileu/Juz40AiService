import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, field_validator

from app.agent.chain import agent
from app.agent.retriever import CodeRetriever
from app.agentic.executor import executor
from app.biz.biz_agent import biz_agent
from app.config import get_settings
from app.indexer.embedder import embed_and_store
from app.indexer.loader import load_files

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Pydantic models ────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Вопрос не может быть пустым")
        return v.strip()


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


class IndexRequest(BaseModel):
    path: str | None = None


class IndexResponse(BaseModel):
    status: str
    files_indexed: int
    chunks_created: int


class HealthResponse(BaseModel):
    status: str
    chromadb: str
    collection_size: int


class BizAskRequest(BaseModel):
    question: str
    role: str = ""

    @field_validator("question")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Вопрос не может быть пустым")
        return v.strip()


class BizAnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    used_code_search: bool


class AgentRequest(BaseModel):
    command: str
    telegram_username: str = ""

    @field_validator("command")
    @classmethod
    def command_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Команда не может быть пустой")
        return v.strip()


class AgentResponse(BaseModel):
    result: str
    steps: list[str]
    jira_url: str | None = None
    mr_url: str | None = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post(
    "/api/ask",
    response_model=AnswerResponse,
    summary="Задать вопрос агенту",
)
async def ask(request: QuestionRequest) -> AnswerResponse:
    try:
        result = agent.ask(request.question)
    except Exception as exc:
        logger.exception("Ошибка при обращении к LLM: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при обращении к LLM",
        ) from exc

    return AnswerResponse(answer=result["answer"], sources=result["sources"])


@router.post(
    "/api/index",
    response_model=IndexResponse,
    summary="Переиндексировать codebase",
)
async def index(request: IndexRequest) -> IndexResponse:
    settings = get_settings()

    paths: list[tuple[str, list[str]]] = []

    if request.path:
        paths.append((request.path, [".java", ".ts", ".tsx", ".sql", ".md"]))
    else:
        if settings.backend_code_path:
            paths.append((settings.backend_code_path, [".java"]))
        if settings.frontend_code_path:
            paths.append((settings.frontend_code_path, [".ts", ".tsx"]))

    if not paths:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Путь не указан и не задан в конфигурации",
        )

    try:
        documents = []
        for path, extensions in paths:
            documents.extend(load_files(path, extensions=extensions))
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if not documents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="По указанному пути не найдено файлов для индексации",
        )

    try:
        stats = embed_and_store(documents, settings)
    except Exception as exc:
        logger.exception("Ошибка при индексации: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при создании embeddings",
        ) from exc

    return IndexResponse(
        status="success",
        files_indexed=stats["files_indexed"],
        chunks_created=stats["chunks_created"],
    )


@router.post(
    "/api/agent/execute",
    response_model=AgentResponse,
    summary="Выполнить команду agentic AI (анализ бага, создание Jira + MR)",
)
async def agent_execute(request: AgentRequest) -> AgentResponse:
    try:
        output = executor.execute(request.command, request.telegram_username)
    except Exception as exc:
        logger.exception("Ошибка AgenticExecutor: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Агент не смог выполнить задачу: {exc}",
        ) from exc

    return AgentResponse(
        result=output["result"],
        steps=output["steps"],
        jira_url=output.get("jira_url"),
        mr_url=output.get("mr_url"),
    )


@router.post(
    "/api/biz/ask",
    response_model=BizAnswerResponse,
    summary="Задать бизнес-вопрос (Confluence + Jira документация)",
)
async def biz_ask(request: BizAskRequest) -> BizAnswerResponse:
    try:
        result = biz_agent.ask(request.question, request.role)
    except Exception as exc:
        logger.exception("Ошибка BizAgent: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при обращении к документации",
        ) from exc

    return BizAnswerResponse(
        answer=result["answer"],
        sources=result["sources"],
        used_code_search=result["used_code_search"],
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния сервиса",
)
async def health() -> HealthResponse:
    try:
        retriever = CodeRetriever()
        collection_size = retriever.get_collection_size()
        chroma_status = "connected"
    except Exception as exc:
        logger.warning("ChromaDB недоступен: %s", exc)
        chroma_status = "disconnected"
        collection_size = 0

    return HealthResponse(
        status="ok",
        chromadb=chroma_status,
        collection_size=collection_size,
    )
