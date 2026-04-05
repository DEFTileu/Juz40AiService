import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator

from app.agent.chain import agent
from app.agent.retriever import CodeRetriever
from app.agentic.executor import executor
from app.api.auth import verify_api_key
from app.biz.biz_agent import biz_agent
from app.config import get_settings
from app.indexer.embedder import embed_and_store
from app.indexer.loader import load_from_gitlab  # noqa: F401 used in index()

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key)])
public_router = APIRouter()  # без токена — только /health


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
    branch: str = ""      # пусто = берём gitlab_index_branch из .env
    backend: bool = True
    frontend: bool = True


class IndexResponse(BaseModel):
    status: str
    files_indexed: int
    chunks_created: int


class IndexDocsRequest(BaseModel):
    confluence: bool = True
    jira: bool = True


class IndexDocsResponse(BaseModel):
    status: str
    confluence_pages: int
    confluence_skipped: int
    jira_stories: int
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
    summary="Переиндексировать codebase из GitLab (бэкенд + фронт)",
)
async def index(request: IndexRequest) -> IndexResponse:
    settings = get_settings()
    branch = request.branch or settings.gitlab_index_branch
    backend_id = settings.gitlab_backend_project_id or settings.gitlab_project_id
    frontend_id = settings.gitlab_frontend_project_id

    all_docs = []
    try:
        if request.backend and backend_id:
            docs = load_from_gitlab(branch, project_id=backend_id)
            all_docs.extend(docs)

        if request.frontend and frontend_id:
            docs = load_from_gitlab(branch, project_id=frontend_id)
            all_docs.extend(docs)
    except Exception as exc:
        logger.exception("Ошибка загрузки из GitLab: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Ошибка GitLab API: {exc}",
        ) from exc

    if not all_docs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"В ветке '{branch}' не найдено файлов для индексации",
        )

    try:
        stats = embed_and_store(all_docs, settings)
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
    "/api/index/docs",
    response_model=IndexDocsResponse,
    summary="Переиндексировать документацию (Confluence + Jira)",
)
async def index_docs(request: IndexDocsRequest) -> IndexDocsResponse:
    from indexer.confluence_indexer import index_confluence
    from indexer.jira_indexer import index_jira_stories

    conf_pages = conf_skipped = conf_chunks = 0
    jira_stories = jira_chunks = 0
    settings = get_settings()

    try:
        if request.confluence:
            result = index_confluence(settings)
            conf_pages = result["pages_indexed"]
            conf_skipped = result["pages_skipped"]
            conf_chunks = result["chunks_created"]

        if request.jira:
            result = index_jira_stories(settings)
            jira_stories = result["stories_indexed"]
            jira_chunks = result["chunks_created"]
    except Exception as exc:
        logger.exception("Ошибка при индексации документов: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка индексации: {exc}",
        ) from exc

    return IndexDocsResponse(
        status="success",
        confluence_pages=conf_pages,
        confluence_skipped=conf_skipped,
        jira_stories=jira_stories,
        chunks_created=conf_chunks + jira_chunks,
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


@public_router.get(
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
