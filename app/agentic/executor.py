import logging
import re

from langchain.agents import AgentState, create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.agentic.prompts import AGENT_SYSTEM_PROMPT
from app.agentic.tools.codebase_search import CodebaseSearchTool
from app.agentic.tools.gitlab_mr import GitLabCommentMRTool, GitLabCreateMRTool, GitLabGetMRTool
from app.agentic.tools.gitlab_read import GitLabListFilesTool, GitLabReadFileTool
from app.agentic.tools.gitlab_write import GitLabCommitTool, GitLabCreateBranchTool
from app.agentic.tools.jira_tools import JiraCreateTaskTool, JiraGetTaskTool
from app.config import get_settings

logger = logging.getLogger(__name__)

_JIRA_URL_RE = re.compile(r"https://[^\s]+/browse/DV-\d+")
_MR_URL_RE = re.compile(r"https://[^\s]+/merge_requests/\d+")


def _extract_url(text: str, pattern: re.Pattern) -> str | None:
    match = pattern.search(text)
    return match.group(0) if match else None


def _collect_steps(messages: list[BaseMessage]) -> list[str]:
    """Извлекает уникальные имена tools из истории сообщений агента."""
    steps: list[str] = []
    seen: set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name", "")
                if name and name not in seen:
                    seen.add(name)
                    steps.append(name)
    return steps


def _extract_urls_from_messages(
    messages: list[BaseMessage],
) -> tuple[str | None, str | None]:
    """Ищет Jira и MR URL во всех сообщениях (ответах tools и финальном ответе)."""
    jira_url: str | None = None
    mr_url: str | None = None
    for msg in messages:
        text = msg.content if isinstance(msg.content, str) else ""
        if not jira_url:
            jira_url = _extract_url(text, _JIRA_URL_RE)
        if not mr_url:
            mr_url = _extract_url(text, _MR_URL_RE)
        if jira_url and mr_url:
            break
    return jira_url, mr_url


class AgenticExecutor:
    """
    Agentic AI — оркестратор пайплайна:
    codebase_search → gitlab_read → анализ → jira → ветка → коммит → MR.

    Использует langchain 1.x create_agent (LangGraph под капотом).
    Инициализируется один раз при старте приложения.
    """

    def __init__(self) -> None:
        settings = get_settings()

        llm = self._build_llm(settings)

        self._tools: list[BaseTool] = [
            CodebaseSearchTool(),
            GitLabReadFileTool(),
            GitLabListFilesTool(),
            GitLabCreateBranchTool(),
            GitLabCommitTool(),
            GitLabCreateMRTool(),
            GitLabCommentMRTool(),
            GitLabGetMRTool(),
            JiraCreateTaskTool(),
            JiraGetTaskTool(),
        ]

        self._graph = create_agent(
            model=llm,
            tools=self._tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
        )

        self._max_iterations = settings.agent_max_iterations
        self._verbose = settings.agent_verbose

        logger.info(
            "AgenticExecutor инициализирован: %d tools, max_iterations=%d",
            len(self._tools),
            self._max_iterations,
        )

    @staticmethod
    def _build_llm(settings):
        if settings.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            logger.info("AgenticExecutor LLM: OpenAI gpt-4o")
            return ChatOpenAI(
                model="gpt-4o",
                openai_api_key=settings.openai_api_key,
                temperature=0.2,
                max_tokens=4096,
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info("AgenticExecutor LLM: Gemini gemini-2.0-flash")
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.gemini_api_key.get_secret_value(),
            temperature=0.2,
            max_output_tokens=4096,
        )

    def execute(self, command: str, telegram_username: str = "") -> dict:
        """
        Запускает агент для выполнения команды.

        Returns:
            {"result": str, "steps": list[str], "jira_url": str|None, "mr_url": str|None}
        """
        from app.agentic.user_mapper import get_jira_account_id

        jira_account_id = get_jira_account_id(telegram_username) if telegram_username else None
        assignee_line = (
            f"Assignee Jira ID: {jira_account_id}" if jira_account_id else "Assignee Jira ID: не определён"
        )
        agent_input = f"{command}\n{assignee_line}"

        logger.info(
            "execute: command='%s' username='%s' assignee=%s",
            command[:80],
            telegram_username,
            jira_account_id,
        )

        try:
            state: AgentState = self._graph.invoke(
                {"messages": [HumanMessage(content=agent_input)]},
                config={"recursion_limit": self._max_iterations * 2},
            )
        except Exception as exc:
            logger.exception("AgenticExecutor ошибка выполнения: %s", exc)
            return {
                "result": f"Агент не смог выполнить задачу: {exc}",
                "steps": [],
                "jira_url": None,
                "mr_url": None,
            }

        messages: list[BaseMessage] = state.get("messages", [])

        # Финальный ответ — последнее AIMessage без tool_calls
        result = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                result = msg.content if isinstance(msg.content, str) else ""
                break

        if self._verbose:
            for msg in messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        logger.info("  → tool: %s  input: %s", tc.get("name"), str(tc.get("args", ""))[:120])
                elif isinstance(msg, ToolMessage):
                    logger.info("  ← %s: %s", msg.name, str(msg.content)[:120])

        steps = _collect_steps(messages)
        jira_url, mr_url = _extract_urls_from_messages(messages)

        logger.info("execute завершён: steps=%s jira=%s mr=%s", steps, jira_url, mr_url)

        return {
            "result": result,
            "steps": steps,
            "jira_url": jira_url,
            "mr_url": mr_url,
        }


# Singleton — инициализируется один раз при импорте модуля
executor = AgenticExecutor()
