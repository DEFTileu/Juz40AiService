import logging
from dataclasses import dataclass

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class UserEntry:
    username: str
    telegram_id: str        # числовой Telegram ID (для отправки сообщений)
    jira_account_id: str    # Jira accountId (для assignee)


def _parse_mapping(raw: str) -> dict[str, UserEntry]:
    """
    Поддерживает два формата (можно миксовать):
      Старый: "username:jira_id"
      Новый:  "username:telegram_id:jira_id"

    Разделитель записей — запятая.
    """
    mapping: dict[str, UserEntry] = {}
    if not raw.strip():
        return mapping

    for pair in raw.split(","):
        pair = pair.strip()
        parts = [p.strip() for p in pair.split(":")]

        if len(parts) == 2:
            # username:jira_id
            username, jira_id = parts
            telegram_id = ""
        elif len(parts) == 3:
            # username:telegram_id:jira_id
            username, telegram_id, jira_id = parts
        else:
            logger.warning("user_mapper: некорректная запись '%s', пропускаем", pair)
            continue

        if not username or not jira_id:
            logger.warning("user_mapper: пустые поля в записи '%s', пропускаем", pair)
            continue

        mapping[username] = UserEntry(
            username=username,
            telegram_id=telegram_id,
            jira_account_id=jira_id,
        )

    return mapping


def get_jira_account_id(telegram_username: str) -> str | None:
    """Возвращает Jira accountId для Telegram username. None если не найден."""
    entry = get_user_entry(telegram_username)
    return entry.jira_account_id if entry else None


def get_user_entry(telegram_username: str) -> UserEntry | None:
    """Возвращает полную запись пользователя (username + telegram_id + jira_id)."""
    settings = get_settings()
    mapping = _parse_mapping(settings.telegram_jira_mapping)

    username = telegram_username.lstrip("@").strip()
    entry = mapping.get(username)

    if entry:
        logger.debug("user_mapper: %s → jira=%s tg_id=%s", username, entry.jira_account_id, entry.telegram_id)
    else:
        logger.debug("user_mapper: %s не найден в маппинге", username)

    return entry
