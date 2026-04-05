"""
Тест agentic AI без Telegram бота.

Использование:
    python scripts/test_agent.py "AuthService падает NPE на line 45, разберись"
    python scripts/test_agent.py "баг в авторизации" --user tilewzhan
    python scripts/test_agent.py "что делает JwtFilter?" --url http://localhost:8090
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Тест Agentic AI агента juz40")
    parser.add_argument("command", help="Команда для агента")
    parser.add_argument("--user", default="", metavar="TG_USERNAME", help="Telegram username для маппинга на Jira assignee")
    parser.add_argument("--url", default="http://localhost:8090", metavar="URL", help="Base URL сервиса (default: http://localhost:8090)")
    args = parser.parse_args()

    endpoint = f"{args.url.rstrip('/')}/api/agent/execute"
    payload = {"command": args.command, "telegram_username": args.user}

    print(f"\n▶  Команда:  {args.command}")
    if args.user:
        print(f"   User:     @{args.user}")
    print(f"   Endpoint: {endpoint}\n")
    print("⏳ Выполняю... (может занять 30–120 секунд)\n")

    try:
        response = requests.post(endpoint, json=payload, timeout=180)
        response.raise_for_status()
    except requests.ConnectionError:
        print(f"❌ Не удалось подключиться к {args.url}")
        print("   Убедитесь что сервис запущен: python -m app.main")
        sys.exit(1)
    except requests.Timeout:
        print("❌ Таймаут (180с). Агент не ответил вовремя.")
        sys.exit(1)
    except requests.HTTPError as exc:
        print(f"❌ HTTP {exc.response.status_code}: {exc.response.text[:300]}")
        sys.exit(1)

    data = response.json()

    # ── Шаги ──────────────────────────────────────────────────────────────────
    steps: list[str] = data.get("steps", [])
    if steps:
        numbered = "  ".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        print(f"[ШАГИ]:  {numbered}")
    else:
        print("[ШАГИ]:  —")

    # ── Jira / MR ─────────────────────────────────────────────────────────────
    print(f"[JIRA]:  {data.get('jira_url') or 'не создан'}")
    print(f"[MR]:    {data.get('mr_url') or 'не создан'}")

    # ── Ответ агента ──────────────────────────────────────────────────────────
    print("\n[ОТВЕТ]:")
    print(data.get("result", "—"))
    print()


if __name__ == "__main__":
    main()
