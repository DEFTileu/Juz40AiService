"""
Тест BizAgent без HTTP слоя.

Запуск:
    python scripts/test_biz.py "есть ли рекалькуляция для методиста?"
    python scripts/test_biz.py "как куратор смотрит посещаемость?" --role curator
    python scripts/test_biz.py  # набор дефолтных вопросов
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

_DEFAULT_CASES: list[tuple[str, str]] = [
    ("Как студент может найти свои оценки?",           "student"),
    ("Как куратор смотрит посещаемость группы?",       "curator"),
    ("Как завуч запускает рекалькуляцию?",             "zavuch"),
    ("Как методист добавляет учебные материалы?",      "methodist"),
    ("Как родитель проверяет успеваемость ребёнка?",   "parent"),
    ("Какие права есть у администратора платформы?",   "admin"),
    ("Есть ли мобильное приложение?",                  ""),
]


def run(question: str, role: str) -> None:
    from app.biz.biz_agent import biz_agent

    print(f"\nВопрос : {question}")
    if role:
        print(f"Роль   : {role}")
    print("-" * 60)

    result = biz_agent.ask(question, role)

    print(f"[SOURCES]     : {', '.join(result['sources']) if result['sources'] else 'нет'}")
    print(f"[CODE SEARCH] : {'да' if result.get('used_code_search') else 'нет'}")
    print(f"[ОТВЕТ]:\n{result['answer']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Тест BizAgent")
    parser.add_argument("question", nargs="?", default=None, help="Вопрос для теста")
    parser.add_argument("--role", default="", help="Роль: student, curator, zavuch, methodist, parent, admin")
    args = parser.parse_args()

    if args.question:
        run(args.question, args.role)
    else:
        from app.biz.biz_agent import biz_agent
        try:
            size = biz_agent.get_collection_size()
            print(f"\nКоллекция docs: {size} чанков")
        except Exception as exc:
            print(f"[WARN] Не удалось получить размер коллекции: {exc}")
        print("=" * 60)

        for i, (q, r) in enumerate(_DEFAULT_CASES, 1):
            print(f"\n[{i}/{len(_DEFAULT_CASES)}]")
            run(q, r)
            print("=" * 60)


if __name__ == "__main__":
    main()
