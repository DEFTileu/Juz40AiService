"""
Индексирует Confluence и/или Jira в ChromaDB коллекцию juz40_docs.

Запуск:
    python scripts/index_docs.py              # оба источника
    python scripts/index_docs.py --confluence # только Confluence
    python scripts/index_docs.py --jira       # только Jira
"""

import argparse
import logging
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from indexer.confluence_indexer import index_confluence
from indexer.jira_indexer import index_jira_stories

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Индексация Confluence и Jira в ChromaDB (коллекция juz40_docs)"
    )
    parser.add_argument(
        "--confluence",
        action="store_true",
        help="Индексировать только Confluence",
    )
    parser.add_argument(
        "--jira",
        action="store_true",
        help="Индексировать только Jira",
    )
    args = parser.parse_args()

    # Без флагов — индексируем всё
    run_confluence = args.confluence or (not args.confluence and not args.jira)
    run_jira = args.jira or (not args.confluence and not args.jira)

    settings = get_settings()

    conf_pages = conf_skipped = conf_chunks = 0
    jira_stories = jira_chunks = 0

    if run_confluence:
        logger.info("=== Запуск индексации Confluence ===")
        result = index_confluence(settings)
        conf_pages = result["pages_indexed"]
        conf_skipped = result["pages_skipped"]
        conf_chunks = result["chunks_created"]

    if run_jira:
        logger.info("=== Запуск индексации Jira ===")
        result = index_jira_stories(settings)
        jira_stories = result["stories_indexed"]
        jira_chunks = result["chunks_created"]

    total_chunks = conf_chunks + jira_chunks

    print()
    print("=" * 55)
    if run_confluence:
        print(f"Confluence: {conf_pages} страниц, пропущено {conf_skipped} (тест кейсы), {conf_chunks} chunks")
    if run_jira:
        print(f"Jira: {jira_stories} stories, {jira_chunks} chunks")
    print(f"Итого в {settings.docs_collection}: {total_chunks} chunks")
    print("=" * 55)


if __name__ == "__main__":
    main()
