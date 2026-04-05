"""
Индексация codebase juz40-edu.kz из GitLab в ChromaDB.

Использование:
    python scripts/index.py
    python scripts/index.py --branch develop
    python scripts/index.py --branch master
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.indexer.embedder import embed_and_store
from app.indexer.loader import load_from_gitlab

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Индексация codebase juz40-edu.kz из GitLab в ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python scripts/index.py\n"
            "  python scripts/index.py --branch develop\n"
            "  python scripts/index.py --branch master\n"
        ),
    )
    parser.add_argument(
        "--branch",
        metavar="BRANCH",
        default=None,
        help="Ветка GitLab (по умолчанию — gitlab_index_branch из .env)",
    )
    args = parser.parse_args()

    settings = get_settings()
    branch = args.branch or settings.gitlab_index_branch

    logger.info("═" * 50)
    logger.info("Запуск индексации juz40 codebase")
    logger.info("GitLab: %s  проект: %s  ветка: %s", settings.gitlab_url, settings.gitlab_project_id, branch)
    logger.info("ChromaDB: %s:%d  коллекция: %s", settings.chroma_host, settings.chroma_port, settings.chroma_collection)
    logger.info("═" * 50)

    documents = load_from_gitlab(branch)

    if not documents:
        logger.error("Не найдено ни одного файла для индексации. Проверьте GITLAB_PROJECT_ID и ветку.")
        sys.exit(1)

    logger.info("── Создание embeddings ─────────────────")
    logger.info("  Всего документов: %d", len(documents))
    logger.info("  chunk_size=%d  chunk_overlap=%d", settings.chunk_size, settings.chunk_overlap)

    try:
        stats = embed_and_store(documents, settings)
    except Exception as exc:
        logger.error("Ошибка при создании embeddings: %s", exc)
        sys.exit(1)

    logger.info("═" * 50)
    logger.info("ИТОГ ИНДЕКСАЦИИ")
    logger.info("═" * 50)
    logger.info("  Файлов:  %d", stats["files_indexed"])
    logger.info("  Чанков:  %d", stats["chunks_created"])
    logger.info("  Ветка:   %s", branch)
    logger.info("═" * 50)


if __name__ == "__main__":
    main()
