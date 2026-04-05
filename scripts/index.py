"""
Индексация codebase juz40-edu.kz из GitLab в ChromaDB.

Использование:
    python scripts/index.py                          # оба проекта, ветка из .env
    python scripts/index.py --branch develop         # конкретная ветка
    python scripts/index.py --backend-only           # только бэкенд
    python scripts/index.py --frontend-only          # только фронт
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
            "  python scripts/index.py --backend-only\n"
            "  python scripts/index.py --frontend-only\n"
        ),
    )
    parser.add_argument("--branch", metavar="BRANCH", default=None,
                        help="Ветка GitLab (по умолчанию — GITLAB_INDEX_BRANCH из .env)")
    parser.add_argument("--backend-only", action="store_true", help="Индексировать только бэкенд")
    parser.add_argument("--frontend-only", action="store_true", help="Индексировать только фронт")
    args = parser.parse_args()

    settings = get_settings()
    branch = args.branch or settings.gitlab_index_branch

    run_backend = not args.frontend_only
    run_frontend = not args.backend_only

    backend_id = settings.gitlab_backend_project_id or settings.gitlab_project_id
    frontend_id = settings.gitlab_frontend_project_id

    logger.info("═" * 55)
    logger.info("Запуск индексации juz40 codebase  ветка: %s", branch)
    logger.info("ChromaDB: %s:%d  коллекция: %s", settings.chroma_host, settings.chroma_port, settings.chroma_collection)
    logger.info("═" * 55)

    all_docs = []

    if run_backend:
        if not backend_id:
            logger.error("GITLAB_BACKEND_PROJECT_ID (или GITLAB_PROJECT_ID) не задан в .env")
            sys.exit(1)
        logger.info("── Backend  project_id=%s", backend_id)
        docs = load_from_gitlab(branch, project_id=backend_id)
        logger.info("   Найдено файлов: %d", len(docs))
        all_docs.extend(docs)

    if run_frontend:
        if not frontend_id:
            logger.warning("GITLAB_FRONTEND_PROJECT_ID не задан — фронт пропускаем")
        else:
            logger.info("── Frontend  project_id=%s", frontend_id)
            docs = load_from_gitlab(branch, project_id=frontend_id)
            logger.info("   Найдено файлов: %d", len(docs))
            all_docs.extend(docs)

    if not all_docs:
        logger.error("Не найдено ни одного файла. Проверьте project ID и ветку.")
        sys.exit(1)

    logger.info("── Создание embeddings  всего файлов: %d", len(all_docs))

    try:
        stats = embed_and_store(all_docs, settings)
    except Exception as exc:
        logger.error("Ошибка при создании embeddings: %s", exc)
        sys.exit(1)

    logger.info("═" * 55)
    logger.info("ИТОГ:  файлов=%d  чанков=%d  ветка=%s", stats["files_indexed"], stats["chunks_created"], branch)
    logger.info("═" * 55)


if __name__ == "__main__":
    main()
