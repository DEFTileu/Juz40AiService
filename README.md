# Juz40 AI Agent Service

RAG-агент для ответов на технические вопросы по коду проекта juz40-edu.kz.

**Стек:** Python 3.11+ · FastAPI · LangChain · Gemini · ChromaDB

---

## Быстрый старт

### 1. Установка зависимостей

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Конфигурация

```bash
cp .env.example .env
# Заполни .env: GEMINI_API_KEY, BACKEND_CODE_PATH, FRONTEND_CODE_PATH
```

### 3. Запуск ChromaDB

```bash
docker compose up -d
# ChromaDB доступен на http://localhost:8001
```

### 4. Индексация codebase

```bash
python scripts/index.py
# Или с явными путями:
python scripts/index.py --backend /path/to/backend/src --frontend /path/to/frontend/src
```

### 5. Запуск AI сервиса

```bash
python -m app.main
# Сервис доступен на http://localhost:8090
```

---

## Эндпойнты

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/api/ask` | Задать вопрос агенту |
| `POST` | `/api/index` | Переиндексировать codebase |
| `GET` | `/health` | Проверка состояния сервиса |

### Пример запроса

```bash
curl -X POST http://localhost:8090/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "как работает авторизация?"}'
```

---

## Переменные окружения

| Переменная | Описание | По умолчанию |
|---|---|---|
| `LLM_PROVIDER` | Провайдер LLM: `gemini` или `openai` | `gemini` |
| `GEMINI_API_KEY` | API ключ Google Gemini | — |
| `CHROMA_HOST` | Хост ChromaDB | `localhost` |
| `CHROMA_PORT` | Порт ChromaDB | `8001` |
| `CHROMA_COLLECTION` | Название коллекции | `juz40_codebase` |
| `BACKEND_CODE_PATH` | Путь к Java коду | — |
| `FRONTEND_CODE_PATH` | Путь к TypeScript коду | — |
| `CHUNK_SIZE` | Размер чанка при индексации | `800` |
| `TOP_K_RESULTS` | Кол-во чанков для контекста | `5` |
| `LLM_TEMPERATURE` | Температура LLM (0 = точный) | `0` |
