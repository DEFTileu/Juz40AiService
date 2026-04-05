import chromadb

client = chromadb.HttpClient(host="localhost", port=8001)

# Все коллекции
collections = client.list_collections()
print(f"Коллекций: {len(collections)}")
for col in collections:
    print(f"  - {col.name}")

# Детали твоей коллекции
col = client.get_collection("juz40_codebase")
count = col.count()
print(f"\nЧанков в juz40_codebase: {count}")

# Посмотреть первые 3 чанка
sample = col.get(limit=3, include=["documents", "metadatas"])
for i, (doc, meta) in enumerate(zip(sample["documents"], sample["metadatas"])):
    print(f"\n--- Чанк {i+1} ---")
    print(f"Файл: {meta.get('source', 'неизвестно')}")
    print(f"Текст: {doc[:200]}...")