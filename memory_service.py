#!/usr/bin/env python3
"""
Local Memory Service — 基于 ChromaDB 的对话记忆向量存储
使用本地 ONNX embedding 模型 (all-MiniLM-L6-v2)，零成本运行。

用法:
  python memory_service.py add --user xing --text "讨论了向量库方案"
  python memory_service.py add --user xing --text "决定用本地 ChromaDB" --metadata '{"topic": "infrastructure"}'
  python memory_service.py search --user xing --query "向量库"
  python memory_service.py search --user xing --query "向量库" --limit 5
  python memory_service.py stats
"""

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "conversations"


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    ef = ONNXMiniLM_L6_V2()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_memory(user: str, text: str, metadata: dict | None = None):
    col = get_collection()
    doc_id = str(uuid.uuid4())
    meta = {
        "user": user,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "epoch": int(time.time()),
    }
    if metadata:
        meta.update(metadata)
    col.add(ids=[doc_id], documents=[text], metadatas=[meta])
    print(json.dumps({"status": "ok", "id": doc_id, "count": col.count()}))


def search_memory(user: str, query: str, limit: int = 3):
    col = get_collection()
    results = col.query(
        query_texts=[query],
        n_results=limit,
        where={"user": user} if user else None,
        include=["documents", "metadatas", "distances"],
    )
    items = []
    for i, doc in enumerate(results["documents"][0]):
        items.append({
            "text": doc,
            "score": round(1 - results["distances"][0][i], 4),  # cosine similarity
            "metadata": results["metadatas"][0][i],
        })
    print(json.dumps({"results": items, "total": col.count()}, ensure_ascii=False))


def show_stats():
    col = get_collection()
    print(json.dumps({"collection": COLLECTION_NAME, "total_memories": col.count()}))


def main():
    parser = argparse.ArgumentParser(description="Local Memory Service")
    sub = parser.add_subparsers(dest="command")

    p_add = sub.add_parser("add")
    p_add.add_argument("--user", required=True)
    p_add.add_argument("--text", required=True)
    p_add.add_argument("--metadata", default=None, help="JSON string")

    p_search = sub.add_parser("search")
    p_search.add_argument("--user", default=None)
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--limit", type=int, default=3)

    sub.add_parser("stats")

    args = parser.parse_args()
    if args.command == "add":
        meta = json.loads(args.metadata) if args.metadata else None
        add_memory(args.user, args.text, meta)
    elif args.command == "search":
        search_memory(args.user, args.query, args.limit)
    elif args.command == "stats":
        show_stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
