"""数据访问层：每个函数封装一次 milvus 调用。

不处理业务逻辑（去重、分组、聚合）——那是 service.py 的事。
所有 chapter 相关查询都用 document_id 过滤（chapter_id 文档内唯一，不保证全局唯一）。
"""
from __future__ import annotations

import logging
from typing import Any

from pymilvus import MilvusClient

from app.rag.document_rag.config import CHAPTER_COLL, SENTENCE_COLL

logger = logging.getLogger(__name__)


# ============ insert ============
def insert_chapters(client: MilvusClient, rows: list[dict]) -> int:
    if not rows:
        return 0
    client.insert(collection_name=CHAPTER_COLL, data=rows)
    return len(rows)


def insert_sentences(client: MilvusClient, rows: list[dict]) -> int:
    if not rows:
        return 0
    client.insert(collection_name=SENTENCE_COLL, data=rows)
    return len(rows)


# ============ query / search ============
def search_sentences(
    client: MilvusClient,
    query_vector: list[float],
    limit: int,
) -> list[dict[str, Any]]:
    """dense vector 检索 sentence；返回 hit 列表，含分数与 chapter 关联字段。"""
    res = client.search(
        collection_name=SENTENCE_COLL,
        data=[query_vector],
        anns_field="dense_vector",
        search_params={"metric_type": "COSINE"},
        limit=limit,
        output_fields=[
            "document_id",
            "chapter_id",
            "chunk_text",
            "chunk_index",
            "document_title",
            "chapter_title",
        ],
    )
    if not res:
        return []
    return res[0]


def query_chapters_by_document(
    client: MilvusClient,
    document_id: int,
) -> list[dict[str, Any]]:
    """按 document_id 反查所有 chapter 记录。"""
    return client.query(
        collection_name=CHAPTER_COLL,
        filter=f"document_id == {int(document_id)}",
        output_fields=[
            "chapter_id",
            "chapter_title",
            "chapter_text",
            "char_count",
            "document_title",
        ],
    )


def query_all_chapters(client: MilvusClient) -> list[dict[str, Any]]:
    """全表扫描所有 chapter；用于 list_documents 在内存聚合 document 维度。

    数据量大时（>10w）应改用 milvus 的 query iterator；
    当前数据量小，简单实现。
    """
    return client.query(
        collection_name=CHAPTER_COLL,
        filter="document_id >= 0",  # 全量过滤（不能为空字符串）
        output_fields=[
            "document_id",
            "document_title",
            "chapter_id",
            "char_count",
        ],
    )


# ============ delete ============
def delete_chapters_by_document(client: MilvusClient, document_id: int) -> None:
    client.delete(
        collection_name=CHAPTER_COLL,
        filter=f"document_id == {int(document_id)}",
    )


def delete_sentences_by_document(client: MilvusClient, document_id: int) -> None:
    client.delete(
        collection_name=SENTENCE_COLL,
        filter=f"document_id == {int(document_id)}",
    )


# ============ collection 管理 ============
def list_collections(client: MilvusClient) -> list[str]:
    """列出 milvus 实例上所有集合名。"""
    return client.list_collections()


def has_collection(client: MilvusClient, name: str) -> bool:
    return client.has_collection(collection_name=name)


def get_collection_row_count(client: MilvusClient, name: str) -> int:
    """get_collection_stats 走元数据，不扫全表。"""
    stats = client.get_collection_stats(collection_name=name)
    return int(stats.get("row_count", 0))


def drop_collection(client: MilvusClient, name: str) -> None:
    client.drop_collection(collection_name=name)
