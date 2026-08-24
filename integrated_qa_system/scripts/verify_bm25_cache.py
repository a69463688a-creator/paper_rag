#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 BM25 缓存机制：TTL 懒重建 + 主动失效

运行前提：MySQL + Redis 在线（docker compose up -d mysql redis）
用法：python scripts/verify_bm25_cache.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.config import Config
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MySQLClient
from mysql_qa.retrieval.bm25_search import BM25Search


def main():
    conf = Config()
    print(f"[1] BM25_CORPUS_TTL = {conf.BM25_CORPUS_TTL}s（期望 300）")

    redis_cli = RedisClient()
    mysql_cli = MySQLClient()

    # 从干净状态开始
    print("[2] 清空旧语料/答案缓存...")
    redis_cli.flush_qa_corpus()
    redis_cli.invalidate_all_answers()

    # 实例化（首次强制加载，走 MySQL 回源）
    print("[3] 实例化 BM25Search（首次强制加载）...")
    bm25 = BM25Search(redis_cli, mysql_cli)
    print(f"    语料条数: {len(bm25.original_questions)}")

    # 验证语料缓存有 TTL（旧代码这里是 -1 永久）
    ttl_orig = redis_cli.client.ttl('qa_original_questions')
    ttl_tok = redis_cli.client.ttl('qa_tokenized_questions')
    print(f"[4] 语料缓存 TTL: original={ttl_orig}s, tokenized={ttl_tok}s（期望 >0，旧代码为 -1）")
    assert ttl_orig > 0 and ttl_tok > 0, "语料缓存缺少 TTL！"

    # 验证懒重建短路（未超 TTL 不重建，零开销）
    ts_before = bm25._last_load_ts
    bm25._load_data()
    short_circuit = (bm25._last_load_ts == ts_before)
    print(f"[5] 懒重建短路: {short_circuit}（未超 TTL 应 True）")
    assert short_circuit, "懒重建未短路，每次查询都重建！"

    # 验证主动失效 + reload
    print("[6] 主动失效语料缓存 + reload...")
    redis_cli.flush_qa_corpus()
    after_flush = redis_cli.get_data('qa_original_questions')
    print(f"    失效后语料缓存: {after_flush}（期望 None）")
    bm25.reload()
    print(f"    reload 后语料条数: {len(bm25.original_questions)}")

    # 验证答案缓存失效
    print("[7] 答案缓存主动失效...")
    redis_cli.set_answer('测试问题XYZ', '测试答案')
    before = redis_cli.get_answer('测试问题XYZ')
    redis_cli.invalidate_answer('测试问题XYZ')
    after = redis_cli.get_answer('测试问题XYZ')
    print(f"    失效前: {before} / 失效后: {after}（期望 None）")
    assert before == '测试答案' and after is None, "答案缓存失效失败！"

    print("\n✅ 缓存机制验证全部通过")


if __name__ == '__main__':
    main()
