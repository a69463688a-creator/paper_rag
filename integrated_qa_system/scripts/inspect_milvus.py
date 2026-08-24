"""
查看 Milvus 数据库（只读，不修改任何数据）
    - 列出所有数据库 / 集合
    - 每个集合的数据量 (row_count)
    - Schema 字段定义
    - 前 2 条样本数据（只取标量字段，跳过向量）

用法:
    D:/conda_envs/PaperRAG-GPU/python scripts/inspect_milvus.py
    （或在 paperrag-app 容器内: cd /app && python scripts/inspect_milvus.py）
"""
from pymilvus import MilvusClient, DataType

HOST = "localhost"
PORT = 19530
DB = "paper_rag"

VECTOR_TYPES = {
    DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR, DataType.BINARY_VECTOR,
    DataType.FLOAT16_VECTOR, DataType.BFLOAT16_VECTOR,
}


def type_name(t):
    name = getattr(t, "name", None)
    if name:
        return name
    try:
        return DataType(t).name
    except Exception:
        return str(t)


def main():
    client = MilvusClient(uri=f"http://{HOST}:{PORT}", db_name=DB)

    print("=== 数据库列表 ===")
    for db in client.list_databases():
        mark = "  ← 当前" if db == DB else ""
        print(f"  {db}{mark}")

    collections = client.list_collections()
    print(f"\n=== 集合列表 ({DB}) ===")
    for c in collections:
        print(f"  {c}")

    for name in collections:
        stats = client.get_collection_stats(name)
        desc = client.describe_collection(name)
        fields = desc["fields"]
        scalar_fields = [f["name"] for f in fields if f["type"] not in VECTOR_TYPES]

        print(f"\n───── {name} ─────")
        print(f"  数据量 row_count = {stats['row_count']}")
        print("  字段: " + ", ".join(
            f"{f['name']}({type_name(f['type'])})" for f in fields
        ))

        # 样本数据：只取前几个标量字段，避开向量
        try:
            rows = client.query(
                name, filter="", limit=2,
                output_fields=scalar_fields[:6],
            )
            if rows:
                print("  样本(前2条):")
                for r in rows:
                    for k, v in r.items():
                        v = str(v)
                        if len(v) > 70:
                            v = v[:70] + "…"
                        print(f"    {k} = {v}")
                    print("    ---")
        except Exception as e:
            print(f"  样本查询失败: {e}")


if __name__ == "__main__":
    main()
