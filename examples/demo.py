"""
BM25 中文文本搜索示例
"""

from bm25_jieba import BM25


def main():
    # 示例文档集
    documents = [
        "Python是一种广泛使用的高级编程语言，它具有简洁的语法和强大的库支持。",
        "机器学习是人工智能的一个分支，它使计算机能够从数据中学习。",
        "深度学习是机器学习的子领域，使用神经网络进行复杂的模式识别。",
        "自然语言处理是人工智能领域研究人与计算机之间语言交互的技术。",
        "Python在数据科学和机器学习领域非常流行，有丰富的生态系统。",
        "搜索引擎使用各种算法来检索和排序相关文档。",
        "BM25是一种常用的文本检索算法，基于词频和逆文档频率。",
        "中文分词是中文自然语言处理的基础步骤。",
    ]

    # 创建 BM25 模型并训练
    bm25 = BM25(k1=1.5, b=0.75)
    bm25.fit(documents)

    # 测试搜索
    queries = [
        "Python 编程语言",
        "机器学习 人工智能",
        "自然语言处理 中文",
        "搜索算法 BM25",
    ]

    print("=" * 60)
    print("BM25 中文搜索示例")
    print("=" * 60)

    for query in queries:
        print(f"\n查询: 「{query}」")
        print("-" * 40)

        results = bm25.search(query, top_k=3)

        if results:
            for rank, (doc_idx, score) in enumerate(results, 1):
                print(f"  {rank}. [分数: {score:.4f}]")
                print(f"     {documents[doc_idx]}")
        else:
            print("  未找到相关文档")

    # 显示所有文档的分数
    print("\n" + "=" * 60)
    print("所有文档对查询「机器学习」的分数:")
    print("-" * 40)

    scores = bm25.get_scores("机器学习")
    for i, (doc, score) in enumerate(zip(documents, scores)):
        print(f"  文档 {i}: {score:.4f} - {doc[:30]}...")


if __name__ == "__main__":
    main()
