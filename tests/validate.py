"""
BM25 算法正确性验证

对比我们的 Rust 实现与 rank-bm25 (Python 参考实现) 的计算结果
"""

import math
import jieba
from rank_bm25 import BM25Okapi
from bm25_jieba import BM25


def tokenize_jieba(text: str) -> list[str]:
    """使用 jieba 分词"""
    return [w for w in jieba.cut(text) if w.strip()]


def validate_against_reference():
    """与 rank-bm25 参考实现对比"""
    print("=" * 60)
    print("BM25 算法正确性验证")
    print("=" * 60)

    # 测试文档集 - 扩展为更大更多样的语料库
    documents = [
        # Python 相关
        "Python是一种广泛使用的高级编程语言，具有简洁的语法",
        "Python在机器学习和数据科学领域非常流行",
        "Python的语法简单易学，适合初学者入门编程",
        "Django是Python的一个流行Web框架",
        # 机器学习/AI相关
        "机器学习是人工智能的一个分支，使计算机能够从数据中学习",
        "深度学习是机器学习的子领域，使用神经网络进行模式识别",
        "神经网络模拟人脑的结构，用于处理复杂的模式识别任务",
        "强化学习是机器学习的另一个分支，通过奖励机制进行学习",
        # 自然语言处理相关
        "自然语言处理研究人与计算机之间的语言交互技术",
        "中文分词是中文自然语言处理的基础步骤",
        "命名实体识别是自然语言处理中的重要任务",
        "文本分类是自然语言处理的常见应用场景",
        # 搜索引擎相关
        "搜索引擎使用倒排索引来快速检索文档",
        "BM25是一种经典的文本相关性排序算法",
        "TF-IDF是信息检索中常用的特征权重计算方法",
        "向量空间模型是信息检索的理论基础之一",
        # 数据库相关
        "PostgreSQL是一个强大的开源关系型数据库",
        "MongoDB是流行的NoSQL文档数据库",
        "Redis常用于缓存和实时数据处理场景",
    ]

    queries = [
        "Python 编程语言",
        "机器学习 人工智能",
        "深度学习 神经网络",
        "自然语言处理 中文",
        "搜索引擎 BM25 算法",
        "数据库 PostgreSQL",
    ]

    # 参数
    k1, b = 1.5, 0.75

    # 准备 rank-bm25 (参考实现)
    tokenized_docs = [tokenize_jieba(doc) for doc in documents]
    bm25_ref = BM25Okapi(tokenized_docs, k1=k1, b=b)

    # 准备我们的实现
    bm25_ours = BM25(k1=k1, b=b)
    bm25_ours.fit(documents)

    print(f"\n📋 测试配置: k1={k1}, b={b}, 文档数={len(documents)}")
    print("-" * 60)

    score_match = True
    ranking_match = True

    for query in queries:
        print(f"\n🔍 查询: 「{query}」")

        # 参考实现的分数
        query_tokens = tokenize_jieba(query)
        ref_scores = bm25_ref.get_scores(query_tokens)

        # 我们的分数
        our_scores = bm25_ours.get_scores(query)

        print(f"  {'文档':<4} {'参考实现':>12} {'我们实现':>12} {'差异':>10} {'状态':>6}")
        print("  " + "-" * 50)

        for i, (ref, ours) in enumerate(zip(ref_scores, our_scores)):
            diff = abs(ref - ours)
            # 允许小于 0.01 的误差
            if diff < 0.01:
                status = "✅"
            else:
                status = "⚠️"
                score_match = False
            
            print(f"  {i:<4} {ref:>12.4f} {ours:>12.4f} {diff:>10.4f} {status:>6}")

    # 验证排序一致性
    print("\n" + "=" * 60)
    print("📊 排序一致性验证")
    print("-" * 60)

    for query in queries:
        query_tokens = tokenize_jieba(query)
        ref_scores = bm25_ref.get_scores(query_tokens)
        our_scores = bm25_ours.get_scores(query)

        # 获取排序后的索引
        ref_ranking = sorted(range(len(ref_scores)), key=lambda i: ref_scores[i], reverse=True)
        our_ranking = sorted(range(len(our_scores)), key=lambda i: our_scores[i], reverse=True)

        match = ref_ranking == our_ranking
        status = "✅" if match else "❌"
        if not match:
            ranking_match = False

        print(f"  查询「{query}」")
        print(f"    参考排序: {ref_ranking}")
        print(f"    我们排序: {our_ranking}")
        print(f"    状态: {status}")

    # 结果汇总
    print("\n" + "=" * 60)
    if ranking_match:
        print("✅ 验证通过！我们的实现与参考实现排序完全一致")
        if not score_match:
            print("⚠️ 注意：绝对分数存在差异（因 IDF +1 修正），这是预期的行为")
    else:
        print("❌ 验证失败！排序结果不一致")
    print("=" * 60)

    return ranking_match


def validate_bm25_formula():
    """验证 BM25 公式的数学正确性"""
    print("\n" + "=" * 60)
    print("📐 BM25 公式手动验证")
    print("=" * 60)

    # 简单测试用例
    docs = ["机器 学习", "深度 学习"]
    query = "机器"

    bm25 = BM25(k1=1.5, b=0.75)
    bm25.fit(docs)

    scores = bm25.get_scores(query)

    # 手动计算
    # N = 2 (文档总数)
    # avgdl = (2 + 2) / 2 = 2 (平均文档长度，按词计算)
    # df("机器") = 1 (包含"机器"的文档数)
    # 
    # IDF("机器") = ln((N - df + 0.5) / (df + 0.5) + 1)
    #            = ln((2 - 1 + 0.5) / (1 + 0.5) + 1)
    #            = ln(1.5 / 1.5 + 1)
    #            = ln(2) ≈ 0.693

    N = 2
    avgdl = 2.0
    df = 1
    k1, b = 1.5, 0.75

    idf_manual = math.log((N - df + 0.5) / (df + 0.5) + 1)

    # 文档0的分数 ("机器 学习")
    # tf = 1, dl = 2
    tf, dl = 1, 2
    numerator = tf * (k1 + 1)
    denominator = tf + k1 * (1 - b + b * dl / avgdl)
    score0_manual = idf_manual * numerator / denominator

    # 文档1的分数 ("深度 学习") - 不包含"机器"
    score1_manual = 0.0

    print(f"\n  文档: {docs}")
    print(f"  查询: 「{query}」")
    print(f"  参数: k1={k1}, b={b}, N={N}, avgdl={avgdl}")
    print(f"\n  手动计算:")
    print(f"    IDF(\"机器\") = ln((2-1+0.5)/(1+0.5)+1) = {idf_manual:.4f}")
    print(f"    文档0分数 = {score0_manual:.4f}")
    print(f"    文档1分数 = {score1_manual:.4f}")
    print(f"\n  实现计算:")
    print(f"    文档0分数 = {scores[0]:.4f}")
    print(f"    文档1分数 = {scores[1]:.4f}")

    # 验证
    diff0 = abs(scores[0] - score0_manual)
    diff1 = abs(scores[1] - score1_manual)

    if diff0 < 0.01 and diff1 < 0.01:
        print(f"\n  ✅ 公式验证通过！")
        return True
    else:
        print(f"\n  ❌ 公式验证失败！差异: 文档0={diff0:.4f}, 文档1={diff1:.4f}")
        return False


if __name__ == "__main__":
    # 禁用 jieba 日志
    import logging
    jieba.setLogLevel(logging.WARNING)

    validate_bm25_formula()
    print()
    validate_against_reference()
