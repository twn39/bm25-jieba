"""
BM25 性能测试

测试不同规模数据集下的索引和搜索性能
"""

import time
import random
import string
from bm25 import BM25


def generate_chinese_text(length: int = 50) -> str:
    """生成随机中文文本"""
    # 常用中文字符范围
    chars = "".join(
        chr(i) for i in range(0x4E00, 0x9FA5) if random.random() < 0.01
    )
    if not chars:
        chars = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情最何发成见手次工场华我体全但是开始可能这样没有什么他们我们你们自己知道怎么为什么如果现在已经那么所以虽然但是因为就是这个那个什么时候怎样还是不过那些这些什么地方"
    return "".join(random.choice(chars) for _ in range(length))


def benchmark_fit(doc_count: int, doc_length: int = 100) -> float:
    """测试索引性能"""
    documents = [generate_chinese_text(doc_length) for _ in range(doc_count)]
    
    bm25 = BM25()
    start = time.perf_counter()
    bm25.fit(documents)
    elapsed = time.perf_counter() - start
    
    return elapsed


def benchmark_search(bm25: BM25, query: str, iterations: int = 100) -> float:
    """测试搜索性能"""
    start = time.perf_counter()
    for _ in range(iterations):
        bm25.search(query, top_k=10)
    elapsed = time.perf_counter() - start
    
    return elapsed / iterations


def run_benchmarks():
    """运行完整的性能测试"""
    print("=" * 60)
    print("BM25 性能测试")
    print("=" * 60)
    
    # 索引性能测试
    print("\n📊 索引性能测试 (fit)")
    print("-" * 40)
    
    doc_counts = [100, 1000, 5000, 10000]
    for count in doc_counts:
        elapsed = benchmark_fit(count)
        rate = count / elapsed
        print(f"  {count:>6} 文档: {elapsed:>6.3f}s ({rate:>8.0f} docs/s)")
    
    # 搜索性能测试
    print("\n🔍 搜索性能测试 (search)")
    print("-" * 40)
    
    # 准备测试数据
    documents = [generate_chinese_text(100) for _ in range(10000)]
    bm25 = BM25()
    bm25.fit(documents)
    
    queries = [
        "机器学习",
        "自然语言处理",
        "Python 编程 数据 分析",
    ]
    
    for query in queries:
        avg_time = benchmark_search(bm25, query, iterations=1000)
        qps = 1 / avg_time
        print(f"  查询「{query[:10]}...」: {avg_time*1000:.3f}ms ({qps:.0f} QPS)")
    
    # 内存效率测试（近似）
    print("\n💾 语料库规模测试")
    print("-" * 40)
    
    sizes = [1000, 5000, 10000, 20000]
    for size in sizes:
        documents = [generate_chinese_text(50) for _ in range(size)]
        bm25 = BM25()
        
        fit_time = time.perf_counter()
        bm25.fit(documents)
        fit_elapsed = time.perf_counter() - fit_time
        
        search_time = benchmark_search(bm25, "测试查询", iterations=100)
        
        print(f"  {size:>6} 文档: 索引 {fit_elapsed:.3f}s, 搜索 {search_time*1000:.3f}ms")
    
    print("\n" + "=" * 60)
    print("✅ 性能测试完成")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmarks()
