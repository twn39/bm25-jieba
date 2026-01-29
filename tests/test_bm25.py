"""
BM25 中文搜索测试
"""

import pytest
from bm25_jieba import BM25


class TestBM25:
    """BM25 核心功能测试"""

    @pytest.fixture
    def sample_documents(self) -> list[str]:
        """测试文档集"""
        return [
            "Python是一种广泛使用的高级编程语言",
            "机器学习是人工智能的一个分支",
            "深度学习是机器学习的子领域",
            "自然语言处理研究人与计算机之间的语言交互",
            "Python在机器学习领域非常流行",
        ]

    @pytest.fixture
    def bm25(self, sample_documents: list[str]) -> BM25:
        """创建并训练 BM25 实例"""
        model = BM25(k1=1.5, b=0.75)
        model.fit(sample_documents)
        return model

    def test_init_default_params(self):
        """测试默认参数初始化"""
        bm25 = BM25()
        assert bm25 is not None

    def test_init_custom_params(self):
        """测试自定义参数初始化"""
        bm25 = BM25(k1=2.0, b=0.5)
        assert bm25 is not None

    def test_fit(self, sample_documents: list[str]):
        """测试模型训练"""
        bm25 = BM25()
        bm25.fit(sample_documents)
        # 训练后应该能正常搜索
        results = bm25.search("Python")
        assert len(results) > 0

    def test_search_returns_sorted_results(self, bm25: BM25):
        """测试搜索结果按分数降序排列"""
        results = bm25.search("Python")
        assert len(results) >= 2
        # 验证分数降序
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k(self, bm25: BM25):
        """测试 top_k 参数"""
        all_results = bm25.search("机器学习")
        top_2 = bm25.search("机器学习", top_k=2)
        assert len(top_2) == 2
        if len(top_2) == 2 and len(all_results) >= 2:
            # Check scores match
            assert top_2[0][1] == all_results[0][1]
            assert top_2[1][1] == all_results[1][1]
            # Verify that returned documents are valid candidates (exist in all_results top N+ties)
            top_2_ids = set(x[0] for x in top_2)
            all_ids = set(x[0] for x in all_results)
            assert top_2_ids.issubset(all_ids)


    def test_search_no_match(self, bm25: BM25):
        """测试无匹配结果"""
        results = bm25.search("区块链加密货币")
        assert len(results) == 0

    def test_get_scores(self, bm25: BM25, sample_documents: list[str]):
        """测试获取所有文档分数"""
        scores = bm25.get_scores("Python")
        assert len(scores) == len(sample_documents)
        # 第一个和最后一个文档应该有分数（都包含 Python）
        assert scores[0] > 0
        assert scores[4] > 0
        # 其他文档不包含 Python，分数为 0
        assert scores[1] == 0
        assert scores[2] == 0
        assert scores[3] == 0

    def test_empty_query(self, bm25: BM25):
        """测试空查询"""
        results = bm25.search("")
        assert len(results) == 0

    def test_empty_corpus(self):
        """测试空语料库"""
        bm25 = BM25()
        bm25.fit([])
        results = bm25.search("Python")
        assert len(results) == 0

    def test_single_document(self):
        """测试单文档语料库"""
        bm25 = BM25()
        bm25.fit(["Python 编程语言"])
        results = bm25.search("Python")
        assert len(results) == 1
        assert results[0][0] == 0
        assert results[0][1] > 0

    def test_save_load(self, bm25: BM25, tmp_path):
        """测试模型保存和加载"""
        save_path = tmp_path / "bm25.bin"
        bm25.save(str(save_path))
        
        loaded_bm25 = BM25.load(str(save_path))
        assert loaded_bm25 is not None
        
        # 验证加载后的模型搜索结果一致
        results_orig = bm25.search("Python")
        results_loaded = loaded_bm25.search("Python")
        assert results_orig == results_loaded


class TestBM25Scoring:
    """BM25 评分算法测试"""

    def test_exact_match_highest_score(self):
        """精确匹配应获得最高分"""
        bm25 = BM25()
        docs = ["Python 语言", "Java 语言", "C++ 语言"]
        bm25.fit(docs)
        results = bm25.search("Python")
        assert results[0][0] == 0  # 第一个文档匹配 Python

    def test_multiple_term_match(self):
        """多词匹配分数应更高"""
        bm25 = BM25()
        docs = [
            "Python 编程",
            "Python 编程 语言",
            "Java 编程",
        ]
        bm25.fit(docs)
        results = bm25.search("Python 编程")
        # 同时匹配 Python 和 编程 的文档分数更高
        assert results[0][0] in [0, 1]

    def test_term_frequency_impact(self):
        """词频应影响分数"""
        bm25 = BM25()
        docs = ["Python", "Python Python Python"]
        bm25.fit(docs)
        scores = bm25.get_scores("Python")
        # 词频更高的文档分数更高（但受 k1 饱和限制）
        assert scores[1] > scores[0]


class TestBM25CaseInsensitive:
    """BM25 大小写不敏感测试"""

    def test_lowercase_enabled(self):
        """测试开启大小写转换"""
        bm25 = BM25(lowercase=True)
        docs = ["Python is great", "JAVA is heavy"]
        bm25.fit(docs)
        
        # 搜索小写 query 应该能匹配到
        results = bm25.search("python")
        assert len(results) > 0
        assert results[0][0] == 0
        
        # 搜索大写 query 也应该匹配
        results = bm25.search("PYTHON")
        assert len(results) > 0
        assert results[0][0] == 0

    def test_lowercase_disabled_by_default(self):
        """测试默认关闭大小写转换"""
        bm25 = BM25()
        docs = ["Python is great"]
        bm25.fit(docs)
        
        # 默认区分大小写，python 不匹配 Python
        results = bm25.search("python")
        assert len(results) == 0
        
        results = bm25.search("Python")
        assert len(results) > 0

    def test_mixed_language_lowercase(self):
        """测试中英文混合的大小写处理"""
        bm25 = BM25(lowercase=True)
        docs = ["Python深度学习", "机器学习AI"]
        bm25.fit(docs)
        
        # 英文部分应该不区分大小写，中文部分正常
        results = bm25.search("python深度学习")
        assert len(results) > 0
        assert results[0][0] == 0
        
        results = bm25.search("机器学习ai")
        assert len(results) > 0
        assert results[0][0] == 1
