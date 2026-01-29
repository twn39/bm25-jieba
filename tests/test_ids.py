
import pytest
from bm25_jieba import BM25

class TestBM25IDs:
    def test_fit_with_ids(self):
        """Test fitting with custom IDs"""
        bm25 = BM25()
        documents = ["Python", "Java", "Rust"]
        ids = [1001, 1002, 1003]
        
        bm25.fit(documents, ids=ids)
        
        # Test search returns correct ID
        results = bm25.search("Python")
        assert len(results) > 0
        doc_id, score = results[0]
        assert doc_id == 1001
        
        results = bm25.search("Rust")
        assert len(results) > 0
        doc_id, score = results[0]
        assert doc_id == 1003

    def test_fit_id_length_mismatch(self):
        """Test fit raises error if IDs length doesn't match documents"""
        bm25 = BM25()
        documents = ["A", "B"]
        ids = [1] # Too short
        
        with pytest.raises(ValueError, match="documents and ids must have the same length"):
            bm25.fit(documents, ids)

    def test_default_ids(self):
        """Test backward compatibility (no IDs provided)"""
        bm25 = BM25()
        documents = ["A", "B", "C"]
        bm25.fit(documents)
        
        results = bm25.search("A")
        assert len(results) > 0
        doc_id, _ = results[0]
        assert doc_id == 0  # Should be index 0
