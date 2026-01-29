//! BM25 中文文本搜索算法 Rust 实现
//!
//! 使用 jieba-rs 进行中文分词，纯 Rust 实现 BM25 算法
//! 通过 PyO3 提供 Python 绑定

use jieba_rs::Jieba;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::LazyLock;

/// 全局 Jieba 实例（线程安全，延迟初始化）
static JIEBA: LazyLock<Jieba> = LazyLock::new(Jieba::new);

/// BM25 中文文本搜索算法
///
/// BM25 (Best Matching 25) 是一种基于概率检索模型的排序函数，
/// 用于估计文档与查询的相关性。
///
/// Args:
///     k1: 词频饱和参数，控制词频的影响程度，默认 1.5
///     b: 文档长度归一化参数，0 表示不考虑长度，1 表示完全归一化，默认 0.75
///     lowercase: 是否将文本转换为小写（用于大小写不敏感的英文匹配），默认 false
#[pyclass]
pub struct BM25 {
    k1: f64,
    b: f64,
    lowercase: bool,
    corpus_size: usize,
    avgdl: f64,
    doc_lengths: Vec<usize>,
    doc_freqs: Vec<HashMap<String, usize>>,
    idf: HashMap<String, f64>,
}

#[pymethods]
impl BM25 {
    /// 创建新的 BM25 实例
    #[new]
    #[pyo3(signature = (k1=1.5, b=0.75, lowercase=false))]
    pub fn new(k1: f64, b: f64, lowercase: bool) -> Self {
        BM25 {
            k1,
            b,
            lowercase,
            corpus_size: 0,
            avgdl: 0.0,
            doc_lengths: Vec::new(),
            doc_freqs: Vec::new(),
            idf: HashMap::new(),
        }
    }

    /// 使用文档语料库训练 BM25 模型
    ///
    /// Args:
    ///     documents: 文档列表，每个文档是一个字符串
    pub fn fit(&mut self, documents: Vec<String>) {
        self.corpus_size = documents.len();
        self.doc_lengths.clear();
        self.doc_freqs.clear();
        self.idf.clear();

        let mut term_doc_count: HashMap<String, usize> = HashMap::new();
        let mut total_length: usize = 0;

        for doc in &documents {
            // 使用 jieba 分词
            let tokens = self.tokenize(doc);
            let doc_len = tokens.len();
            self.doc_lengths.push(doc_len);
            total_length += doc_len;

            // 统计词频
            let mut freq: HashMap<String, usize> = HashMap::new();
            for token in &tokens {
                *freq.entry(token.clone()).or_insert(0) += 1;
            }

            // 统计包含每个词的文档数
            for token in freq.keys() {
                *term_doc_count.entry(token.clone()).or_insert(0) += 1;
            }

            self.doc_freqs.push(freq);
        }

        // 计算平均文档长度
        self.avgdl = if self.corpus_size > 0 {
            total_length as f64 / self.corpus_size as f64
        } else {
            0.0
        };

        // 计算 IDF
        // IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
        for (term, doc_count) in &term_doc_count {
            let numerator = self.corpus_size as f64 - *doc_count as f64 + 0.5;
            let denominator = *doc_count as f64 + 0.5;
            let idf = (numerator / denominator + 1.0).ln();
            self.idf.insert(term.clone(), idf);
        }
    }

    /// 搜索与查询最相关的文档
    ///
    /// Args:
    ///     query: 查询字符串
    ///     top_k: 返回的最大文档数，None 表示返回所有有得分的文档
    ///
    /// Returns:
    ///     按分数降序排列的 (文档索引, 分数) 元组列表
    #[pyo3(signature = (query, top_k=None))]
    pub fn search(&self, query: &str, top_k: Option<usize>) -> Vec<(usize, f64)> {
        let query_tokens = self.tokenize(query);

        // 计算所有文档的分数
        let mut scores: Vec<(usize, f64)> = (0..self.corpus_size)
            .map(|i| (i, self.score_document(&query_tokens, i)))
            .filter(|(_, score)| *score > 0.0)
            .collect();

        // 按分数降序排序
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 返回 top_k 个结果
        if let Some(k) = top_k {
            scores.truncate(k);
        }

        scores
    }

    /// 获取所有文档的 BM25 分数
    ///
    /// Args:
    ///     query: 查询字符串
    ///
    /// Returns:
    ///     每个文档的分数列表，索引对应文档顺序
    pub fn get_scores(&self, query: &str) -> Vec<f64> {
        let query_tokens = self.tokenize(query);
        (0..self.corpus_size)
            .map(|i| self.score_document(&query_tokens, i))
            .collect()
    }
}

impl BM25 {
    /// 使用 jieba 对中文文本进行分词
    /// 如果 lowercase 为 true，则将分词结果转换为小写
    fn tokenize(&self, text: &str) -> Vec<String> {
        JIEBA
            .cut(text, false)
            .into_iter()
            .map(|s| {
                if self.lowercase {
                    s.to_lowercase()
                } else {
                    s.to_string()
                }
            })
            .filter(|s| !s.trim().is_empty())
            .collect()
    }

    /// 计算单个文档的 BM25 分数
    ///
    /// BM25 分数公式:
    /// score(D, Q) = Σ IDF(q) * (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b * |D| / avgdl))
    fn score_document(&self, query_tokens: &[String], doc_index: usize) -> f64 {
        let doc_len = self.doc_lengths[doc_index];
        let freq = &self.doc_freqs[doc_index];

        let mut score = 0.0;

        for token in query_tokens {
            if let Some(&term_freq) = freq.get(token) {
                let idf = self.idf.get(token).copied().unwrap_or(0.0);

                // BM25 公式
                let numerator = term_freq as f64 * (self.k1 + 1.0);
                let denominator = term_freq as f64
                    + self.k1 * (1.0 - self.b + self.b * doc_len as f64 / self.avgdl);

                score += idf * numerator / denominator;
            }
        }

        score
    }
}

/// Python 模块定义
#[pymodule]
fn bm25_jieba(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BM25>()?;
    Ok(())
}
