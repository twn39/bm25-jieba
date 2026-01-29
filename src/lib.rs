//! BM25 中文文本搜索算法 Rust 实现
//!
//! 使用 jieba-rs 进行中文分词，基于倒排索引和 Block-Max WAND 算法实现高效检索
//! 支持索引持久化

use jieba_rs::Jieba;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::sync::LazyLock;

/// 全局 Jieba 实例（线程安全，延迟初始化）
static JIEBA: LazyLock<Jieba> = LazyLock::new(Jieba::new);

/// 常量定义
const BLOCK_SIZE: usize = 128; // BMW 算法块大小

/// 倒排索引块
#[derive(Debug, Serialize, Deserialize)]
struct Block {
    max_score: f64,     // 块内最大可能得分 (BMW 优化核心)
    last_doc_id: u32,   // 块内最后一个文档ID (Skip List)
    doc_ids: Vec<u32>,  // 文档ID列表
    freqs: Vec<u32>,    // 词频列表
    doc_lens: Vec<u32>, // 文档长度列表 (用于计算 BM25)
}

/// 倒排列表
#[derive(Debug, Default, Serialize, Deserialize)]
struct InvertedList {
    blocks: Vec<Block>,
    doc_count: usize, // 包含该词的文档总数
}

/// 候选文档得分（用于 Top-K 堆）
#[derive(PartialEq)]
struct ScoredDoc {
    score: f64,
    doc_id: u32,
}

// 实现 Ord trait 使得 BinaryHeap 成为最小堆（用于维护 Top-K）
impl Eq for ScoredDoc {}
impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.score.partial_cmp(&self.score)
    }
}
impl Ord for ScoredDoc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order for score: Higher score is "Smaller" (so it stays in Heap, Low score is popped)
        // If scores are equal, prefer smaller doc_id (Smaller ID is "Smaller", Larger ID is "Greater" -> popped)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.doc_id.cmp(&other.doc_id))
    }
}

/// BM25 中文文本搜索算法
#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct BM25 {
    k1: f64,
    b: f64,
    lowercase: bool,
    corpus_size: usize,
    avgdl: f64,
    index: HashMap<String, InvertedList>,
    doc_lengths: Vec<u32>, // 全局文档长度
    doc_ids: Vec<u64>,     // 映射: 内部ID(usize) -> 外部ID(u64)
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
            index: HashMap::new(),
            doc_lengths: Vec::new(),
            doc_ids: Vec::new(),
        }
    }

    /// 使用文档语料库训练 BM25 模型
    ///
    /// documents: 文档内容列表
    /// ids: 可选的文档 ID 列表 (必须与 documents 长度一致)
    #[pyo3(signature = (documents, ids=None))]
    pub fn fit(&mut self, documents: Vec<String>, ids: Option<Vec<u64>>) -> PyResult<()> {
        if let Some(ref external_ids) = ids {
            if external_ids.len() != documents.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "documents and ids must have the same length",
                ));
            }
        }

        self.corpus_size = documents.len();
        self.index.clear();
        self.doc_lengths.clear();
        self.doc_ids.clear();

        // 初始化 ID 映射
        if let Some(external_ids) = ids {
            self.doc_ids = external_ids;
        } else {
            self.doc_ids = (0..self.corpus_size as u64).collect();
        }

        let mut temp_index: HashMap<String, Vec<(u32, u32, u32)>> = HashMap::new();
        let mut total_length: u64 = 0;

        // 1. 分词并收集 Postings
        for (doc_id, doc) in documents.iter().enumerate() {
            let doc_id = doc_id as u32;
            let tokens = self.tokenize(doc);
            let doc_len = tokens.len() as u32;

            self.doc_lengths.push(doc_len);
            total_length += doc_len as u64;

            let mut freq_map: HashMap<String, u32> = HashMap::new();
            for token in tokens {
                *freq_map.entry(token).or_insert(0) += 1;
            }

            for (term, freq) in freq_map {
                temp_index
                    .entry(term)
                    .or_default()
                    .push((doc_id, freq, doc_len));
            }
        }

        self.avgdl = if self.corpus_size > 0 {
            total_length as f64 / self.corpus_size as f64
        } else {
            0.0
        };

        // 2. 构建 Block-Max 倒排索引
        for (term, mut postings) in temp_index {
            postings.sort_by_key(|k| k.0); // 按 doc_id 排序

            let mut inverted_list = InvertedList {
                doc_count: postings.len(),
                blocks: Vec::new(),
            };

            for chunk in postings.chunks(BLOCK_SIZE) {
                let mut block = Block {
                    max_score: 0.0,
                    last_doc_id: chunk.last().unwrap().0,
                    doc_ids: Vec::with_capacity(chunk.len()),
                    freqs: Vec::with_capacity(chunk.len()),
                    doc_lens: Vec::with_capacity(chunk.len()),
                };

                let idf = self.calc_idf(postings.len());

                for &(doc_id, freq, doc_len) in chunk {
                    block.doc_ids.push(doc_id);
                    block.freqs.push(freq);
                    block.doc_lens.push(doc_len);

                    // 计算该文档的 BM25 分数，更新 Block Max Score
                    let score = self.calc_bm25_score(idf, freq, doc_len);
                    if score > block.max_score {
                        block.max_score = score;
                    }
                }
                inverted_list.blocks.push(block);
            }

            self.index.insert(term, inverted_list);
        }
        Ok(())
    }

    /// 搜索与查询最相关的文档 (Block-Max WAND)
    /// 返回: List[(doc_id, score)]，其中 doc_id 是外部 ID (u64)
    #[pyo3(signature = (query, top_k=None))]
    pub fn search(&self, query: &str, top_k: Option<usize>) -> Vec<(u64, f64)> {
        let k = top_k.unwrap_or(10); // 默认 Top 10
        let query_tokens = self.tokenize(query);
        let mut heap = BinaryHeap::new(); // 最小堆，保存 Top-K

        // 收集所有相关词的 Block 迭代器
        let mut cursors: Vec<BlockCursor> = Vec::new();
        for token in query_tokens {
            if let Some(inv_list) = self.index.get(&token) {
                if !inv_list.blocks.is_empty() {
                    let idf = self.calc_idf(inv_list.doc_count);
                    cursors.push(BlockCursor::new(inv_list, idf));
                }
            }
        }

        if cursors.is_empty() {
            return Vec::new();
        }

        // 简化的 BMW/WAND 逻辑
        let mut active_cursors: Vec<&mut BlockCursor> = cursors.iter_mut().collect();

        loop {
            // 1. 找出当前所有 cursor 中最小的 doc_id
            let mut min_doc_id = u32::MAX;
            let mut all_finished = true;

            for cursor in &active_cursors {
                if let Some(doc_id) = cursor.curr_doc_id() {
                    all_finished = false;
                    if doc_id < min_doc_id {
                        min_doc_id = doc_id;
                    }
                }
            }

            if all_finished {
                break;
            }

            // 2. 剪枝检查
            // TODO: WAND threshold check

            // 3. 计算 min_doc_id 的准确分数
            let mut score = 0.0;
            let mut advanced_any = false;

            for cursor in &mut active_cursors {
                if let Some(doc_id) = cursor.curr_doc_id() {
                    if doc_id == min_doc_id {
                        score += cursor.curr_score(self.k1, self.b, self.avgdl);
                        cursor.advance();
                        advanced_any = true;
                    }
                }
            }

            if !advanced_any {
                break;
            }

            // 4. 更新堆
            if heap.len() < k {
                heap.push(ScoredDoc {
                    score,
                    doc_id: min_doc_id,
                });
            } else if let Some(min_node) = heap.peek() {
                if score > min_node.score {
                    heap.pop();
                    heap.push(ScoredDoc {
                        score,
                        doc_id: min_doc_id,
                    });
                }
            }
        }

        // 结果排序 (分数降序)
        let results: Vec<(u64, f64)> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|d| {
                // 映射回外部 ID
                let internal_id = d.doc_id as usize;
                let external_id = if internal_id < self.doc_ids.len() {
                    self.doc_ids[internal_id]
                } else {
                    internal_id as u64 // Fallback, shout not happen
                };
                (external_id, d.score)
            })
            .collect();

        results
    }

    /// 获取所有文档的 BM25 分数
    pub fn get_scores(&self, query: &str) -> Vec<f64> {
        let mut scores = vec![0.0; self.corpus_size];
        let query_tokens = self.tokenize(query);

        for token in query_tokens {
            if let Some(inv_list) = self.index.get(&token) {
                // 计算 idf (注意：inv_list.doc_count 存储包含词 t 的文档总数 n(t))
                let idf = self.calc_idf(inv_list.doc_count);

                for block in &inv_list.blocks {
                    for i in 0..block.doc_ids.len() {
                        let doc_id = block.doc_ids[i] as usize;
                        let freq = block.freqs[i];
                        let doc_len = block.doc_lens[i];

                        scores[doc_id] += self.calc_bm25_score(idf, freq, doc_len);
                    }
                }
            }
        }
        scores
    }

    /// 保存索引到文件 (MessagePack)
    pub fn save(&self, path: &str) -> PyResult<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        rmp_serde::encode::write(&mut writer, self)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(())
    }

    /// 从文件加载索引 (MessagePack)
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let bm25: BM25 = rmp_serde::decode::from_read(reader)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(bm25)
    }
}

impl BM25 {
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

    fn calc_idf(&self, matched_docs: usize) -> f64 {
        let numerator = self.corpus_size as f64 - matched_docs as f64 + 0.5;
        let denominator = matched_docs as f64 + 0.5;
        (numerator / denominator + 1.0).ln()
    }

    fn calc_bm25_score(&self, idf: f64, freq: u32, doc_len: u32) -> f64 {
        let freq = freq as f64;
        let numerator = freq * (self.k1 + 1.0);
        let denominator = freq + self.k1 * (1.0 - self.b + self.b * doc_len as f64 / self.avgdl);
        idf * numerator / denominator
    }
}

/// 辅助游标，用于遍历倒排索引
struct BlockCursor<'a> {
    list: &'a InvertedList,
    block_idx: usize,
    in_block_idx: usize,
    idf: f64,
}

impl<'a> BlockCursor<'a> {
    fn new(list: &'a InvertedList, idf: f64) -> Self {
        BlockCursor {
            list,
            block_idx: 0,
            in_block_idx: 0,
            idf,
        }
    }

    fn curr_doc_id(&self) -> Option<u32> {
        if self.block_idx >= self.list.blocks.len() {
            return None;
        }
        let block = &self.list.blocks[self.block_idx];
        if self.in_block_idx >= block.doc_ids.len() {
            return None;
        }
        Some(block.doc_ids[self.in_block_idx])
    }

    fn curr_score(&self, k1: f64, b: f64, avgdl: f64) -> f64 {
        let block = &self.list.blocks[self.block_idx];
        let freq = block.freqs[self.in_block_idx] as f64;
        let doc_len = block.doc_lens[self.in_block_idx] as f64;

        let numerator = freq * (k1 + 1.0);
        let denominator = freq + k1 * (1.0 - b + b * doc_len / avgdl);
        self.idf * numerator / denominator
    }

    fn advance(&mut self) {
        if self.block_idx >= self.list.blocks.len() {
            return;
        }

        self.in_block_idx += 1;

        // 如果当前块遍历完了，移动到下一个块
        if self.in_block_idx >= self.list.blocks[self.block_idx].doc_ids.len() {
            self.block_idx += 1;
            self.in_block_idx = 0;

            // TODO: 这里可以加入 Block 级剪枝逻辑
            // if self.list.blocks[self.block_idx].max_score < threshold { skip }
        }
    }
}

/// Python 模块定义
#[pymodule]
fn bm25_jieba(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BM25>()?;
    Ok(())
}
