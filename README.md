# BM25-Jieba 中文文本搜索

[![PyPI version](https://img.shields.io/pypi/v/bm25-jieba.svg)](https://pypi.org/project/bm25-jieba/)
[![Python](https://img.shields.io/pypi/pyversions/bm25-jieba.svg)](https://pypi.org/project/bm25-jieba/)
[![License](https://img.shields.io/pypi/l/bm25-jieba.svg)](https://github.com/twn39/bm25-jieba/blob/main/LICENSE)
[![Build](https://github.com/twn39/bm25-jieba/actions/workflows/release.yml/badge.svg)](https://github.com/twn39/bm25-jieba/actions)
[![Downloads](https://img.shields.io/pypi/dm/bm25-jieba.svg)](https://pypi.org/project/bm25-jieba/)

基于 Rust + PyO3 的高性能 BM25 中文文本搜索库，使用 jieba-rs 进行中文分词。

## 特性

- 🚀 **高性能**: Rust 实现，采用 **倒排索引** + **Block-Max WAND** 算法加速，比纯 Python 快数倍
- 💾 **持久化**: 支持存取索引到磁盘 (MessagePack 格式)，无需重复训练
- 🔤 **中文分词**: 内置 jieba-rs 分词器
- 🎯 **精确搜索**: 经典 BM25 算法
- 🔠 **大小写混合**: 支持大小写不敏感搜索
- 🐍 **Python 3.11 ~ 3.14**: 支持最新 Python 版本

## 安装

```bash
# 开发模式安装
uv run maturin develop

# 或构建 wheel
maturin build --release
pip install target/wheels/*.whl
```

## 快速开始

```python
from bm25_jieba import BM25

# 准备文档
documents = [
    "Python是一种广泛使用的高级编程语言",
    "机器学习是人工智能的一个分支",
    "深度学习是机器学习的子领域",
]

# 创建并训练模型
bm25 = BM25(k1=1.5, b=0.75)
bm25.fit(documents)

# 搜索
results = bm25.search("机器学习", top_k=3)
for doc_idx, score in results:
    print(f"[{score:.4f}] {documents[doc_idx]}")

# 保存模型 (无需重复训练)
bm25.save("bm25_model.bin")

# 加载模型
loaded_bm25 = BM25.load("bm25_model.bin")
```

## API 参考

### `BM25(k1=1.5, b=0.75, lowercase=False)`

创建 BM25 实例。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `k1` | float | 1.5 | 词频饱和参数 |
| `b` | float | 0.75 | 文档长度归一化参数 |
| `lowercase` | bool | False | 是否将文本转换为小写（大小写不敏感） |

### `fit(documents: list[str])`

使用文档语料库训练模型。

### `search(query: str, top_k: int = None) -> list[tuple[int, float]]`

搜索最相关的文档，返回 `(文档索引, 分数)` 列表。

### `save(path: str)`
保存当前索引和配置到文件 (MessagePack 格式)。

### `load(path: str) -> BM25`
从文件加载 BM25 模型。

### `get_scores(query: str) -> list[float]`
获取所有文档的 BM25 分数。

## 开发

```bash
# 安装依赖
uv sync

# 编译并安装
uv run maturin develop

# 运行测试
uv run pytest

# 运行示例
uv run python examples/demo.py
```

## 技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| PyO3 | 0.27.2 | Rust-Python 绑定 |
| maturin | 1.11.5 | 构建工具 |
| jieba-rs | 0.8.1 | 中文分词 |

## 性能测试

在 Apple M1 上测试 (10,000 文档，每文档约 100 字)：

| 测试项 | 结果 |
|--------|------|
| 索引速度 | ~37,000 docs/s |
| 搜索 QPS | ~1,000,000 QPS |
| 搜索延迟 | ~0.001ms |

> *注：得益于 Block-Max WAND 算法的剪枝优化，搜索性能有数量级提升。*

```bash
# 运行性能测试
uv run python tests/benchmark.py
```

## 算法验证

测试语料库：19 个文档，6 个查询

| 验证项 | 结果 | 说明 |
|--------|------|------|
| 公式正确性 | ✅ | 手动计算与实现一致 |
| 排序一致性 | ✅ | 与 rank-bm25 排序完全一致 |
| 绝对分数 | ⚠️ | 因 IDF +1 修正略有差异（符合预期） |

```bash
# 运行验证
uv sync --group validation
uv run python tests/validate.py
```

## 算法实现说明

本实现采用带 `+1` 修正的 IDF 公式：

```
IDF(t) = ln((N - df + 0.5) / (df + 0.5) + 1)
```

与标准 BM25Okapi 的区别：

| 公式 | 特点 |
|------|------|
| 标准: `ln((N-df+0.5)/(df+0.5))` | 可能产生负 IDF |
| 本实现: `ln(...+1)` | 保证 IDF ≥ 0 |

**影响**：
- ✅ **排序一致** - 与 rank-bm25 等标准实现排序结果相同
- ⚠️ **绝对分数不同** - 因 `+1` 修正，分数值略有差异
- ✅ **数值稳定** - 无负值，无需额外处理

这种变体在只关心**相对排序**（而非绝对分数）的场景下完全适用。

## License

MIT
