# Local Memory Store

本地 AI Agent 对话记忆系统 — 基于 ChromaDB 向量数据库 + 本地 ONNX Embedding 模型，零成本运行。

## 它是什么

一个轻量级的语义记忆存储服务，让 AI Agent 能够：
- **存储**对话中的重要信息（决策、结论、调研结果）
- **语义搜索**历史记忆（不是关键词匹配，而是理解意思）
- **纯本地运行**，不调外部 API，不花钱

## 架构

```
用户对话 → Agent 判断是否重要 → 调用 memory_service.py
                                        ↓
                               ONNX Embedding 模型
                            (all-MiniLM-L6-v2, 79MB)
                                        ↓
                              文本 → 384维向量
                                        ↓
                              ChromaDB 存储到本地磁盘
                                        ↓
                              搜索时：query → 向量 → 余弦相似度 → 返回最相关的记忆
```

## 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| 向量数据库 | ChromaDB | 轻量、嵌入式、Python 原生、持久化到本地文件 |
| Embedding 模型 | all-MiniLM-L6-v2 (ONNX) | 22M 参数、79MB、CPU 毫秒级推理、效果够用 |
| 索引算法 | HNSW (Hierarchical Navigable Small World) | ChromaDB 默认，百万级数据毫秒级搜索 |
| 相似度度量 | 余弦相似度 (Cosine Similarity) | 语义搜索标准选择 |

### 为什么不用 OpenAI Embedding？

- 本地 ONNX 模型：免费、无网络依赖、更快、隐私安全
- OpenAI text-embedding-3-small：效果更好，但需要 API key + 网络 + 费用（虽然很便宜）
- 对话记忆场景，本地模型效果完全够用

### 为什么不用 Mem0/Zep/Letta？

这些是更完整的 Agent 记忆框架，但对于基础的"存+搜"需求，自建更轻量：
- **Mem0**：自动提取记忆 + 向量存储，功能全但依赖多
- **Zep**：时序感知 + 实体提取，适合复杂场景
- **Letta (MemGPT)**：Agent 自主管理记忆，概念先进但较重

当前方案 = 最小可用版本，未来可以按需升级到这些框架。

## 工作原理

### Embedding 过程（文本 → 向量）

```
"决定用 ChromaDB"
    ↓ 分词 (Tokenization)
["决", "定", "用", "Chroma", "DB"]
    ↓ 查 Embedding 表（每个 token → 初始向量）
    ↓ 6层 Transformer 编码（Self-Attention 融合上下文）
    ↓ Mean Pooling（多个词向量 → 一个句子向量）
    ↓ 归一化
[0.012, 0.025, -0.037, ...] （384维向量）
```

### 搜索过程（语义匹配）

```
搜索 "之前选了什么数据库"
    ↓ 同样转成 384维向量
    ↓ HNSW 索引快速找最近邻（不是逐条比较）
    ↓ 计算余弦相似度
    ↓ 返回 Top N 最相关的记忆 + 原文
```

### 存储策略

不是每条对话都存，而是存**浓缩后的重要信息**：

| ✅ 存 | ❌ 不存 |
|-------|---------|
| 决策和结论 | 闲聊寒暄 |
| 调研结果 | 过程性提问 |
| 用户偏好 | "好的""懂了" |
| 任务完成记录 | 中间试错过程 |

## 安装与设置

### 环境要求

- Python 3.10+
- 磁盘空间：~200MB（ChromaDB + ONNX 模型）
- 内存：~200MB（模型加载时）
- 无需 GPU，CPU 即可

### 一键安装

```bash
git clone https://github.com/masonxfan/local-memory-store.git
cd local-memory-store
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

**Linux / macOS:**
```bash
chmod +x setup.sh && ./setup.sh
```

脚本会自动：检查 Python → 创建虚拟环境 → 安装 ChromaDB → 下载 ONNX 模型 → 运行验证测试

### 手动安装

如果你不想用一键脚本：

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/Mac: source .venv/bin/activate
pip install chromadb
python memory_service.py stats  # 首次运行会下载模型 (~79MB)
```

### 目录结构

```
local-memory-store/
├── memory_service.py    ← 主程序
├── chroma_db/           ← 向量数据（自动创建，已 gitignore）
│   ├── chroma.sqlite3   ← 元数据 + 原文
│   └── *.bin            ← HNSW 向量索引
└── README.md
```

### ChromaDB 数据存储说明

ChromaDB 使用 `PersistentClient` 模式，数据存在本地 `chroma_db/` 目录：

- **chroma.sqlite3**：SQLite 数据库，存储原始文本、metadata、ID 映射
- **HNSW 索引文件**：二进制文件，存储向量索引结构
- 所有数据持久化在磁盘，重启不丢失
- 备份：直接复制整个 `chroma_db/` 目录即可
- 迁移：把 `chroma_db/` 复制到另一台机器，装好 ChromaDB 就能用

### 可选：ChromaDB Server 模式（多客户端访问）

如果需要多个 Agent/程序同时访问同一个向量库：

```bash
# 启动 ChromaDB 服务（默认端口 8000）
chroma run --host 0.0.0.0 --port 8000 --path ./chroma_db

# 客户端连接时改用 HttpClient：
# client = chromadb.HttpClient(host="localhost", port=8000)
```

## 使用

### 存储记忆

```bash
python memory_service.py add \
  --user xing \
  --text "决定用 ChromaDB 做本地向量存储，配合 ONNX embedding" \
  --metadata '{"topic": "infrastructure", "type": "decision"}'
```

### 语义搜索

```bash
python memory_service.py search \
  --user xing \
  --query "之前选了什么数据库" \
  --limit 5
```

### 查看统计

```bash
python memory_service.py stats
```

## 数据存储

向量数据存储在 `chroma_db/` 目录（已 gitignore），包含：
- 向量索引（HNSW 图结构）
- 原始文本
- Metadata（用户、时间戳、标签等）

容量：百万条记忆约占几百 MB，个人使用绰绰有余。

## 扩展方向

### 多用户共享（团队知识库）

```
ChromaDB server 模式: chroma run --host 0.0.0.0 --port 8000
每个 Agent 通过 HTTP 连接同一个 ChromaDB
用 metadata 的 visibility 字段控制权限（private/team/org）
```

### 记忆策略升级

- MD 文件 = 核心记忆（每次 session 必读）
- 向量库 = 扩展记忆（按需搜索）
- 类似人脑：工作记忆 vs 长期回忆

### 自动化

- Compaction hook：对话压缩时自动存入向量库
- 隐式触发：检测到"之前聊过"等关键词时自动搜索
- 定期整理：合并重复记忆，清理过时信息

## License

MIT
