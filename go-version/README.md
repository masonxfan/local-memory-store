# Local Memory Store — Go 版

单二进制文件 + ONNX Runtime，无需 Python 环境。

## 架构

```
memory-store (Go 二进制)
    ├── ONNX Runtime (动态库) — 模型执行引擎
    ├── all-MiniLM-L6-v2 (ONNX 模型) — 文本 → 384维向量
    └── SQLite — 向量 + 原文持久化存储
```

## 目录结构

```
go-version/
├── main.go          ← 主程序（embedder + vector store + CLI）
├── go.mod
├── setup.ps1        ← Windows 一键安装
├── setup.sh         ← Linux/macOS 一键安装
└── models/          ← 运行时需要（setup 脚本自动下载）
    ├── model.onnx   ← all-MiniLM-L6-v2 ONNX 模型
    └── vocab.txt    ← WordPiece 词表
```

## 一键安装

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

### Linux / macOS

```bash
chmod +x setup.sh && ./setup.sh
```

脚本会自动：
1. 检查/安装 Go 编译环境
2. 下载 ONNX Runtime 动态库 (~50MB)
3. 下载 all-MiniLM-L6-v2 模型 + 词表
4. 编译 Go 程序
5. 运行验证测试

## 手动构建

```bash
# 1. 下载 ONNX Runtime
# https://github.com/microsoft/onnxruntime/releases
# 解压，设置环境变量：
export ONNXRUNTIME_LIB_PATH=/path/to/libonnxruntime.so

# 2. 下载模型
mkdir -p models
# model.onnx: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# 用 optimum 导出：optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 models/
# 或直接下载预导出版本

# 3. 编译
cd go-version
CGO_ENABLED=1 go build -o memory-store .

# 4. 运行
./memory-store add --user xing --text "测试记忆"
./memory-store search --query "测试"
```

## 交叉编译

```bash
# Windows
GOOS=windows GOARCH=amd64 CGO_ENABLED=1 go build -o memory-store.exe .

# Linux
GOOS=linux GOARCH=amd64 CGO_ENABLED=1 go build -o memory-store .

# macOS (ARM)
GOOS=darwin GOARCH=arm64 CGO_ENABLED=1 go build -o memory-store .
```

注意：每个平台需要对应的 ONNX Runtime 动态库。

## 使用方法

```bash
# 存储
./memory-store add --user xing --text "决定用 Go 重写 memory store"

# 搜索
./memory-store search --query "重写" --limit 5

# 统计
./memory-store stats
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ONNXRUNTIME_LIB_PATH` | ONNX Runtime 动态库路径 | 自动查找 |
| `MEMORY_DB_PATH` | SQLite 数据库路径 | `./memories.db` |
| `MEMORY_MODEL_DIR` | 模型目录路径 | `./models/` |

## vs Python 版

| | Python 版 | Go 版 |
|--|----------|-------|
| 依赖 | Python + pip + venv | 单二进制 + 动态库 |
| 启动速度 | ~2秒 | ~50ms |
| 分发 | 需要安装脚本 | 复制文件即可 |
| 向量库 | ChromaDB (HNSW) | SQLite (暴力搜索) |
| 搜索性能 | 百万级毫秒 | 万级毫秒，十万级秒级 |
| 适用场景 | 快速原型 | 生产部署、嵌入其他系统 |

> Go 版使用 SQLite 暴力搜索（逐条计算余弦相似度），对于个人使用（<10万条）完全够用。
> 如需百万级，可以后续加 HNSW 索引库（如 `github.com/viterin/vek`）。
