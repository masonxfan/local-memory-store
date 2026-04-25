#!/usr/bin/env bash
# Local Memory Store - 一键安装脚本 (Linux/macOS)
# 用法: chmod +x setup.sh && ./setup.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "========================================"
echo "  Local Memory Store - 一键安装"
echo "  ChromaDB + ONNX Embedding"
echo "========================================"
echo ""

# 1. 检查 Python
echo "[1/4] 检查 Python..."
if ! command -v python3 &>/dev/null; then
    echo "  ✗ 未找到 python3，请先安装 Python 3.10+"
    exit 1
fi
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MINOR" -lt 10 ]; then
    echo "  ✗ 需要 Python 3.10+，当前版本: $PY_VERSION"
    exit 1
fi
echo "  ✓ Python $PY_VERSION"

# 2. 创建虚拟环境
if [ -d "$SCRIPT_DIR/.venv" ]; then
    echo "[2/4] 虚拟环境已存在，跳过"
else
    echo "[2/4] 创建虚拟环境..."
    python3 -m venv "$SCRIPT_DIR/.venv"
    echo "  ✓ 虚拟环境创建完成"
fi
source "$SCRIPT_DIR/.venv/bin/activate"

# 3. 安装依赖
echo "[3/4] 安装 ChromaDB（含 ONNX embedding）..."
pip install chromadb -q
echo "  ✓ ChromaDB 安装完成"

# 4. 初始化
echo "[4/4] 初始化向量数据库 + 下载 Embedding 模型 (~79MB)..."
python3 "$SCRIPT_DIR/memory_service.py" stats 2>/dev/null
echo "  ✓ 初始化完成"

# 验证
echo ""
echo "========================================"
echo "  安装完成！"
echo "========================================"
echo ""
echo "使用方法:"
echo ""
echo "  # 先激活虚拟环境（每次打开新终端时）"
echo "  source .venv/bin/activate"
echo ""
echo '  # 存储记忆'
echo '  python memory_service.py add --user your_name --text "要记住的内容"'
echo ""
echo '  # 语义搜索'
echo '  python memory_service.py search --user your_name --query "搜索内容"'
echo ""
echo '  # 查看统计'
echo '  python memory_service.py stats'
echo ""

# 测试
echo "运行验证测试..."
RESULT=$(python3 "$SCRIPT_DIR/memory_service.py" add --user test --text "安装验证测试 - Local Memory Store 初始化成功" --metadata '{"type": "setup"}' 2>&1)
if echo "$RESULT" | grep -q '"status": "ok"'; then
    echo "  ✓ 验证通过！"
else
    echo "  ✗ 验证失败: $RESULT"
fi
echo ""
