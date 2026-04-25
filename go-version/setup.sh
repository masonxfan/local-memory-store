#!/usr/bin/env bash
# Go 版 Local Memory Store 一键安装 (Linux/macOS)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models"
ORT_VERSION="1.20.1"

echo ""
echo "========================================"
echo "  Local Memory Store (Go) - 一键安装"
echo "========================================"
echo ""

# Detect OS and arch
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case "$ARCH" in
    x86_64) ARCH="x64" ;;
    aarch64|arm64) ARCH="aarch64" ;;
esac

# 1. Check Go
echo "[1/5] 检查 Go..."
if ! command -v go &>/dev/null; then
    echo "  ✗ 未找到 Go，请安装: https://go.dev/dl/"
    exit 1
fi
echo "  ✓ $(go version)"

# 2. Download ONNX Runtime
echo "[2/5] 下载 ONNX Runtime v${ORT_VERSION}..."
ORT_DIR="$SCRIPT_DIR/onnxruntime"
if [ -f "$ORT_DIR/lib/libonnxruntime.so" ] || [ -f "$ORT_DIR/lib/libonnxruntime.dylib" ]; then
    echo "  ✓ 已存在，跳过"
else
    mkdir -p "$ORT_DIR"
    if [ "$OS" = "darwin" ]; then
        ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-osx-${ARCH}-${ORT_VERSION}.tgz"
        LIB_NAME="libonnxruntime.dylib"
    else
        ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ARCH}-${ORT_VERSION}.tgz"
        LIB_NAME="libonnxruntime.so"
    fi
    echo "  下载: $ORT_URL"
    curl -L "$ORT_URL" | tar xz -C "$ORT_DIR" --strip-components=1
    echo "  ✓ ONNX Runtime 下载完成"
fi

# 3. Download model
echo "[3/5] 下载 all-MiniLM-L6-v2 模型..."
mkdir -p "$MODEL_DIR"
if [ -f "$MODEL_DIR/model.onnx" ]; then
    echo "  ✓ 模型已存在，跳过"
else
    echo "  下载 model.onnx..."
    curl -L "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx" -o "$MODEL_DIR/model.onnx"
    echo "  ✓ 模型下载完成"
fi
if [ -f "$MODEL_DIR/vocab.txt" ]; then
    echo "  ✓ 词表已存在，跳过"
else
    echo "  下载 vocab.txt..."
    curl -L "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt" -o "$MODEL_DIR/vocab.txt"
    echo "  ✓ 词表下载完成"
fi

# 4. Build
echo "[4/5] 编译 Go 程序..."
cd "$SCRIPT_DIR"
export CGO_ENABLED=1
export CGO_CFLAGS="-I$ORT_DIR/include"
export CGO_LDFLAGS="-L$ORT_DIR/lib"
go mod tidy
go build -o memory-store .
echo "  ✓ 编译完成: memory-store"

# 5. Test
echo "[5/5] 验证..."
export LD_LIBRARY_PATH="$ORT_DIR/lib:$LD_LIBRARY_PATH"
export DYLD_LIBRARY_PATH="$ORT_DIR/lib:$DYLD_LIBRARY_PATH"
export ONNXRUNTIME_LIB_PATH="$ORT_DIR/lib/$LIB_NAME"
export MEMORY_MODEL_DIR="$MODEL_DIR"

RESULT=$(./memory-store add --user test --text "安装验证 - Go 版 Local Memory Store" 2>&1)
if echo "$RESULT" | grep -q '"status": "ok"'; then
    echo "  ✓ 验证通过！"
else
    echo "  ✗ 验证失败: $RESULT"
    exit 1
fi

echo ""
echo "========================================"
echo "  安装完成！"
echo "========================================"
echo ""
echo "使用前设置环境变量（加到 .bashrc/.zshrc）:"
echo ""
echo "  export LD_LIBRARY_PATH=$ORT_DIR/lib:\$LD_LIBRARY_PATH"
echo "  export ONNXRUNTIME_LIB_PATH=$ORT_DIR/lib/$LIB_NAME"
echo "  export MEMORY_MODEL_DIR=$MODEL_DIR"
echo ""
echo "使用:"
echo '  ./memory-store add --user your_name --text "要记住的内容"'
echo '  ./memory-store search --query "搜索内容"'
echo '  ./memory-store stats'
echo ""
