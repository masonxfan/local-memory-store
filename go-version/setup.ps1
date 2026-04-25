# Go 版 Local Memory Store 一键安装 (Windows)
$ErrorActionPreference = "Stop"

$ORT_VERSION = "1.20.1"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$MODEL_DIR = Join-Path $SCRIPT_DIR "models"
$ORT_DIR = Join-Path $SCRIPT_DIR "onnxruntime"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Local Memory Store (Go) - 一键安装" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Go
Write-Host "[1/5] 检查 Go..." -ForegroundColor Yellow
try {
    $goVer = go version 2>&1
    Write-Host "  ✓ $goVer" -ForegroundColor Green
} catch {
    Write-Host "  ✗ 未找到 Go，请安装: https://go.dev/dl/" -ForegroundColor Red
    exit 1
}

# 2. Download ONNX Runtime
Write-Host "[2/5] 下载 ONNX Runtime v${ORT_VERSION}..." -ForegroundColor Yellow
$ortLib = Join-Path $ORT_DIR "lib\onnxruntime.dll"
if (Test-Path $ortLib) {
    Write-Host "  ✓ 已存在，跳过" -ForegroundColor Green
} else {
    New-Item -ItemType Directory -Path $ORT_DIR -Force | Out-Null
    $ortUrl = "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-win-x64-${ORT_VERSION}.zip"
    $ortZip = Join-Path $env:TEMP "onnxruntime.zip"
    Write-Host "  下载: $ortUrl" -ForegroundColor Gray
    Invoke-WebRequest -Uri $ortUrl -OutFile $ortZip
    Expand-Archive -Path $ortZip -DestinationPath $env:TEMP -Force
    $extracted = Join-Path $env:TEMP "onnxruntime-win-x64-${ORT_VERSION}"
    Copy-Item -Path "$extracted\*" -Destination $ORT_DIR -Recurse -Force
    Remove-Item $ortZip -Force
    Write-Host "  ✓ ONNX Runtime 下载完成" -ForegroundColor Green
}

# 3. Download model
Write-Host "[3/5] 下载 all-MiniLM-L6-v2 模型..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $MODEL_DIR -Force | Out-Null
$modelPath = Join-Path $MODEL_DIR "model.onnx"
$vocabPath = Join-Path $MODEL_DIR "vocab.txt"

if (Test-Path $modelPath) {
    Write-Host "  ✓ 模型已存在，跳过" -ForegroundColor Green
} else {
    Write-Host "  下载 model.onnx..." -ForegroundColor Gray
    Invoke-WebRequest -Uri "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx" -OutFile $modelPath
    Write-Host "  ✓ 模型下载完成" -ForegroundColor Green
}

if (Test-Path $vocabPath) {
    Write-Host "  ✓ 词表已存在，跳过" -ForegroundColor Green
} else {
    Write-Host "  下载 vocab.txt..." -ForegroundColor Gray
    Invoke-WebRequest -Uri "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt" -OutFile $vocabPath
    Write-Host "  ✓ 词表下载完成" -ForegroundColor Green
}

# 4. Build
Write-Host "[4/5] 编译 Go 程序..." -ForegroundColor Yellow
Push-Location $SCRIPT_DIR
$env:CGO_ENABLED = "1"
$env:CGO_CFLAGS = "-I$ORT_DIR\include"
$env:CGO_LDFLAGS = "-L$ORT_DIR\lib"
go build -o memory-store.exe .
Pop-Location
Write-Host "  ✓ 编译完成: memory-store.exe" -ForegroundColor Green

# 5. Test
Write-Host "[5/5] 验证..." -ForegroundColor Yellow
$env:ONNXRUNTIME_LIB_PATH = $ortLib
$env:MEMORY_MODEL_DIR = $MODEL_DIR
$env:PATH = "$ORT_DIR\lib;$env:PATH"

$result = & (Join-Path $SCRIPT_DIR "memory-store.exe") add --user test --text "安装验证 - Go 版 Local Memory Store" 2>&1
if ($result -match '"status": "ok"') {
    Write-Host "  ✓ 验证通过！" -ForegroundColor Green
} else {
    Write-Host "  ✗ 验证失败: $result" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "使用前将 ONNX Runtime 加入 PATH:" -ForegroundColor Yellow
Write-Host "  `$env:PATH = `"$ORT_DIR\lib;`$env:PATH`"" -ForegroundColor White
Write-Host "  `$env:ONNXRUNTIME_LIB_PATH = `"$ortLib`"" -ForegroundColor White
Write-Host "  `$env:MEMORY_MODEL_DIR = `"$MODEL_DIR`"" -ForegroundColor White
Write-Host ""
Write-Host "使用:" -ForegroundColor Yellow
Write-Host '  .\memory-store.exe add --user your_name --text "要记住的内容"' -ForegroundColor White
Write-Host '  .\memory-store.exe search --query "搜索内容"' -ForegroundColor White
Write-Host '  .\memory-store.exe stats' -ForegroundColor White
Write-Host ""
