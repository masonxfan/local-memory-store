# Local Memory Store - Windows 一键安装脚本
# 用法: 右键 → 用 PowerShell 运行，或在终端执行:
#   powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Local Memory Store - 一键安装" -ForegroundColor Cyan
Write-Host "  ChromaDB + ONNX Embedding" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. 检查 Python
Write-Host "[1/4] 检查 Python..." -ForegroundColor Yellow
try {
    $pyVersion = python --version 2>&1
    if ($pyVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
            Write-Host "  ✗ 需要 Python 3.10+，当前版本: $pyVersion" -ForegroundColor Red
            Write-Host "  下载: https://www.python.org/downloads/" -ForegroundColor Gray
            exit 1
        }
        Write-Host "  ✓ $pyVersion" -ForegroundColor Green
    }
} catch {
    Write-Host "  ✗ 未找到 Python，请先安装 Python 3.10+" -ForegroundColor Red
    Write-Host "  下载: https://www.python.org/downloads/" -ForegroundColor Gray
    Write-Host "  安装时勾选 'Add Python to PATH'" -ForegroundColor Gray
    exit 1
}

# 2. 创建虚拟环境
$venvPath = Join-Path $PSScriptRoot ".venv"
if (Test-Path $venvPath) {
    Write-Host "[2/4] 虚拟环境已存在，跳过" -ForegroundColor Yellow
} else {
    Write-Host "[2/4] 创建虚拟环境..." -ForegroundColor Yellow
    python -m venv $venvPath
    Write-Host "  ✓ 虚拟环境创建完成" -ForegroundColor Green
}

# 激活虚拟环境
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
. $activateScript

# 3. 安装依赖
Write-Host "[3/4] 安装 ChromaDB（含 ONNX embedding）..." -ForegroundColor Yellow
pip install chromadb --quiet 2>&1 | Out-Null
Write-Host "  ✓ ChromaDB 安装完成" -ForegroundColor Green

# 4. 首次运行，下载 ONNX 模型
Write-Host "[4/4] 初始化向量数据库 + 下载 Embedding 模型 (~79MB)..." -ForegroundColor Yellow
python (Join-Path $PSScriptRoot "memory_service.py") stats 2>&1 | Out-Null
Write-Host "  ✓ 初始化完成" -ForegroundColor Green

# 验证
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "使用方法:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  # 先激活虚拟环境（每次打开新终端时）" -ForegroundColor Gray
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "  # 存储记忆" -ForegroundColor Gray
Write-Host '  python memory_service.py add --user your_name --text "要记住的内容"' -ForegroundColor White
Write-Host ""
Write-Host "  # 语义搜索" -ForegroundColor Gray
Write-Host '  python memory_service.py search --user your_name --query "搜索内容"' -ForegroundColor White
Write-Host ""
Write-Host "  # 查看统计" -ForegroundColor Gray
Write-Host "  python memory_service.py stats" -ForegroundColor White
Write-Host ""

# 运行测试
Write-Host "运行验证测试..." -ForegroundColor Yellow
$testResult = python (Join-Path $PSScriptRoot "memory_service.py") add --user test --text "安装验证测试 - Local Memory Store 初始化成功" --metadata '{"type": "setup"}' 2>&1
if ($testResult -match '"status": "ok"') {
    Write-Host "  ✓ 验证通过！" -ForegroundColor Green
} else {
    Write-Host "  ✗ 验证失败，请检查错误信息:" -ForegroundColor Red
    Write-Host $testResult
}
Write-Host ""
