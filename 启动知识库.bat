@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo   企业知识库 Demo - 启动服务
echo ============================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 找不到虚拟环境 .venv\Scripts\python.exe
    echo        请先在项目目录执行：
    echo        python -m venv .venv
    echo        .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo [1/2] 检查依赖...
.venv\Scripts\python.exe -c "import fastapi, chromadb, pypdf, docx, openpyxl" 2>nul
if errorlevel 1 (
    echo       依赖不完整，正在安装 requirements.txt ...
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 ( echo [错误] 依赖安装失败 & pause & exit /b 1 )
)

echo [2/2] 启动服务（首次启动需下载约95MB向量模型，请耐心等待）
echo.
echo       界面地址: http://127.0.0.1:8000
echo       停止服务: 按 Ctrl+C
echo.
start "" http://127.0.0.1:8000
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

echo.
echo 服务已退出。
pause
