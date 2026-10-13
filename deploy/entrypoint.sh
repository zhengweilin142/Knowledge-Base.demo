#!/usr/bin/env bash
# 容器启动入口：先预热模型与索引，再拉起「演示模型 + 知识库」两个服务。
#
# 为什么用脚本而不是直接 CMD uvicorn：
#   1. 预热能把首次请求从 1~3 分钟降到 1 秒内（详见 tools/preload.py 说明）；
#   2. 公开演示不能挂私人 API Key，需要内置一个规则式演示模型兜底。
set -u

PORT="${PORT:-7860}"
DEMO_PORT="${KB_DEMO_LLM_PORT:-8001}"

echo "############################################################"
echo "#  企业知识库 Demo —— 容器启动"
echo "#  知识库端口 : ${PORT}"
echo "#  演示模型   : 127.0.0.1:${DEMO_PORT}"
echo "############################################################"

# ---- 默认指向内置演示模型（可用真实 Key 覆盖）----
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:${DEMO_PORT}/v1}"
export LLM_MODEL="${LLM_MODEL:-demo-rule-based}"
# 演示模型不校验 Key，但应用要求非空才会调用生成，故给一个占位值。
# 访问者若在页面「设置」里填入自己的 Key，会经 /api/settings 覆盖本值并写入 .env。
if [ -z "${LLM_API_KEY:-}" ]; then
  export LLM_API_KEY="demo-mode-no-real-key"
fi

# ---- 1. 启动内置演示模型 ----
echo "[entrypoint] 启动演示模型 (${DEMO_PORT}) ..."
python -m uvicorn tools.demo_llm:app --host 127.0.0.1 --port "${DEMO_PORT}" --log-level warning &
DEMO_PID=$!

# ---- 2. 预热（模型下载 + 示例文档入库）----
echo "[entrypoint] 开始预热 ..."
python tools/preload.py

# ---- 3. 启动知识库服务 ----
echo "[entrypoint] 启动知识库服务 (0.0.0.0:${PORT}) ..."
python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT}" --log-level info &
APP_PID=$!

# ---- 优雅退出：任一进程退出即整体退出，避免容器看似在跑却无响应 ----
term() {
  echo "[entrypoint] 收到退出信号，正在停止子进程 ..."
  kill -TERM "${APP_PID}" "${DEMO_PID}" 2>/dev/null || true
  wait "${APP_PID}" "${DEMO_PID}" 2>/dev/null || true
  exit 0
}
trap term TERM INT

# 等待任一子进程结束
while kill -0 "${APP_PID}" 2>/dev/null && kill -0 "${DEMO_PID}" 2>/dev/null; do
  sleep 2
done

echo "[entrypoint] 有子进程退出，容器即将结束"
wait "${APP_PID}" 2>/dev/null
echo "[entrypoint] 知识库服务退出码: $?"
exit 1
