#!/usr/bin/env bash
# 变压器故障诊断多智能体系统 —— 前端启动脚本
# 用法：
#   bash run_frontend.sh                 # 默认：加载 5143 条 DGA 学习到的贝叶斯参数
#   USE_LEARNED_CPT=0 bash run_frontend.sh   # 对比实验：使用专家默认 CPT
set -e
cd "$(dirname "$0")"

# 诊断引擎参数来源（P0：显式设置，避免「参数文件已生成但默认未加载」）
USE_LEARNED_CPT="${USE_LEARNED_CPT:-1}"
LEARNED_PARAMS="data/real/dga/learned_params.json"
if [ "$USE_LEARNED_CPT" = "1" ]; then
  if [ -f "$LEARNED_PARAMS" ]; then
    export FAULT_ATTR_PARAMS="$LEARNED_PARAMS"
    echo "[run_frontend] FAULT_ATTR_PARAMS=$FAULT_ATTR_PARAMS（数据学习参数）"
  else
    echo "[run_frontend] 警告：未找到 $LEARNED_PARAMS，将使用专家默认 CPT。可先运行 python3 scripts/learn_cpt.py 生成。"
    unset FAULT_ATTR_PARAMS
  fi
else
  unset FAULT_ATTR_PARAMS
  echo "[run_frontend] USE_LEARNED_CPT=0，使用专家默认 CPT"
fi

# 若 streamlit 未安装则安装
if ! python3 -c "import streamlit" >/dev/null 2>&1; then
  echo "[run_frontend] 正在安装 streamlit ..."
  python3 -m pip install streamlit
fi

echo "[run_frontend] 启动 Streamlit 前端 http://localhost:8501 ..."
exec python3 -m streamlit run app.py --server.port 8501
