#!/usr/bin/env bash
set -euo pipefail

# 只有 SLURM_PROCID=0 的 task 执行 sweep，其它 task 直接退出
if [[ -n "${SLURM_PROCID-}" && "${SLURM_PROCID}" != "0" ]]; then
  exit 0
fi

export WORLD_SIZE=8
export SEED=42
export CHECK=0

export HIDDEN="${HIDDEN:-4096}"
export CAPACITY="${CAPACITY:-8192}"

EXPERTS_TOTAL_LIST=(32 64)
TOPK_LIST=(2 4 8)
BATCH_LIST=(2 4 8 16)
SEQ_LIST=(2048 4096)

for EXPERTS_TOTAL in "${EXPERTS_TOTAL_LIST[@]}"; do
  E_LOCAL=$((EXPERTS_TOTAL / WORLD_SIZE))
  echo "=== EXPERTS_TOTAL=${EXPERTS_TOTAL} -> E_LOCAL=${E_LOCAL} (WORLD_SIZE=${WORLD_SIZE}) ==="

  for TOPK in "${TOPK_LIST[@]}"; do
    for BATCH in "${BATCH_LIST[@]}"; do
      for SEQ in "${SEQ_LIST[@]}"; do
        export TOPK BATCH SEQ E_LOCAL

        echo "[RUN] W=${WORLD_SIZE} E_TOTAL=${EXPERTS_TOTAL} E_LOCAL=${E_LOCAL} TOPK=${TOPK} BATCH=${BATCH} SEQ=${SEQ} HIDDEN=${HIDDEN} CAP=${CAPACITY} SEED=${SEED}"

        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT=$((29500 + (RANDOM % 1000)))

        python benchmark.py
      done
    done
  done
done
