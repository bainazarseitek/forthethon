#!/bin/bash
export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
MODEL=${VLM_MODEL:-QuantTrio/Qwen3-VL-32B-Instruct-AWQ}
exec /workspace/detector/vllm/.venv/bin/vllm serve "$MODEL" \
  --served-model-name vlm \
  --host 127.0.0.1 --port ${VLM_PORT:-10301} \
  --max-model-len ${VLM_MAX_LEN:-26000} \
  --gpu-memory-utilization ${VLM_GPU_UTIL:-0.92} \
  --limit-mm-per-prompt '{"image":16,"video":1}' \
  --media-io-kwargs '{"video":{"num_frames":-1}}' \
  --mm-processor-cache-gb 0 \
  --max-num-seqs ${VLM_MAX_SEQS:-32} \
  --enable-prefix-caching
