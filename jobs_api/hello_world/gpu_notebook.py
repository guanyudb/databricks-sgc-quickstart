# Databricks notebook source
# MAGIC %md
# MAGIC # SGC Hello World — Jobs API (`runs/submit`)
# MAGIC
# MAGIC This notebook is submitted to a single SGC GPU node via the Databricks Jobs REST API
# MAGIC (`/api/2.1/jobs/runs/submit`). It runs on the GPU, returns metrics via
# MAGIC `dbutils.notebook.exit(...)`, and the caller reads them with `runs/get-output`.
# MAGIC
# MAGIC ### What this notebook does
# MAGIC
# MAGIC 1. Prints GPU + CUDA + torch info from the node it landed on
# MAGIC 2. Runs a 4096×4096 matmul benchmark to confirm the GPU works end-to-end
# MAGIC 3. Returns a JSON dict via `dbutils.notebook.exit(...)` — surfaces as `notebook_output.result`
# MAGIC
# MAGIC ### Notes
# MAGIC
# MAGIC - Works identically on `GPU_1xA10`, `GPU_1xH100`, and `GPU_8xH100` — only `compute.hardware_accelerator` in `submit.json` differs.
# MAGIC - With `environment_version: "5"` (Standard, the default in `submit.json`), the node has no torch preinstalled — we install it in the first cell.
# MAGIC - With `base_environment: databricks_ai_v5`, torch + ML libs are preinstalled; the install cell becomes a no-op.

# COMMAND ----------

# Install torch on the Standard v5 environment. (No-op if already present.)
%pip install --quiet "torch>=2.7"
dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import time

import torch

device_count = torch.cuda.device_count()
print(f"torch={torch.__version__}, cuda={torch.version.cuda}, devices={device_count}")
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

assert device_count >= 1, "No GPU visible — check the compute.hardware_accelerator in submit.json"

device = torch.device("cuda:0")
gpu_name = torch.cuda.get_device_name(device)
print(f"GPU 0: {gpu_name}")

# COMMAND ----------

# 4096x4096 matmul benchmark with warmup
N = 4096
a = torch.randn(N, N, device=device, dtype=torch.float32)
b = torch.randn(N, N, device=device, dtype=torch.float32)

for _ in range(3):
    torch.matmul(a, b)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
c = torch.matmul(a, b)
end.record()
torch.cuda.synchronize()

elapsed_ms = start.elapsed_time(end)
tflops = (2.0 * N**3) / (elapsed_ms / 1000.0) / 1e12
print(f"{N}x{N} fp32 matmul: {elapsed_ms:.2f} ms ({tflops:.2f} TFLOPS)")

# COMMAND ----------

# Return metrics as the notebook result — read on the caller side via runs/get-output
result = {
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "device_count": device_count,
    "gpu_name": gpu_name,
    "matmul_ms": elapsed_ms,
    "tflops": tflops,
}
print(json.dumps(result, indent=2))
dbutils.notebook.exit(json.dumps(result))
