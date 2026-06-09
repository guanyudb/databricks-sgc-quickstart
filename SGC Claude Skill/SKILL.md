---
name: sgc
description: Build, run, and debug training/inference on Databricks Serverless GPU Compute (SGC). Use when the user mentions sgcli, @distributed, GPU_1xA10 / GPU_1xH100 / GPU_8xH100, runs/submit on a GPU, multi-node H100 training, or DCS / custom Docker images for SGC.
---

# Serverless GPU Compute (SGC) — practitioner's skill

Three execution paths share most concepts:

1. **`sgcli` job submission** — point-and-shoot GPU jobs from a local repo with a YAML spec. Best for headless multi-node training, reproducibility, scripted launches.
2. **Notebook with `@distributed`** — interactive multi-GPU training launched from a Databricks notebook. Best for prototyping + iterating.
3. **Jobs API `runs/submit` (notebook task on GPU)** — submit a workspace notebook to a GPU node (`GPU_1xA10`, `GPU_1xH100`, or `GPU_8xH100`) via the Databricks Jobs REST API. Best for inference / batch / fine-tuning / single-node training.

The platform underneath is the same; the difference is in how you describe the job and where the driver code runs.

---

## Submitting jobs with `sgcli`

`sgcli` is the Databricks Serverless GPU CLI. It packages a local code directory, ships it to GPU workers, and runs your GPU workload, training, inference, etc.

### Install

`sgcli` is distributed as a Python wheel — it's **not on PyPI**. Get the wheel either way:

- **Search your org's Google Drive** for `databricks_serverless_gpu_cli` (Databricks-internal folder maintained by the SGC team).
- **Or ask your Databricks account team** (your SA/DSA) for the latest wheel. Ask for the largest version without `dev` or `staging` qualifiers.

Install with `uv` (recommended — isolated env, adds `sgcli` to your PATH):

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the wheel — Python ≥ 3.10 required, 3.12 recommended
uv tool install --python 3.12 /path/to/databricks_serverless_gpu_cli-X.Y.Z-py3-none-any.whl
```

If you previously `pip install`ed `sgcli` into a venv, that copy takes precedence when the venv is active — uninstall it first:

```bash
pip uninstall sgcli
```

### Auth

`sgcli` reads from `~/.databrickscfg` profiles, same as the regular Databricks CLI. Set one up if you don't have one:

```bash
# Install Databricks CLI (macOS)
brew install databricks

# OAuth login — opens a browser; creates / updates ~/.databrickscfg
databricks auth login --host https://<your-workspace>.cloud.databricks.com
```

This writes a profile like:

```ini
[DEFAULT]
host      = https://<your-workspace>.cloud.databricks.com
auth_type = databricks-cli

[dev]
host      = https://dev-workspace.cloud.databricks.com
auth_type = databricks-cli
```

Pick a profile per `sgcli` call with `-p PROFILE` / `--profile PROFILE`. Or set an env var once so `-p` becomes unnecessary:

```bash
export DATABRICKS_CONFIG_PROFILE=dev
```

Without either, `sgcli` uses `[DEFAULT]`.

### Command surface

```
sgcli run       — submit a job run from a YAML spec
sgcli monitor   — block until a run completes, streaming logs + events
sgcli get runs  — list submitted runs
sgcli get status <run_id>
sgcli get logs <run_id> [--node N] [--local-rank R]
sgcli get pools — list available GPU pools (on-demand + reservations)
sgcli cancel <run_id>
sgcli register  — register custom Docker images
```

Common one-liners:

```bash
sgcli run --file training.yaml                              # submit
sgcli run --file training.yaml --watch                      # submit + stream logs
sgcli run --file training.yaml --dry-run                    # validate, don't submit
sgcli run --file training.yaml --override compute.gpus=32 timeout_minutes=120
sgcli get runs --limit 10                                   # list recent runs
sgcli get runs --active                                     # only active runs
sgcli get status 388840544681214                            # specific run
sgcli get logs 388840544681214                              # default: node 0
sgcli get logs 388840544681214 --node 2 --local-rank 3      # specific GPU
sgcli cancel 388840544681214
sgcli monitor 388840544681214                               # agent-friendly: blocks until done
```

Inline YAML config help is always available via `sgcli -h config.<field>` (e.g., `sgcli -h config.compute`).

### YAML spec — `training.yaml`

Required fields: `experiment_name`, `environment`, `compute`, and exactly one of `command` / `bash_script`. Everything else is optional.

```yaml
experiment_name: /Users/me/my-experiment    # MLflow experiment (repeat to share)
run_name: 2026-06-02-256gpu                 # optional, becomes the MLflow run name

# How your local code gets shipped to workers
code_source:
  type: snapshot
  snapshot:
    repo_path: $HOME/my_repo                 # local dir to package
    # Optional pinning:
    # git_branch: main
    # git_commit: <sha>
    # allow_uncommitted: false               # default; flip to true for WIP
    # remote_volume: /Volumes/main/me/sgc    # ship via UC volume instead of workspace

# Python deps + Docker
environment:
  dependencies: requirements.yaml            # path to requirements YAML (see below)
  # OR mutually exclusive:
  # docker_image:
  #   url: <custom-image-uri>   # see DCS section

# Top-level env vars (preferred over the deprecated environment.env_variables)
env_variables:
  NCCL_DEBUG: INFO
  MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL: "180"
env_variables_secrets:
  # Format: "scope_name/key_name"  (set up via Databricks Secrets first)
  HF_TOKEN: "my-scope/hf-token"
  AWS_ACCESS_KEY_ID: "my-scope/aws-access-key"

compute:
  gpus: 256                                  # must be a multiple of GPUs-per-node
  gpu_type: h100                             # "h100" (8 GPUs/node; alias of h100_80gb) or "a10" (1 GPU/node)
  gpu_pool_name: "my-reservation-pool"       # for reservation pools (omit for on-demand)
  # gpu_node_pool_id: <uuid>                 # alternate: pool by ID (mutually exclusive)

# Parameters get serialized to a YAML on the cluster; path exposed as $HYPERPARAMETERS_PATH
parameters:
  learning_rate: 1.5e-4
  batch_size: 512
  seq_len: 8192
  data_path: s3://my-bucket/teddy/mds
  checkpoint_dir: s3://my-bucket/teddy/checkpoints

max_retries: 2                               # default 0; bump for production
timeout_minutes: 720                         # optional hard cap
# budget_policy_id: <uuid>                   # for billing routing
# priority: 500                              # 0–999, for pool-scheduled workloads

command: |
  cd $HOME/my_repo
  python train.py
```

### `requirements.yaml` (the dependencies file)

Modeled on the Databricks workspace base environment format. Path is referenced from `environment.dependencies`.

```yaml
version: '4'
dependencies:
  - --index-url https://pypi.org/simple
  - torch==2.4.0
  - transformers>=4.40.0
  - mosaicml==0.23.0
  - /Workspace/Shared/wheels/my-internal-pkg-0.2.0-py3-none-any.whl
```

### `parameters` → `$HYPERPARAMETERS_PATH`

Whatever you put under `parameters:` is serialized to a YAML file on the worker. The path is exposed as the `HYPERPARAMETERS_PATH` env var. Load it in your script:

```python
import os, yaml
with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    params = yaml.safe_load(f)

lr = params["learning_rate"]
data_path = params["data_path"]
```

This is a cleaner pattern than committing a `parameters.yaml` into your repo when the same code runs across many sweeps — just override `parameters.learning_rate=2e-4` per submission.

### Code shipping (`code_source`)

The snapshot type packages a local directory and unpacks it under `$HOME/<last-folder-of-repo_path>` on every worker. So if `repo_path: /Users/me/work/model`, your code lands at `$HOME/model/` on each node. Your `command` should `cd` there first.

Snapshot options worth knowing:

- `git_branch` / `git_commit` — pin the snapshot to a specific git ref (uncommitted changes ignored).
- `allow_uncommitted: true` — include WIP local changes (useful for iteration; incompatible with git pins).
- `remote_volume: /Volumes/.../...` — upload the snapshot to a UC volume rather than the workspace (faster for large repos).
- `include_paths` — selectively include subpaths to keep the snapshot small.

### Submission patterns

**Smoke test first, scale after.** 

```bash
sgcli run --file training.yaml --override compute.gpus=8                 # 1 node
sgcli run --file training.yaml --override compute.gpus=16                # 2 nodes (multi-node path)
sgcli run --file training.yaml                                            # full scale
```

**Same code, multiple sweeps.**

```bash
for lr in 1e-4 2e-4 5e-4; do
  sgcli run --file training.yaml --override "parameters.learning_rate=$lr" "run_name=lr-$lr"
done
```

**Watch + auto-cancel on bad signal.**

```bash
sgcli run --file training.yaml --watch
# Ctrl-C cancels the run cleanly via sgcli cancel <run_id>
```

**Agent-friendly headless monitor (e.g., from CI):**

```bash
RUN_ID=$(sgcli run --file training.yaml --json | jq -r '.run_id')
sgcli monitor "$RUN_ID"     # blocks until terminal state, streams events
```

---

## Docker Container Service (DCS) — custom images for sgcli

DCS (private preview) lets `sgcli` workloads run inside your own Docker image instead of the default base + `requirements.yaml`. Reach for it when you need: specific system library versions, CUDA extensions (`flash-attn`, `apex`, `xformers`, custom kernels), exact-reproduction environments for SOTA papers, or a corporate platform/security-team image.

### Register the image (one-time per tag)

```bash
# Public image
sgcli register image docker.io/nvidia/cuda:13.1.0-devel-ubuntu24.04 -p PROFILE

# Private Docker Hub image — uses creds from ~/.docker/config.json (run `docker login` first)
sgcli register image myorg/myrepo:mytag -p PROFILE

# Private image — prompted for username + PAT, stored in a Databricks secret scope (CI-friendly form: --scope SCOPE --key KEY)
sgcli register image myorg/myrepo:mytag --interactive-authenticate -p PROFILE
```

Registration takes 2–6 min and blocks until the image is cached. Re-run only when you push a new tag or rotate the PAT.

### Reference the image in your workload YAML

`environment.docker_image.url` is **mutually exclusive** with `environment.dependencies`:

```yaml
experiment_name: my-dcs-job
environment:
  docker_image:
    url: myorg/myrepo:mytag
compute:
  gpus: 1
  gpu_type: a10
command: python /app/train.py    # absolute paths — see WORKDIR gotcha below
```

### Distributed env vars injected into your container

For multi-node `torchrun` / `accelerate` `command:` strings, the platform sets:

| Variable | Meaning |
|---|---|
| `NUM_NODES` | total nodes |
| `LOCAL_WORLD_SIZE` | GPUs per node |
| `WORLD_SIZE` | total processes |
| `POD_RANK` / `NODE_RANK` | current node rank (0-indexed) |
| `MASTER_ADDR`, `MASTER_PORT` | coordination endpoint (multi-node; use `localhost` single-node) |
| `IS_HOST` | `1` on the rank-0 node |
| `LOCAL_ADDR` | local node IP |

### Gotchas

- **WORKDIR is ignored at runtime.** `sgcli` runs from a platform-controlled dir, not your image's `WORKDIR`. Use **absolute paths** in `command:` (`python /app/train.py`, not `python train.py`). Same for `COPY . /app` in the Dockerfile.
- **flash-attn + CXX11 ABI mismatch.** v2.8.0+ release wheels tagged `cxx11abiFALSE` are actually built with ABI=True, and source builds on Ubuntu 24.04 always produce new-ABI. Pin `flash-attn==2.7.4.post1`, install with `--no-deps`. Check after build: `python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"`.
- **OS-level NCCL is usually unused.** `pip install torch` ships its own bundled NCCL (e.g. torch 2.6 → 2.21.5) and loads it preferentially over `/usr/lib/.../libnccl.so`. Verify: `python -c "import torch; print(torch.cuda.nccl.version())"`.
- **EFA on A10 = bloat + noise.** Base images install EFA unconditionally; on `gpu_type: a10` (no EFA hardware), NCCL emits three `NET/OFI ... initialization failed` WARN lines per job and falls back to socket. Silence with `env_variables: { NCCL_NET_PLUGIN: "none" }`.
- **Limitations (private preview):** Docker Hub only (no ECR/GCR/GHCR), AWS + Azure only (no GCP), image size < 20 GB, can't combine `docker_image.url` with `dependencies`.

### Recommended base images

`databricksruntime/air` on Docker Hub ships CUDA + NCCL + cloud RDMA preconfigured — start here unless you genuinely need a from-scratch image:

| Tag | Cloud | Variant | Use |
|---|---|---|---|
| `databricksruntime/air:dcs-base-aws-runtime` | AWS | runtime (~4.7 GB) | `pip install` pre-built wheels only |
| `databricksruntime/air:dcs-base-aws-devel` | AWS | devel (~11 GB) | needs `nvcc` (flash-attn, apex, custom kernels) |
| `databricksruntime/air:dcs-base-azure-runtime` | Azure | runtime (~4.1 GB) | pre-built wheels |
| `databricksruntime/air:dcs-base-azure-devel` | Azure | devel (~10.3 GB) | needs `nvcc` |

All four: Ubuntu 24.04, Python 3.12.3, CUDA 12.9.0 runtime, OS-level NCCL 2.26.5. AWS variants add EFA 1.42.0 + aws-ofi-nccl 1.15.0 + libfabric; Azure adds rdma-core + ibverbs.

For full Dockerfile patterns (from-scratch builds, driver/CUDA/NCCL/PyTorch/CXX11 ABI compatibility matrix, pre-build verification checklist) see the **[External] SGCLI Docker (DCS): Private Preview User Guide** — "Docker image build skill" tab.

---

## Notebook-based distributed training (`@distributed`)

A managed launcher that fans a function across GPU workers and collects per-rank returns to the caller. Two modes:

- **`remote=False` (default)** — runs on the driver node's GPUs (the GPU node the notebook is attached to, e.g. `GPU_8xH100`). Most manageable: one node, inline logs/results.
- **`remote=True`** — provisions a fresh GPU node of the requested `gpus` × `gpu_type` and runs there, independent of the notebook's compute. Pattern: develop on a 1×A10 notebook, dispatch the heavy run to 8×H100.

**Current API** (`serverless_gpu` v0.5.x in AI v5; older `databricks.distributed` / `num_gpus_per_node` / `framework=` form is gone):

```python
from serverless_gpu import distributed

@distributed(gpus=8, gpu_type="h100_80gb", remote=False)
def train():
    import os, torch, torch.distributed as dist
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    # ... training loop ...
    dist.destroy_process_group()
    return {"rank": dist.get_rank()}

results = train.distributed()   # launch — calling train() directly raises RuntimeConfigurationError
```

`gpu_type` is `"h100_80gb"` or `"a10"`; `gpus` is the total count. Run on a notebook attached to (or `runs/submit` targeting) a GPU node with `base_environment: databricks_ai_v5` — torch + `serverless_gpu` are preinstalled.

---

## Submitting GPU notebook jobs via the Jobs API (`runs/submit`)

The simplest path for **inference, batch, training, or fine-tuning** is a **notebook task** submitted through the Databricks Jobs REST API — no sgcli. The notebook runs directly on one GPU node. The **same payload works for 1×A10, 1×H100, and 8×H100** — only `compute.hardware_accelerator` changes; an 8×H100 box is a single 8-GPU node submitted identically. (Verified working on A10 and H100, June 2026.)

### Compute types (`compute.hardware_accelerator`)

| Value | GPUs / node | VRAM | Notes |
|---|---|---|---|
| `GPU_1xA10` | 1× A10 | 24 GB | smallest/cheapest; fits models up to ~16 GB weights with room for KV cache |
| `GPU_1xH100` | 1× H100 | 80 GB | **Beta** — a workspace admin must enable the "AI Runtime Beta" preview, else submit fails with `PERMISSION_DENIED: GPU_1xH100 is not available in your workspace` |
| `GPU_8xH100` | 8× H100 | 8×80 GB = 640 GB | single node, 8 GPUs. Submit exactly like 1×H100; **using all 8 is up to your code** (see below) |

### Using all 8 GPUs on `GPU_8xH100`

The notebook process already sees all 8 (`torch.cuda.device_count() == 8`, no `CUDA_VISIBLE_DEVICES` masking). How to drive them:

- **Training / fine-tuning:** `@distributed(gpus=8, gpu_type="h100_80gb", remote=False)` from the notebook — see the `@distributed` section. (Raw `torchrun --nproc_per_node=8` / HF Accelerate / DeepSpeed also work since all 8 GPUs are visible.)
- **Inference (vLLM):** `LLM(model=..., tensor_parallel_size=8)` — vLLM spawns its own workers across all 8 GPUs; no `@distributed` needed.
- **1×A10 / 1×H100** — one GPU; the notebook just runs on it, nothing special needed.

### Environment options (`environments[].spec`)

| Spec | Base | Preinstalled | Use when |
|---|---|---|---|
| `{"environment_version": "5"}` | **Standard** (minimal) | none — **no torch** | bring your own stack (e.g. latest vLLM); node supports CUDA 13 |
| `{"base_environment": "databricks_ai_v5"}` | **AI v5** | torch 2.9 + vLLM 0.13 + ML libs (CUDA 12.9) | use the bundled stack, no install |

For **brand-new models** needing a newer vLLM/transformers than the AI base bundles, use **Standard** (`environment_version: "5"`) and `%pip install` your own — the AI env's bundled vLLM lags (see "Bringing a recent vLLM" below).

### Working submit payload

```json
{
  "run_name": "my-gpu-job",
  "tasks": [
    {
      "task_key": "my_task",
      "notebook_task": {
        "notebook_path": "/Workspace/Users/me@databricks.com/path/to/notebook",
        "source": "WORKSPACE"
      },
      "environment_key": "gpu_env",
      "compute": { "hardware_accelerator": "GPU_1xA10" }
    }
  ],
  "queue": { "enabled": true },
  "environments": [
    { "environment_key": "gpu_env", "spec": { "environment_version": "5" } }
  ],
  "performance_target": "PERFORMANCE_OPTIMIZED"
}
```

### Submit / poll / read output (Databricks CLI)

```bash
# 1. upload the notebook to the workspace first
databricks -p PROFILE workspace import /Workspace/Users/me/llm/nb \
  --file nb.py --language PYTHON --format SOURCE --overwrite

# 2. submit
databricks -p PROFILE api post /api/2.1/jobs/runs/submit --json @submit.json
# -> {"run_id": 123...}

# 3. poll job state (life_cycle_state: PENDING -> RUNNING -> TERMINATED; result_state SUCCESS/FAILED)
databricks -p PROFILE api get /api/2.1/jobs/runs/get --json '{"run_id": 123}'

# 4. read notebook output / error — USE THE TASK run_id (tasks[0].run_id), NOT the job run_id
databricks -p PROFILE api get /api/2.1/jobs/runs/get-output --json '{"run_id": <TASK_RUN_ID>}'
```

Return a result from the notebook with `dbutils.notebook.exit(json.dumps(metrics))` → it surfaces as `notebook_output.result`.

### Gotchas

- **`get-output` needs the TASK run_id** — `tasks[0].run_id` from `runs/get`, not the top-level job `run_id`.
- **"Execution ran out of memory" is often a MISLEADING label.** It's the generic message when the Python kernel dies — frequently an *import crash / SIGABRT*, not real OOM. Don't trust it; read the driver logs.
- **Full errors live in the driver logs (run-page UI), NOT the API.** `get-output` returns only the top-level Python exception wrapper — not pip stderr (`No matching distribution…`), not vLLM EngineCore subprocess tracebacks, not the SIGABRT C-stack. For those, open the run page → driver/stderr logs.
- **Provisioning takes ~1–4 min** (PENDING is normal). 1×H100 Beta must be enabled per workspace.

---

## Environment selection (SGC v4 / v5 / AI base)

- **SGC v4** — older AI V4 base image. Ships NCCL 2.27.5 + CUDA 12.9 + an AWS-OFI-NCCL plugin. **Has the latent cuMem multi-node hang** (see troubleshooting).
- **SGC v5** — current newest (released Feb 2026; **no v6 as of June 2026**). Two flavors:
  - **Standard** (`environment_version: "5"`) — minimal, **no torch preinstalled**. Bring your own stack.
  - **AI** (`base_environment: "databricks_ai_v5"`) — bundles torch 2.9 + vLLM 0.13 + ML libs, CUDA 12.9.
- **CUDA version is NOT fixed at 12.9 anymore.** Verified: the **Standard v5 node runs CUDA 13** (a cu13 torch wheel imports and runs). The "CUDA 12.9" figure is specific to the v4 / AI-v5 *bundled* stack — what you actually get depends on the env version **and what you install**.
- **Python in current SGC v4/v5:** 3.12.x. Bumps within `>=3.12,<3.13` need no extra testing.
- **Where to check the latest** (env version, bundled torch/vLLM/CUDA): the Databricks docs **"Serverless environment versions"** release-notes page (`docs.databricks.com/aws/en/release-notes/serverless/environment-version/`) and its per-version GPU pages. You can also introspect a live env: `pip show vllm torch` / `python -c "import torch; print(torch.version.cuda)"` in a notebook cell.

### Dependency pinning gotchas

- **Don't replace `nvidia-nccl-cu12` from the base image.** The torch wheel ships its own bundled NCCL and the host stack's CUDA libs are coupled to the OFI plugin — pinning a different `nvidia-nccl-cu12` than the torch wheel expects can break multi-node EFA transport. Pin everything *except* nccl-cu12 / cuBLAS / cuDNN; install torch `--no-deps` if needed.
- **Composer** still works on SGC, but custom PyTorch trainers are now the recommended path for new code.

---

## Reservation pools

### On-demand vs reservation

- **On-demand** — omit `gpu_pool_name`. Best-effort scheduling, pay-per-pod usage, faster spin-up.
- **Reservation pool** — specify `gpu_pool_name` (or `gpu_node_pool_id`). Guaranteed capacity (e.g., 256 H100 reserved for N days), billed per-node uptime, coordinated through the Databricks account team.

```yaml
compute:
  gpus: 256
  gpu_type: h100
  gpu_pool_name: "my-2026-06-02"
```

List available pools with `sgcli get pools`.

---

## Checkpointing — best practices

- **Cadence:** every 1–2 hours for jobs running > 24h. Loss of progress to a node fault is expensive on 256 H100.
- **Checkpoint dataloader + RNG state**, not just model weights. Restart should resume at the exact batch without re-seeing data. (Composer did this automatically; custom PyTorch trainers must do it explicitly.)
- **Async, distributed checkpointing** beats synchronous gather-to-rank-0. Use `torch.distributed.checkpoint.async_save` paired with the [AWS S3 Connector for PyTorch](https://github.com/awslabs/s3-connector-for-pytorch) for parallel uploads.
- **Resume logic:** training code must look up the latest checkpoint on restart (e.g., via a symlink the trainer maintains). The `max_retries` retry gives you a fresh process with the same workload spec, but it's on you to actually pick up the checkpoint.

---

## Pre-launch checklist for big runs

Before kicking off a multi-hundred-GPU production run:

1. **Smoke test** at 8 GPU (1 node) and 16 GPU (2 nodes) with identical code + deps.
2. **Verify checkpointing** end-to-end: save → kill → restart → confirm resume from correct step.
3. **Set `max_retries: 2`** in the YAML spec.
4. **Set `env_variables.MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL: "180"`** (or request an MLflow quota bump if the run will exceed the 50M datapoint cap).

---

## Troubleshooting

- **MLflow 50M datapoint cap.** Long-running training hits the cap with the default 10s system-metric sampling. Set `MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL: "180"` to sample every 3 minutes, or request a workspace quota bump.
- **Job UI shows "failed" while the pod is healthy.** A UI/status sync issue when a job runs past the 10-day boundary; the pod and logs are still good. Check driver logs / pod state rather than trusting the UI.
- **`Guessing device ID based on global rank` warning.** Pass `device_id` explicitly to `init_process_group` (see the `@distributed` skeleton) and set `LOCAL_RANK` manually if you're not using a launcher that injects it.

## What not to do

- Don't run multi-node without explicit `LOCAL_RANK` handling + `device_id` passed to `init_process_group`.
- Don't checkpoint less often than every 1–2 hours for jobs > 24h. You're one node fault away from a bad day.
- Don't open `mlflow.start_run()` inside the distributed function without `run_id=...`. Create the run in notebook context and pass `MLFLOW_RUN_ID` via env, or you'll get N fragmented runs.
- Don't put `parameters` inline in your code and commit it. Use `parameters:` in the YAML so sweeps + overrides are trivial.
- Don't ship a 5 GB repo via the default snapshot. Use `include_paths` or `remote_volume` for large code bases.
