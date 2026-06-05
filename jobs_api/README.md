# Jobs API Workflow (`runs/submit`)

This folder demonstrates the **Jobs API workflow** for running GPU workloads on Databricks Serverless GPU Compute. You submit a workspace notebook to a single GPU node via the Databricks **Jobs REST API** (`/api/2.1/jobs/runs/submit`) — no sgcli, no `@distributed` decorator.

## When to use this (vs sgcli / `@distributed`)

| Need | Best fit |
|---|---|
| Headless inference / batch / fine-tuning on a single GPU node | **Jobs API** ← this folder |
| Interactive multi-GPU training, prototyping | `@distributed` ([notebook_interactive/](../notebook_interactive/)) |
| Production multi-node training with retries, code snapshots, log streaming | [sgcli](../sgcli/) |

The Jobs API path is the right choice when:
- You want to run a **single GPU node** (1×A10, 1×H100, or 8×H100) without sgcli overhead
- The workload is **inference**, **batch processing**, or **single-node fine-tuning** (no multi-node coordination needed)
- You want it submittable from CI or a script, with the standard Databricks Jobs lifecycle (PENDING → RUNNING → TERMINATED + result_state)
- You're already invested in the Databricks Jobs CLI / API and prefer reusing that surface

For multi-node training (>1 H100 node), use sgcli — Jobs API submits a single node.

---

## How it works

```
┌────────────┐    workspace import    ┌──────────────────┐
│ gpu_notebook│ ─────────────────────> │ /Workspace/Users │
│  .py        │                        │ /you/...         │
└────────────┘                        └──────────────────┘
                                              │
┌────────────┐    runs/submit                  ▼
│ submit.json│ ─────────────────────> ┌──────────────────┐
└────────────┘                        │  Jobs scheduler  │
                                       │  provisions GPU  │
                                       │  node → runs nb  │
                                       └──────────────────┘
                                              │
                                              ▼
                                       runs/get-output → metrics
```

The notebook runs in the GPU node, returns a JSON result via `dbutils.notebook.exit(...)`, and the caller reads it back through `runs/get-output`.

---

## Compute types (`compute.hardware_accelerator`)

| Value | GPUs / node | VRAM | Notes |
|---|---|---|---|
| `GPU_1xA10` | 1× A10 | 24 GB | Smallest / cheapest; fits models up to ~16 GB weights with room for KV cache. |
| `GPU_1xH100` | 1× H100 | 80 GB | **Beta** — a workspace admin must enable the "AI Runtime Beta" preview, else submit fails with `PERMISSION_DENIED: GPU_1xH100 is not available in your workspace`. |
| `GPU_8xH100` | 8× H100 | 8 × 80 GB = 640 GB | Single 8-GPU node. Submit exactly like 1×H100 — using all 8 is up to your code (e.g. vLLM `tensor_parallel_size=8`, or `@distributed(gpus=8, remote=False)` from inside the notebook). |

---

## Environment options (`environments[].spec`)

| Spec | Base | Preinstalled | Use when |
|---|---|---|---|
| `{"environment_version": "5"}` | **Standard** (minimal) | none — no torch | bring your own stack (e.g. latest vLLM); node supports CUDA 13 |
| `{"base_environment": "databricks_ai_v5"}` | **AI v5** | torch 2.9 + vLLM 0.13 + ML libs (CUDA 12.9) | use the bundled stack, no install |

For brand-new models needing a newer vLLM/transformers than the AI base bundles, use **Standard** (`environment_version: "5"`) and `%pip install` your own.

---

## Setup

```bash
# Install Databricks CLI (macOS)
brew install databricks
# OAuth login — creates / updates ~/.databrickscfg
databricks auth login --host https://your-workspace.cloud.databricks.com

# Or set a profile env var so you don't need -p on every call
export DATABRICKS_CONFIG_PROFILE=DEFAULT
```

Verify auth:

```bash
databricks current-user me
```

---

## Examples in this folder

### Hello World

Submit a tiny notebook to a 1×A10 GPU node, return GPU info + matmul timing as the run output. See [`hello_world/`](hello_world/).

```bash
cd hello_world
./submit_and_poll.sh
```

---

## Gotchas

- **`get-output` requires the TASK `run_id`, NOT the top-level job `run_id`.** Fetch from `runs/get` → `tasks[0].run_id`, then pass to `runs/get-output`. Using the job `run_id` returns an empty / wrong payload.
- **`PENDING` for 1–4 minutes is normal** — GPU provisioning isn't instant. `GPU_1xH100` Beta can be slower.
- **"Execution ran out of memory" is often misleading.** It's the generic message when the Python kernel dies — frequently an *import crash* (e.g. missing CUDA libs, unmet pip pin), not real OOM. Always cross-check with the run-page driver logs before assuming OOM.
- **Full errors live in driver logs (run-page UI), NOT the API.** `runs/get-output` returns the top-level Python exception wrapper. For pip stderr (`No matching distribution …`), vLLM EngineCore subprocess tracebacks, or SIGABRT C-stacks, open the run page in the workspace and read driver/stderr logs.
- **1×H100 requires workspace Beta enablement.** If you hit `PERMISSION_DENIED: GPU_1xH100 is not available in your workspace`, file a request with your account team to enable the AI Runtime Beta preview.

---

## API reference

- [`POST /api/2.1/jobs/runs/submit`](https://docs.databricks.com/api/workspace/jobs/submit) — submit a one-shot run
- [`GET /api/2.1/jobs/runs/get`](https://docs.databricks.com/api/workspace/jobs/runsget) — poll lifecycle state
- [`GET /api/2.1/jobs/runs/get-output`](https://docs.databricks.com/api/workspace/jobs/runsgetoutput) — fetch `notebook_output.result` and any error info (use the **task** run_id)
- [Workspace import](https://docs.databricks.com/aws/en/dev-tools/cli/reference/workspace-commands) — upload the notebook before submitting
