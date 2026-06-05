# Hello World (Jobs API)

Submit a minimal GPU notebook to SGC via the **Jobs API** (`/api/2.1/jobs/runs/submit`).

## What this does

1. **Uploads** `gpu_notebook.py` to your workspace
2. **Submits** it to a single GPU node (default: `GPU_1xA10`)
3. **Polls** until the run finishes
4. **Reads** the result (GPU info + matmul timing) back via `runs/get-output`

The notebook itself:
- Installs `torch` (the Standard v5 environment has no torch preinstalled)
- Prints CUDA / GPU info
- Runs a 4096×4096 fp32 matmul benchmark
- Returns metrics as JSON via `dbutils.notebook.exit(...)`

## Files

| File | Purpose |
|---|---|
| `gpu_notebook.py` | The Databricks notebook (Python source format) |
| `submit.json` | Jobs API `runs/submit` payload — defines compute + environment |
| `submit_and_poll.sh` | Upload + submit + poll + read-output, all in one |

## Prerequisites

```bash
brew install databricks                                                # macOS
databricks auth login --host https://your-workspace.cloud.databricks.com
export DATABRICKS_CONFIG_PROFILE=DEFAULT                               # or your profile
```

You also need `jq` installed (`brew install jq` on macOS).

## Run it

```bash
cd jobs_api/hello_world
./submit_and_poll.sh
```

Expected timeline:
- Notebook upload: < 5s
- `runs/submit` → `PENDING`: instant
- `PENDING` → `RUNNING`: ~1–4 min (GPU provisioning)
- Notebook execution (torch install + matmul): ~30–90s
- Output fetched: instant

## Expected output

The tail of the script prints something like:

```json
{
  "notebook_output": {
    "result": "{\"torch_version\": \"2.7.1+cu126\", \"cuda_version\": \"12.6\", \"device_count\": 1, \"gpu_name\": \"NVIDIA A10G\", \"matmul_ms\": 4.21, \"tflops\": 32.59}",
    "truncated": false
  },
  "error": null,
  "error_trace": null
}
```

## Switching compute

Edit `submit.json`:

```json
"compute": { "hardware_accelerator": "GPU_1xA10" }   // 1×A10  — 24 GB, cheapest
"compute": { "hardware_accelerator": "GPU_1xH100" }  // 1×H100 — Beta, requires workspace enablement
"compute": { "hardware_accelerator": "GPU_8xH100" }  // 8×H100 — single 8-GPU node (use vLLM/@distributed inside)
```

## Switching to the AI bundled environment

By default the script uses Standard v5 (`environment_version: "5"`) — minimal, no torch. To use the bundled AI stack (torch + vLLM + ML libs preinstalled), edit `submit.json`:

```json
"environments": [
  { "environment_key": "gpu_env", "spec": { "base_environment": "databricks_ai_v5" } }
]
```

Then you can drop the `%pip install torch` cell from the notebook.

## Gotchas

- **`get-output` needs the TASK run_id** — `tasks[0].run_id` from `runs/get`, not the top-level job `run_id`. The script handles this.
- **`PENDING` for 1–4 min is normal.** GPU provisioning isn't instant.
- **"Execution ran out of memory" can mislead.** Generic message for any kernel death — often an import crash, not real OOM. Open the run page (URL printed by the script) and check **driver logs** for the real error.
- **`PERMISSION_DENIED: GPU_1xH100 is not available in your workspace`** — H100 Beta needs admin enablement. Ask your account team.
