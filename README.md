# Databricks Serverless GPU Compute (SGC) — Onboarding Guide

A hands-on onboarding repository for **Databricks Serverless GPU Compute (SGC)**. Start with "Hello World" examples to learn the basics, then progress to real-world distributed training workloads.

## What is Serverless GPU Compute?

[Serverless GPU Compute (SGC)](https://docs.databricks.com/aws/en/compute/serverless/gpu) is a fully managed component of Databricks serverless compute for custom deep learning workloads. It eliminates GPU cluster management — you select a GPU type and Databricks provisions resources on demand, sets up the distributed environment (NCCL, process groups, ranks), and tears everything down when your job completes.

### Key Features

- **Zero infrastructure management** — no cluster configuration, driver selection, or CUDA setup
- **On-demand GPU provisioning** — GPUs are allocated when needed and auto-terminate after 60 minutes of inactivity
- **Seamless distributed training** — multi-GPU and multi-node orchestration is handled automatically
- **Integrated ML ecosystem** — built-in Unity Catalog, MLflow, and Spark Connect integration
- **Pre-optimized environments** — PyTorch, CUDA 12.6, and common ML libraries come pre-installed

### Supported GPU Types

| GPU | VRAM | Max GPUs | Multi-Node | Best For |
|-----|------|----------|------------|----------|
| **A10** (NVIDIA A10G) | 24 GB | 32 | Yes (up to 70 nodes) | Small-to-medium ML/DL, fine-tuning smaller models |
| **H100** (NVIDIA H100) | 80 GB HBM3 | 128 | Yes (up to 16 nodes, 8 GPUs/node) | Large-scale training, LLM fine-tuning, foundation models |

### Pre-installed Environment (v5)

SGC v5 nodes come with a pre-optimized environment. Two flavors:

- **Standard** (`environment_version: "5"`) — minimal: Ubuntu 24.04, Python 3.12, **no torch preinstalled**. Bring your own stack. Supports CUDA 13.
- **AI** (`base_environment: "databricks_ai_v5"`) — bundled stack: torch 2.9, vLLM 0.13, MLflow, transformers, accelerate, pytorch-lightning, ray, and more. CUDA 12.9.

Older `v4` images are still around but new workspaces should default to v5. Check the [Serverless environment versions](https://docs.databricks.com/aws/en/release-notes/serverless/environment-version/) release notes for the latest.

### Supported Frameworks & Use Cases

**Frameworks:** PyTorch (DDP, FSDP), DeepSpeed, Ray, PyTorch Lightning, HuggingFace Transformers/Accelerate, Axolotl, Unsloth, MosaicML Composer

**Use cases:** LLM fine-tuning (LoRA, QLoRA, full), computer vision, single-cell genomics, digital pathology, recommender systems, reinforcement learning, distributed batch inference

---

## Three Ways to Use SGC

SGC supports three workflows for submitting GPU workloads:

| | SGCLI | Notebook (`@distributed`) | Jobs API (`runs/submit`) |
|---|---|---|---|
| **Interface** | CLI tool (`sgcli`) | Databricks notebook | Databricks Jobs REST API |
| **How it works** | Define a YAML config, submit via `sgcli run` | Decorate a function with `@distributed`, call `.distributed()` | Upload notebook + POST a `submit.json` payload |
| **Best for** | Production multi-node training, large-scale jobs | Interactive multi-GPU development, prototyping | Headless inference / batch / fine-tuning on a single GPU node |
| **Compute target** | Reservation pools or on-demand multi-node | Notebook's attached GPU compute (or `remote=True`) | Single GPU node (`GPU_1xA10` / `GPU_1xH100` / `GPU_8xH100`) |
| **Observability** | Full log streaming, retry management, run history | Notebook cell output | Standard Jobs run page + `runs/get-output` API |
| **Error handling** | Configurable retries (`max_retries`), autoresume from checkpoints | Limited — function either succeeds or raises | Optional retries via Jobs config |
| **Code management** | Git snapshots, reproducible configs | Notebook state | Workspace notebook file |

### Recommendation

- **Start with notebooks** (`@distributed`) for prototyping and interactive multi-GPU development
- **Use SGCLI** for production multi-node training that needs retry logic, code snapshots, and run history
- **Use Jobs API** for single-node inference / batch / fine-tuning, especially when you want to script submissions from CI

---

## Repository Structure

```
databricks-sgc-quickstart/
├── README.md                              # This file
│
├── sgcli_wheel/                           # SGCLI Python wheel (not on PyPI)
│   ├── README.md
│   └── databricks_serverless_gpu_cli-0.1.0-py3-none-any.whl
│
├── sgcli/                                 # SGCLI workflow examples
│   ├── README.md                          # Detailed SGCLI setup and usage guide
│   ├── hello_world/                       # Minimal SGCLI example
│   │   ├── train.yaml                     # Workload definition
│   │   ├── train.py                       # Simple distributed training script
│   │   ├── dependencies.yaml              # Python dependencies
│   │   └── commands.sh                    # Entry script
│   └── geneformer_pretrain/               # Real-world example: Geneformer pretraining
│       └── README.md                      # Guide + link to full example repo
│
├── notebook_interactive/                  # Interactive notebook workflow examples
│   ├── README.md                          # Detailed notebook workflow guide
│   ├── hello_world/                       # Minimal @distributed example
│   │   └── hello_world_distributed.py     # Databricks notebook
│   └── cifar10_classification/            # Real-world example: CIFAR-10 on H100
│       └── README.md                      # Guide + link to full example repo
│
└── jobs_api/                              # Jobs API workflow examples
    ├── README.md                          # Detailed Jobs API setup and usage guide
    └── hello_world/                       # Minimal runs/submit example
        ├── README.md
        ├── gpu_notebook.py                # Notebook to upload + run on GPU
        ├── submit.json                    # Jobs API runs/submit payload
        └── submit_and_poll.sh             # Upload + submit + poll + read-output script
```

---

## Quick Start

### Option 1: Notebook Interactive (Recommended for Getting Started)

1. Import `notebook_interactive/hello_world/hello_world_distributed.py` into your Databricks workspace
2. Attach to a **Serverless GPU** notebook compute (select GPU type from the dropdown)
3. Run the cells — the `@distributed` decorator handles everything

```python
from serverless_gpu import distributed

@distributed(gpus=2, gpu_type='a10')
def hello():
    import torch, os
    rank = int(os.environ.get("RANK", 0))
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    print(f"Hello from rank {rank} on {device} — {torch.cuda.get_device_name(device)}")
    return f"rank-{rank}-ok"

results = hello.distributed()
print(results)
```

### Option 2: SGCLI

1. Install SGCLI from the bundled wheel (recommended: local `env/` in this repo):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   
   ```bash
   uv venv env --python 3.12
   uv pip install --python env/bin/python sgcli_wheel/databricks_serverless_gpu_cli-0.1.0-py3-none-any.whl
   ```
   - This installs SGCLI into `./env` in the current repo (keeps setup project-local and easy to clean up).
   - Run SGCLI directly:
     ```bash
     env/bin/sgcli --help
     ```
   - Or activate first:
     ```bash
     source env/bin/activate
     ```
2. Authenticate: `databricks auth login --host https://your-workspace.cloud.databricks.com`
3. Submit: `cd sgcli/hello_world && ../../env/bin/sgcli run -f train.yaml --watch`

See the [SGCLI README](sgcli/README.md) and [`sgcli_wheel/README.md`](sgcli_wheel/README.md) for full setup instructions.

### Option 3: Jobs API (`runs/submit`)

1. Authenticate the Databricks CLI: `databricks auth login --host https://your-workspace.cloud.databricks.com`
2. Submit + poll a GPU notebook in one command:
   ```bash
   cd jobs_api/hello_world && ./submit_and_poll.sh
   ```

The script uploads `gpu_notebook.py` to your workspace, submits it via `runs/submit` to `GPU_1xA10`, polls until done, and prints the notebook output. See the [Jobs API README](jobs_api/README.md) for compute types and switching to `GPU_1xH100` / `GPU_8xH100`.

---

## Prerequisites

- A Databricks workspace with **Serverless GPU Compute** enabled (see [Supported Regions](#supported-regions))
  - Not supported on compliance security profile workspaces (HIPAA, PCI)
  - PrivateLink workspaces are not supported
- **Unity Catalog** enabled (for Volumes-based data storage)
- Python 3.10+
- [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/) (for SGCLI workflow)

---

## Key Concepts

### The `@distributed` Decorator

The core API for notebook-based SGC workloads:

```python
from serverless_gpu import distributed

@distributed(
    gpus=8,              # Number of GPUs to allocate
    gpu_type='h100',     # 'a10' or 'h100'
    remote=False         # False = attached compute (recommended), True = remote cluster
)
def train():
    ...

results = train.distributed()  # Launch distributed execution
```

**`remote=False` (default, recommended):** Uses GPUs attached to the current notebook. Better observability and error handling.

**`remote=True`:** Provisions a separate remote GPU cluster. Useful when notebook compute is insufficient, but has limited observability.

### SGCLI Workload YAML

The configuration file for workloads submitted via `sgcli`:

```yaml
experiment_name: my-experiment
environment:
  env_variables:
    MY_VAR: "value"
  dependencies: dependencies.yaml
compute:
  gpus: 8
  gpu_type: h100
max_retries: 3
code_source:
  type: snapshot
  snapshot:
    repo_path: /path/to/local/repo
command: |-
  cd $HOME/my_project
  python train.py
```

### MLflow Integration

SGC integrates with Databricks MLflow for experiment tracking. For distributed workloads, the recommended pattern is to **create a single MLflow run in the driver/notebook context** and pass the run ID to workers via `MLFLOW_RUN_ID` so all ranks log to the same run.

See [`notebook_interactive/README.md` → MLflow Integration](notebook_interactive/README.md#mlflow-integration) for the full pattern (with a runnable example in [`notebook_interactive/hello_world/hello_world_distributed.py`](notebook_interactive/hello_world/hello_world_distributed.py)).

### Data Loading

- **Unity Catalog Volumes** — `/Volumes/<catalog>/<schema>/<volume>/...` (recommended)
- **Delta Tables** — via Spark Connect, convert with `.toPandas()`
- **Tip:** Copy data to `/tmp` (NVMe SSD) for multi-epoch training to avoid repeated I/O

### Checkpointing

- Save checkpoints to Unity Catalog Volumes for persistence across job restarts
- SGCLI supports `autoresume` — training automatically resumes from the last checkpoint
- 7-day maximum execution time — implement checkpointing for long training runs

---

## Supported Regions

| Cloud | Regions |
|-------|---------|
| **AWS** | `us-east-1`, `us-east-2`, `us-west-1`, `us-west-2` |
| **Azure** | `eastus`, `eastus2`, `eastusc2`, `eastusc3`, `centralus`, `northcentralus`, `westcentralus`, `westus`, `westus2`, `westus3` |

## Limitations

- **H100 multi-node:** Supported up to 16 nodes (128 GPUs), but currently gated — contact your Databricks account team to enable
- **A10 multi-node:** Supported up to 70 nodes, but provisioning can take up to 20 minutes
- **Runtime cap:** 7-day maximum execution
- **Pip environment size:** Max 15 GB
- **Compliance:** Not supported for HIPAA/PCI workspaces during Beta
- **PrivateLink:** Not supported

---

## Resources

- [Databricks Serverless GPU Docs](https://docs.databricks.com/aws/en/compute/serverless/gpu)
- [Multi-GPU and Multi-Node Workloads](https://docs.databricks.com/aws/en/compute/serverless/distributed-training)
- [SGC Best Practices](https://docs.databricks.com/aws/en/compute/serverless/sgc-best-practices)
- [Data Loading on SGC](https://docs.databricks.com/aws/en/compute/serverless/sgc-dataloading)
- [Serverless GPU API Reference](https://api-docs.databricks.com/python/serverless_gpu/overview.html)
- [Full Life Sciences Examples (Geneformer + GigaPath + CIFAR-10)](https://github.com/databricks-industry-solutions/sgc-examples-lifesciences)
