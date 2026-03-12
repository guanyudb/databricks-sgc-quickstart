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
| **H100** (NVIDIA H100) | 80 GB HBM3 | 32 | Single-node only | Large-scale training, LLM fine-tuning, foundation models |

### Pre-installed Environment (v4+)

SGC nodes come with a pre-optimized environment including:
- **Ubuntu 24.04**, **Python 3.12**, **CUDA 12.6**
- **PyTorch 2.7+**, **torchvision**, **torchaudio**, **flash-attention 2.8+**
- **MLflow 3.6+**, **Databricks Connect**
- **AI environment option**: adds transformers, pytorch-lightning, ray, accelerate, langchain, and more

### Supported Frameworks & Use Cases

**Frameworks:** PyTorch (DDP, FSDP), DeepSpeed, Ray, PyTorch Lightning, HuggingFace Transformers/Accelerate, Axolotl, Unsloth, MosaicML Composer

**Use cases:** LLM fine-tuning (LoRA, QLoRA, full), computer vision, single-cell genomics, digital pathology, recommender systems, reinforcement learning, distributed batch inference

---

## Two Ways to Use SGC

SGC supports two workflows for submitting GPU workloads:

| | SGCLI | Notebook Interactive |
|---|---|---|
| **Interface** | CLI tool (`sgcli`) | Databricks notebook |
| **How it works** | Define a YAML config, submit via `sgcli run` | Decorate a function with `@distributed`, call `.distributed()` |
| **Best for** | Production training, large-scale jobs | Interactive development, prototyping, small-to-medium jobs |
| **Observability** | Full log streaming, retry management, run history | Notebook cell output |
| **Error handling** | Configurable retries (`max_retries`), autoresume from checkpoints | Limited — function either succeeds or raises an exception |
| **Code management** | Git snapshots, reproducible configs | Notebook state |

### Recommendation

- **Start with notebooks** (`@distributed`) for prototyping and interactive development
- **Move to SGCLI** for production training workloads that need better observability, retry logic, and checkpoint recovery

---

## Repository Structure

```
sgc-onboarding/
├── README.md                              # This file
│
├── sgcli/                           # SGCLI workflow examples
│   ├── README.md                          # Detailed SGCLI setup and usage guide
│   ├── hello_world/                       # Minimal SGCLI example
│   │   ├── train.yaml                     # Workload definition
│   │   ├── train.py                       # Simple distributed training script
│   │   ├── dependencies.yaml              # Python dependencies
│   │   └── commands.sh                    # Entry script
│   └── geneformer_pretrain/               # Real-world example: Geneformer pretraining
│       └── README.md                      # Guide + link to full example repo
│
└── notebook_interactive/                  # Interactive notebook workflow examples
    ├── README.md                          # Detailed notebook workflow guide
    ├── hello_world/                       # Minimal @distributed example
    │   └── hello_world_distributed.py     # Databricks notebook
    └── cifar10_classification/            # Real-world example: CIFAR-10 on H100
        └── README.md                      # Guide + link to full example repo
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

1. Install SGCLI: `pip install sgcli_wheel/databricks_serverless_gpu_cli-*.whl`
2. Authenticate: `databricks auth login --host https://your-workspace.cloud.databricks.com`
3. Submit: `sgcli run -f sgcli/hello_world/train.yaml --watch`

See the [SGCLI README](sgcli/README.md) for full setup instructions.

---

## Prerequisites

- A Databricks workspace with **Serverless GPU Compute** enabled
  - Region must be `us-west-2` or `us-east-1`
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

SGC integrates with Databricks MLflow for experiment tracking. For distributed workloads, create a single MLflow run in the driver/notebook context and pass the run ID to workers:

```python
import mlflow, os

mlflow.set_experiment("/Workspace/Users/you@example.com/my_experiment")
run = mlflow.start_run(run_name="my-run")
os.environ["MLFLOW_RUN_ID"] = run.info.run_id
os.environ["MLFLOW_EXPERIMENT_NAME"] = "/Workspace/Users/you@example.com/my_experiment"
mlflow.end_run()

# Inside @distributed or SGCLI command, all nodes log to the same run:
# with mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")):
#     mlflow.log_metric("loss", loss_val)
```

### Data Loading

- **Unity Catalog Volumes** — `/Volumes/<catalog>/<schema>/<volume>/...` (recommended)
- **Delta Tables** — via Spark Connect, convert with `.toPandas()`
- **Tip:** Copy data to `/tmp` (NVMe SSD) for multi-epoch training to avoid repeated I/O

### Checkpointing

- Save checkpoints to Unity Catalog Volumes for persistence across job restarts
- SGCLI supports `autoresume` — training automatically resumes from the last checkpoint
- 7-day maximum execution time — implement checkpointing for long training runs

---

## Limitations

- **Region:** Workspace must be in `us-west-2` or `us-east-1`
- **H100 multi-node:** Not yet supported (single-node only, up to 32 GPUs)
- **A10 multi-node:** Supported up to 70 nodes, but provisioning can take up to 20 minutes
- **Runtime cap:** 7-day maximum execution
- **Pip environment size:** Max 15 GB
- **Compliance:** Not supported for HIPAA/PCI workspaces
- **PrivateLink:** Not supported

---

## Resources

- [Databricks Serverless GPU Docs](https://docs.databricks.com/aws/en/compute/serverless/gpu)
- [Multi-GPU and Multi-Node Workloads](https://docs.databricks.com/aws/en/compute/serverless/distributed-training)
- [SGC Best Practices](https://docs.databricks.com/aws/en/compute/serverless/sgc-best-practices)
- [Data Loading on SGC](https://docs.databricks.com/aws/en/compute/serverless/sgc-dataloading)
- [Serverless GPU API Reference](https://api-docs.databricks.com/python/serverless_gpu/overview.html)
- [Full Life Sciences Examples (Geneformer + GigaPath + CIFAR-10)](https://github.com/databricks-industry-solutions/sgc-examples-lifesciences)
