# Notebook Interactive Workflow

This folder demonstrates the **interactive notebook workflow** for running GPU workloads on Databricks Serverless GPU Compute using the `@distributed` decorator from the `serverless_gpu` package.

## How It Works

1. **Open** a Databricks notebook and attach it to **Serverless GPU** compute
2. **Decorate** your training function with `@distributed`
3. **Call** `.distributed()` to launch — SGC provisions GPUs, sets up NCCL, and runs your function
4. **View** results directly in the notebook cell output

```python
from serverless_gpu import distributed

@distributed(gpus=8, gpu_type='h100')
def train():
    import torch.distributed as dist
    dist.init_process_group("nccl")
    # ... your training code ...
    dist.destroy_process_group()

results = train.distributed()
```

---

## Local vs Remote Execution

The `@distributed` decorator supports two execution modes:

### `remote=False` (Default — Recommended)

Uses GPUs **attached to the current notebook** compute. This is the recommended approach:
- Better observability — output streams directly to notebook cells
- Better error handling — exceptions propagate naturally
- Faster startup — no need to provision a separate cluster

```python
@distributed(gpus=8, gpu_type='h100')           # remote=False by default
def train():
    ...
train.distributed()
```

### `remote=True`

Provisions a **separate remote GPU cluster**. Use this when:
- You need more GPUs than the notebook compute provides
- You want to keep the notebook responsive during long training

```python
@distributed(gpus=8, gpu_type='h100', remote=True)
def train():
    ...
train.distributed()
```

**Trade-offs:** Remote execution has limited observability (logs are captured as artifacts rather than streamed) and limited error handling. For large production workloads, consider using [SGCLI](../sgcli/) instead.

---

## Runtime Utilities

The `serverless_gpu.runtime` module provides helper functions:

```python
from serverless_gpu import runtime as rt

rank = rt.get_global_rank()        # Global rank across all processes
local_rank = rt.get_local_rank()   # Local GPU rank within this node
world_size = rt.get_world_size()   # Total number of processes
```

Or use standard PyTorch environment variables: `RANK`, `LOCAL_RANK`, `WORLD_SIZE`.

---

## MLflow Integration

For distributed training, create a **single MLflow run in the notebook context** and pass the run ID to all workers via environment variables. This ensures all nodes log to the same run.

```python
import mlflow, os

# Create run in notebook context
mlflow.set_experiment("/Workspace/Users/you@example.com/my_experiment")
run = mlflow.start_run(run_name="my-training-run")
os.environ["MLFLOW_RUN_ID"] = run.info.run_id
os.environ["MLFLOW_EXPERIMENT_NAME"] = "/Workspace/Users/you@example.com/my_experiment"
mlflow.end_run()

# Inside @distributed function — all nodes log to the same run
@distributed(gpus=8, gpu_type='h100')
def train():
    import mlflow, os
    run_id = os.environ.get("MLFLOW_RUN_ID")
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("loss", 0.5)

train.distributed()
```

---

## Examples

### Hello World (Start Here)

A minimal notebook that runs a simple distributed GPU operation to verify your setup.

Import [`hello_world/hello_world_distributed.py`](hello_world/hello_world_distributed.py) into your Databricks workspace and run it.

### CIFAR-10 Image Classification (Real-World)

Three notebooks demonstrating end-to-end distributed image classification:
1. Data preparation (MDS streaming format)
2. Training with PyTorch DDP
3. Training with HuggingFace Accelerate

See [`cifar10_classification/`](cifar10_classification/) for setup instructions and a link to the full example repo.

---

## Tips

- **Start with `remote=False`** — use the notebook's attached compute for better debugging
- **Move imports inside the function** — the `@distributed` decorator serializes the function with cloudpickle; large modules/datasets may exceed pickle limits
- **Use Unity Catalog Volumes** — store data at `/Volumes/<catalog>/<schema>/<volume>/` for shared access across nodes
- **Cache data locally** — copy frequently-accessed data to `/tmp` (NVMe SSD) for faster I/O
- **Log on rank 0 only** — avoid duplicate MLflow entries by checking `rank == 0` before logging
