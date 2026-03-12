# Databricks notebook source
# MAGIC %md
# MAGIC # SGC Hello World — Interactive Notebook
# MAGIC
# MAGIC A minimal example demonstrating the `@distributed` decorator on Databricks Serverless GPU Compute.
# MAGIC
# MAGIC ### What This Notebook Does
# MAGIC
# MAGIC 1. Provisions 2 GPUs via the `@distributed` decorator
# MAGIC 2. Each GPU prints its rank, device name, and CUDA info
# MAGIC 3. Performs an all-reduce operation to verify inter-GPU communication
# MAGIC 4. Runs a simple matrix multiplication benchmark
# MAGIC 5. Logs results to MLflow
# MAGIC
# MAGIC ### Prerequisites
# MAGIC
# MAGIC - Attach this notebook to **Serverless GPU** compute (select GPU type from the dropdown)
# MAGIC - No additional packages needed — everything is pre-installed

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup MLflow

# COMMAND ----------

import mlflow
import os

# Set experiment — update this path to your own workspace path
username = spark.sql("SELECT current_user()").collect()[0][0]
experiment_name = f"/Workspace/Users/{username}/mlflow_experiments/sgc_hello_world"

mlflow.set_experiment(experiment_name)

# Create a single run — all distributed workers will log to this run
run = mlflow.start_run(run_name="hello-world-distributed")
mlflow_run_id = run.info.run_id
mlflow.end_run()

os.environ["MLFLOW_RUN_ID"] = mlflow_run_id
os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name

print(f"MLflow experiment: {experiment_name}")
print(f"MLflow run ID: {mlflow_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Libraries

# COMMAND ----------

import torch
import torch.distributed as dist
from serverless_gpu import distributed

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Distributed Function
# MAGIC
# MAGIC The `@distributed` decorator handles:
# MAGIC - Provisioning GPUs
# MAGIC - Setting up NCCL distributed environment
# MAGIC - Running the function on each GPU
# MAGIC - Collecting return values from all ranks

# COMMAND ----------

@distributed(gpus=2, gpu_type='a10')
def hello_world():
    """Minimal distributed function — runs on each GPU."""
    import os
    import torch
    import torch.distributed as dist
    import mlflow

    # Initialize distributed process group
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    gpu_name = torch.cuda.get_device_name(device)

    print(f"[Rank {rank}/{world_size}] Hello! Device: {gpu_name}, "
          f"CUDA {torch.version.cuda}, PyTorch {torch.__version__}")

    # ---- Test 1: All-reduce ----
    tensor = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))
    assert tensor.item() == expected, f"All-reduce failed: {tensor.item()} != {expected}"
    print(f"[Rank {rank}] All-reduce OK: {tensor.item()}")

    # ---- Test 2: GPU matmul benchmark ----
    a = torch.randn(2000, 2000, device=device)
    b = torch.randn(2000, 2000, device=device)

    # Warmup
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
    tflops = (2 * 2000**3) / (elapsed_ms / 1000) / 1e12
    print(f"[Rank {rank}] 2000x2000 matmul: {elapsed_ms:.2f} ms ({tflops:.2f} TFLOPS)")

    # ---- Log to MLflow (rank 0 only) ----
    if rank == 0:
        run_id = os.environ.get("MLFLOW_RUN_ID")
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME")
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_params({
                "world_size": world_size,
                "gpu_type": gpu_name,
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            })
            mlflow.log_metrics({
                "all_reduce_result": tensor.item(),
                "matmul_time_ms": elapsed_ms,
                "tflops": tflops,
            })
            print(f"[Rank {rank}] Results logged to MLflow")

    dist.destroy_process_group()

    return {
        "rank": rank,
        "gpu": gpu_name,
        "all_reduce": tensor.item(),
        "matmul_ms": elapsed_ms,
        "tflops": tflops,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run It!
# MAGIC
# MAGIC Call `.distributed()` to launch the function across all GPUs.
# MAGIC Results are returned as a list (one entry per rank).

# COMMAND ----------

results = hello_world.distributed()

print("\n" + "=" * 60)
print("RESULTS FROM ALL RANKS")
print("=" * 60)
for r in results:
    print(f"  Rank {r['rank']}: {r['gpu']} — {r['matmul_ms']:.2f} ms ({r['tflops']:.2f} TFLOPS)")
print("=" * 60)
print("Hello World complete!")
