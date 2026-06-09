"""
SGC Hello World — Minimal Distributed Training Script (SGCLI)

This script demonstrates:
1. Initializing a PyTorch distributed process group (NCCL)
2. Running a simple tensor operation on each GPU
3. Performing an all-reduce to verify inter-GPU communication
4. Logging results to MLflow

Run via SGCLI:
    sgcli run -f train.yaml --watch
"""

import os
import torch
import torch.distributed as dist
import mlflow


def log(msg: str) -> None:
    print(msg, flush=True)


def main():
    log("[Bootstrap] train.py starting")
    log(
        "[Bootstrap] env "
        f"RANK={os.environ.get('RANK')} "
        f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
        f"NODE_RANK={os.environ.get('NODE_RANK')} "
        f"MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
        f"MASTER_PORT={os.environ.get('MASTER_PORT')}"
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    log(f"[Bootstrap] setting CUDA device for local_rank={local_rank}")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    log(f"[Bootstrap] cuda device set -> {torch.cuda.get_device_name(device)}")

    log("[Bootstrap] calling init_process_group(backend=nccl)")
    dist.init_process_group(backend="nccl", device_id=device)
    log("[Bootstrap] init_process_group completed")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    node_rank = os.environ.get("NODE_RANK", "0")

    log(
        f"[Rank {rank}] Hello from node {node_rank}, "
        f"local_rank {local_rank}, world_size {world_size}, "
        f"device: {torch.cuda.get_device_name(device)}"
    )

    # Simple tensor operation on GPU
    tensor = torch.tensor([rank + 1.0], device=device)
    log(f"[Rank {rank}] Created tensor: {tensor.item()}")

    # All-reduce: sum tensors across all GPUs
    log(f"[Rank {rank}] Entering all_reduce")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    log(f"[Rank {rank}] Exited all_reduce")
    expected_sum = float(sum(range(1, world_size + 1)))
    log(f"[Rank {rank}] After all-reduce: {tensor.item()} (expected: {expected_sum})")

    # Simple matrix multiply to verify GPU compute
    log(f"[Rank {rank}] Building random matrices")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)

    log(f"[Rank {rank}] Starting timed matmul")
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    c = torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    log(f"[Rank {rank}] 1000x1000 matmul: {elapsed_ms:.2f} ms")

    # Log to MLflow (rank 0 only)
    if rank == 0:
        log(f"[Rank {rank}] Starting MLflow logging")
        active_run = mlflow.active_run()
        if active_run is not None:
            log(f"[Rank {rank}] Logging to active MLflow run: {active_run.info.run_id}")
            mlflow.log_params({
                "world_size": world_size,
                "gpu_type": torch.cuda.get_device_name(device),
            })
            mlflow.log_metrics({
                "all_reduce_result": tensor.item(),
                "matmul_time_ms": elapsed_ms,
            })
        else:
            log(f"[Rank {rank}] No active MLflow run found; creating one")
            with mlflow.start_run(run_name="hello-world"):
                mlflow.log_params({
                    "world_size": world_size,
                    "gpu_type": torch.cuda.get_device_name(device),
                })
                mlflow.log_metrics({
                    "all_reduce_result": tensor.item(),
                    "matmul_time_ms": elapsed_ms,
                })
        log(f"[Rank {rank}] Logged results to MLflow")

    # Verify all-reduce correctness
    assert tensor.item() == expected_sum, (
        f"All-reduce failed: got {tensor.item()}, expected {expected_sum}"
    )
    log(f"[Rank {rank}] All checks passed!")

    log(f"[Rank {rank}] Destroying process group")
    dist.destroy_process_group()
    log(f"[Rank {rank}] Finished cleanly")


if __name__ == "__main__":
    main()
