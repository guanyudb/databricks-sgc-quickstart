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


def main():
    # Initialize distributed process group
    # SGC automatically sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    node_rank = os.environ.get("NODE_RANK", "0")

    # Set the GPU for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Rank {rank}] Hello from node {node_rank}, "
          f"local_rank {local_rank}, world_size {world_size}, "
          f"device: {torch.cuda.get_device_name(device)}")

    # Simple tensor operation on GPU
    tensor = torch.tensor([rank + 1.0], device=device)
    print(f"[Rank {rank}] Created tensor: {tensor.item()}")

    # All-reduce: sum tensors across all GPUs
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected_sum = sum(range(1, world_size + 1))
    print(f"[Rank {rank}] After all-reduce: {tensor.item()} (expected: {expected_sum})")

    # Simple matrix multiply to verify GPU compute
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    c = torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    print(f"[Rank {rank}] 1000x1000 matmul: {elapsed_ms:.2f} ms")

    # Log to MLflow (rank 0 only)
    if rank == 0:
        mlflow.set_experiment("/mlflow_experiments/sgc_hello_world")
        with mlflow.start_run(run_name="hello-world"):
            mlflow.log_params({
                "world_size": world_size,
                "gpu_type": torch.cuda.get_device_name(device),
            })
            mlflow.log_metrics({
                "all_reduce_result": tensor.item(),
                "matmul_time_ms": elapsed_ms,
            })
            print(f"[Rank {rank}] Logged results to MLflow")

    # Verify all-reduce correctness
    assert tensor.item() == expected_sum, (
        f"All-reduce failed: got {tensor.item()}, expected {expected_sum}"
    )
    print(f"[Rank {rank}] All checks passed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
