# SGCLI Workflow

This folder demonstrates the **SGCLI workflow** for submitting GPU training jobs to Databricks Serverless GPU Compute. SGCLI is the recommended approach for production training workloads that need log streaming, retry management, and checkpoint recovery.

## How SGCLI Works

1. **Define** your workload in a YAML configuration file (`train.yaml`)
2. **Submit** the job via `sgcli run -f train.yaml --watch`
3. SGCLI **snapshots** your local code, uploads it to Databricks, provisions GPU nodes, installs dependencies, and runs your training command
4. **Monitor** via streamed logs, MLflow, or `sgcli get status`

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Local Code  │────>│  SGCLI Snapshot   │────>│  Databricks SGC      │
│  + train.yaml│     │  Upload to DBFS   │     │  H100/A10 GPU Nodes  │
└──────────────┘     └──────────────────┘     │  - Install deps       │
                                               │  - Run command         │
                                               │  - Stream logs         │
                                               └──────────────────────┘
```

---

## Setup

### Step 1: Install Databricks CLI

```bash
# macOS
brew install databricks

# Or via pip
pip install databricks-cli
```

### Step 2: Authenticate

```bash
databricks auth login --host https://your-workspace.cloud.databricks.com
```

This creates `~/.databrickscfg` with your credentials. If you have multiple profiles:

```bash
export DATABRICKS_CONFIG_PROFILE=MY_PROFILE
```

### Step 3: Install SGCLI

SGCLI is distributed as an internal wheel package. Install the latest version:

```bash
pip install /path/to/databricks_serverless_gpu_cli-*.whl --force-reinstall
sgcli --version
```

### Step 4: Verify

```bash
sgcli --help
```

---

## Examples

### Hello World (Start Here)

A minimal example that runs a simple PyTorch distributed operation on 2 H100 GPUs to verify your setup works.

```bash
cd hello_world
sgcli run -f train.yaml --watch
```

**What it does:**
- Provisions 2 H100 GPUs
- Initializes NCCL distributed process group
- Each GPU prints its rank, device name, and runs a simple tensor operation
- Logs results to MLflow

See [`hello_world/`](hello_world/) for the full example.

### Geneformer Pretraining (Real-World)

Pre-train a BERT-style genomics model (Geneformer) on 30M single-cell transcriptomics samples using MosaicML Composer with FSDP across 16 H100 GPUs (2 nodes).

See [`geneformer_pretrain/`](geneformer_pretrain/) for setup instructions and a link to the full example repo.

---

## YAML Configuration Reference

The `train.yaml` file defines your SGCLI workload:

```yaml
# Required: unique experiment name
experiment_name: my-experiment

# Optional: human-readable run name
run_name: my-run-001

# Environment configuration
environment:
  env_variables:
    MY_VAR: "value"
    NCCL_DEBUG: "INFO"                        # Useful for debugging distributed issues
    NCCL_TIMEOUT: "1800"                      # 30 min timeout for multi-node
  dependencies: dependencies.yaml             # Pip/conda dependencies

# Compute resources
compute:
  gpus: 8                                     # Total GPUs (8 per H100 node)
  gpu_type: h100                              # 'a10' or 'h100'

# Retry configuration (for fault tolerance)
max_retries: 3                                # Auto-retry on failure

# Code source
code_source:
  type: snapshot
  snapshot:
    repo_path: /path/to/local/repo            # Local code directory
    allow_uncommitted: true                   # Include uncommitted changes
    # git_branch: main                        # Or pin to a specific branch
    # git_commit: abc123                      # Or pin to a specific commit

# Training command (runs on each node)
command: |-
  cd $HOME/my_project
  pip install -r requirements.txt
  python train.py --epochs 10
```

### Dependencies YAML

```yaml
version: "4"                                  # SGC environment version
dependencies:
  - torch==2.8.0
  - mlflow>=3.6.0
  - transformers==4.44.0
```

### Environment Variables Provided by SGC

These are automatically set on each node:

| Variable | Description |
|----------|-------------|
| `RANK` | Global rank of this process |
| `LOCAL_RANK` | Local GPU rank within this node |
| `WORLD_SIZE` | Total number of processes |
| `NODE_RANK` | Node index (0, 1, 2, ...) |
| `MASTER_ADDR` | IP address of the rank-0 node |
| `MASTER_PORT` | Port for distributed coordination |

---

## Common SGCLI Commands

```bash
# Submit a job and stream logs
sgcli run -f train.yaml --watch

# Submit with a specific profile
sgcli run -f train.yaml --watch --profile MY_PROFILE

# Dry run (validate config without submitting)
sgcli run -f train.yaml --dry-run

# List recent runs
sgcli get runs

# Check run status
sgcli get status <run-id>

# Fetch run logs
sgcli get logs <run-id>
```

---

## Tips

- **Start small:** Test with `gpus: 2` and a small dataset before scaling up
- **Use `--watch`:** Always use `--watch` to stream logs in real-time
- **Enable autoresume:** For long training runs, save checkpoints to Unity Catalog Volumes and configure `max_retries` for automatic recovery
- **NCCL debugging:** Set `NCCL_DEBUG: "INFO"` in env_variables to diagnose distributed communication issues
- **MLflow:** Log metrics to MLflow for experiment tracking — see the hello_world example for setup
