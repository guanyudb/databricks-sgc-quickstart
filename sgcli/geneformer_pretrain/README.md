# Geneformer Pretraining (SGCLI)

A real-world example of using SGCLI to pre-train **Geneformer**, a BERT-style foundation model for single-cell transcriptomics, on 30 million cells from the Genecorpus-30M dataset.

## Overview

| Property | Value |
|----------|-------|
| **Model** | Geneformer (BERT-style, 6 layers, 256 hidden) |
| **Dataset** | Genecorpus-30M (~30M single-cell RNA-seq samples) |
| **Framework** | MosaicML Composer + FSDP |
| **Compute** | 16x H100 GPUs (2 nodes) |
| **Data Format** | MDS (Mosaic Data Shard) streaming from Unity Catalog Volumes |
| **Logging** | MLflow with system metrics |
| **Checkpointing** | Autoresume from Unity Catalog Volumes |

## Full Example

The complete code, data preparation notebooks, and detailed instructions are available in the life sciences examples repository:

**[databricks-industry-solutions/sgc-examples-lifesciences/sgc_geneformer_pretrain](https://github.com/databricks-industry-solutions/sgc-examples-lifesciences/tree/main/sgc_geneformer_pretrain)**

## Quick Overview of the Workflow

### 1. Prepare Data (One-Time)

Run the data preparation notebook on a Databricks CPU cluster to:
- Download the Genecorpus-30M dataset from HuggingFace
- Convert to MDS streaming format
- Store in a Unity Catalog Volume

### 2. Configure

Edit `train.yaml` to set your experiment name, compute resources, and local repo path.
Edit `parameters.yaml` to set your Unity Catalog volume paths, batch sizes, and training duration.

### 3. Submit

```bash
cd SGC_geneformer
sgcli run -f train.yaml --watch
```

### Key SGCLI Features Demonstrated

- **Multi-node training** — 2 nodes of 8x H100 GPUs each, coordinated via NCCL
- **Code snapshots** — SGCLI snapshots your local code and uploads it to Databricks
- **Dependency management** — `dependencies.yaml` specifies the Python environment
- **Environment variables** — NCCL settings, MLflow config passed via `train.yaml`
- **Max retries** — automatic retry on transient failures
- **Autoresume** — training resumes from the last checkpoint after a restart
- **MLflow logging** — experiment tracking with system metrics (GPU utilization, memory)
