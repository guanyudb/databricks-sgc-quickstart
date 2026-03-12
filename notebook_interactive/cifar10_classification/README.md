# CIFAR-10 Image Classification (Interactive Notebooks)

A real-world example of using the `@distributed` decorator to train **ResNet-18** on CIFAR-10 with 8x H100 GPUs, demonstrating both PyTorch DDP and HuggingFace Accelerate approaches.

## Overview

| Property | Value |
|----------|-------|
| **Model** | ResNet-18 (10 classes) |
| **Dataset** | CIFAR-10 (60K images, MDS streaming format) |
| **Frameworks** | PyTorch DDP, HuggingFace Accelerate |
| **Compute** | 8x H100 GPUs via `@distributed` |
| **Data Format** | MDS (Mosaic Data Shard) streaming from Unity Catalog Volumes |
| **Logging** | MLflow with system metrics |

## Notebooks

| Notebook | Description |
|----------|-------------|
| **01_Prepare_Streaming_Dataset** | Download CIFAR-10, convert to MDS format, store in UC Volume |
| **02_Distributed_Training_H100_DDP** | Train with PyTorch DDP — manual process group, DDP wrapping |
| **03_Distributed_Training_H100_Accelerate** | Train with HF Accelerate — simpler API, automatic mixed precision |

## Full Example

The complete notebooks are available in the life sciences examples repository:

**[databricks-industry-solutions/sgc-examples-lifesciences/sgc_cifar10_image_classification](https://github.com/databricks-industry-solutions/sgc-examples-lifesciences/tree/main/sgc_cifar10_image_classification)**

## Quick Start

1. Import the 3 notebooks into your Databricks workspace
2. Run Notebook 01 on a CPU cluster to prepare the streaming dataset
3. Update the configuration cell (catalog, schema, volume) in Notebooks 02/03
4. Run Notebook 02 or 03 to launch distributed training

## Key Patterns Demonstrated

- **MLflow run sharing** — create a single run in the notebook, pass `MLFLOW_RUN_ID` via env vars so all distributed nodes log to the same run
- **MDS streaming** — efficient data loading with automatic distributed sharding
- **DDP vs Accelerate** — side-by-side comparison of two distributed training approaches
- **Model checkpointing** — log model artifacts to MLflow at each epoch
