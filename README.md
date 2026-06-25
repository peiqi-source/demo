# AHD-EC

Code-only release for the paper:

**Efficient Ensemble Clustering via Diverse Anchor-Based High-Order Graphs and Adaptive Discrete Consensus Fusion**

Authors: Hang Guo, Jinyang Zhai, Haitao Nie, Zihua Zhao, Rong Wang, and Feiping Nie.

## Overview

Anchor-Based High-Order Diverse Ensemble Clustering (AHD-EC) improves ensemble clustering from two sides:

1. It generates diverse base partitions from anchor-based high-order bipartite graphs.
2. It learns the final discrete consensus labels with Adaptive Discrete Consensus Fusion (ADCF), using cluster-level overlap statistics instead of explicitly materializing an `n x n` co-association matrix.

This repository intentionally keeps only source code and project instructions. Result tables, paper figures, generated plots, cached experiment outputs, and dataset `.mat` files are not included.

## Repository Layout

| Path | Description |
| --- | --- |
| `utils/AHD_EC.m` | Main AHD-EC algorithm entry point. |
| `utils/ADCF.m` | Adaptive Discrete Consensus Fusion backend. |
| `utils/*.m` | Shared clustering, anchor graph, optimization, data-loading, and evaluation utilities. |
| `experiments/` | Main AHD-EC grid search, hyperparameter, ablation, and scalability scripts. |
| `comparison/` | Baseline ensemble clustering source code and runners. |
| `datasets/*.m`, `datasets/*.py` | Dataset and base-partition generation scripts. |
| `plot/*.py` | Plotting scripts only; generated figures are ignored. |
| `setup_paths.m` | MATLAB path setup helper. |

## Requirements

MATLAB:

- A recent MATLAB release.
- Statistics and Machine Learning Toolbox is recommended because the code uses routines such as `pdist2` and clustering helpers.

Python, for selected baselines, dataset generation, and plotting:

```bash
pip install numpy scipy pandas scikit-learn matplotlib seaborn
```

## Data Preparation

Place benchmark `.mat` files under `datasets/ec_data/` before running the main experiments. The data loaders expect the following dataset names:

| Index | Dataset file |
| --- | --- |
| 1 | `Umist.mat` |
| 2 | `VS.mat` |
| 3 | `COIL20.mat` |
| 4 | `SPF.mat` |
| 5 | `IS.mat` |
| 6 | `FCT.mat` |
| 7 | `MNIST.mat` |
| 8 | `OpticDigits.mat` |
| 9 | `LS.mat` |
| 10 | `ISOLET.mat` |
| 11 | `USPS.mat` |
| 12 | `PenDigits.mat` |

Large MNIST scalability subsets can be generated locally with:

```bash
cd datasets
python datasets_download.py
```

Generated `.mat`, `.csv`, `.pdf`, image, and result directories are ignored by Git.

## Quick Start

After preparing data, run a compact AHD-EC example in MATLAB:

```matlab
cd path/to/demo
setup_paths

[X, Y] = loaddata_small(1);     % 1 = UMIST
X = X ./ max(X, [], 2);
X(isnan(X)) = 0;

c = length(unique(Y));
k = 5;
order = 3;
num_sampling = 3;
anchor_rate = 20;
delta = 5;
anchors = (anchor_rate + (0:num_sampling-1) * delta) * c;

[pred, obj, runtime, alpha_history] = AHD_EC(k, order, X, anchors, c);
[ACC, NMI, Purity, Fscore, ~, ~, ~, ARI] = ClusteringMeasure4(Y, pred);

fprintf('ACC %.4f | NMI %.4f | Purity %.4f | Fscore %.4f | ARI %.4f | Time %.2fs\n', ...
    ACC, NMI, Purity, Fscore, ARI, runtime);
```

## Reproducing Experiments

Main AHD-EC parameter search:

```matlab
setup_paths
run experiments/run_AHD_GridSearch.m
```

Ablation studies:

```matlab
setup_paths
cd experiments/Ablation
run run_AHD_Ablation_Frontend.m
run run_AHD_Ablation_Backend.m
```

Scalability study:

```matlab
setup_paths
cd experiments/Scalability
run run_AHD_Scalability.m
```

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{guo2026ahdec,
  title  = {Efficient Ensemble Clustering via Diverse Anchor-Based High-Order Graphs and Adaptive Discrete Consensus Fusion},
  author = {Guo, Hang and Zhai, Jinyang and Nie, Haitao and Zhao, Zihua and Wang, Rong and Nie, Feiping},
  year   = {2026},
  note   = {Manuscript}
}
```

The venue and DOI will be updated after publication.
