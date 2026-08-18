# PopPert

Population-level joint-distribution modeling for single-cell perturbation prediction.

This repository contains the official research implementation of **PopPert**, a model that predicts how a perturbation changes a population of cells. Instead of learning a one-to-one mapping between control and perturbed cells, PopPert models gene-wise expression distributions together with cross-gene dependence and generates a virtual perturbed cell population.

> **Release status:** research code. Dataset files and pretrained checkpoints are not included in the repository.

## Highlights

- Models zero inflation and non-negative expression with zero-inflated truncated Gaussian mixtures (ZI-GMMs).
- Captures cross-gene dependence with a factor-decomposed Gaussian copula.
- Includes an alternative zero-inflated negative binomial (ZINB) marginal model.


## Method overview

PopPert first summarizes control and perturbed cell populations as distribution parameters. A Transformer then predicts the residual change from the control population to the target perturbation. During generation, the predicted marginals and copula factors are sampled jointly to reconstruct a virtual perturbed population.

## Repository structure

```text
.
|-- configs/                 # Fixed train/validation/test split definitions
|-- tools/
|   `-- compute_de_emd.py    # EMD and energy-distance evaluation
|-- zinb/                    # Alternative ZINB implementation
|-- dataset.py               # PyTorch dataset for population distributions
|-- gmm_utils.py             # ZI-GMM and Gaussian-copula utilities
|-- inference.py             # Inference and AnnData export
|-- model.py                 # PopPert model architecture
|-- preprocess.py            # AnnData-to-distribution preprocessing
|-- sampling.py              # Training losses and population sampling
|-- train.py                 # Training loop
|-- run.sh                   # End-to-end pipeline
|-- requirements.txt         # Python dependencies
`-- vocab.json               # Gene vocabulary
```

## Requirements

- Python 3.10 or later
- PyTorch 2.0 or later
- A CUDA-capable GPU is strongly recommended
- Linux, macOS, or Windows with WSL2 for the `run.sh` pipeline

The chemical-perturbation workflow also requires RDKit. Evaluation requires `cell-eval`.

## Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/whd1125/PopPert.git
cd PopPert
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows, run these commands in WSL2. For a CUDA build of PyTorch, follow the platform-specific command on the [PyTorch installation page](https://pytorch.org/get-started/locally/) before installing the remaining requirements.

## Data preparation

The raw datasets are not redistributed here. Obtain each dataset from its original provider and make sure that its license permits your intended use. Place the AnnData files in the following layout:

```text
data/
|-- replogle/
|   `-- replogle.h5ad
|-- adamson/
|   `-- adamson.h5ad
|-- norman/
|   `-- norman.h5ad
`-- sciplex/
    `-- sciplex.h5ad
```

The included presets expect these main `adata.obs` fields:

| Dataset | Perturbation field | Context field | Control value | Additional fields |
| --- | --- | --- | --- | --- |
| Replogle | `gene` | `cell_line` (`hepg2`) | `non-targeting` | - |
| Adamson | `gene` | `cell_line` (`K562`) | `ctrl` | - |
| Norman | `gene` | `cell_line` (`K562`) | `ctrl` | - |
| sci-Plex | `condition` | `cell_type` | `control` | `dose_val`, `SMILES`, `split_ood_finetuning` |

If your files use different column names or control labels, run the individual scripts with the appropriate command-line arguments instead of relying on the presets.

## Quick start

Run preprocessing, training, inference, and evaluation for one of the supported datasets:

```bash
bash run.sh norman --device cuda:0
```

Supported dataset names are:

```text
replogle  adamson  norman  sciplex
```

Common options include:

```bash
# Change training settings
bash run.sh adamson --device cuda:0 --epochs 100 --batch_size 16

# Run without the copula component
bash run.sh adamson --no_copula

# Use the ZINB marginal model (genetic perturbations only)
bash run.sh norman --marginal zinb

# Reuse preprocessed data or an existing checkpoint
bash run.sh norman --skip_preprocess
bash run.sh norman --skip_preprocess --skip_train

# Skip cell-eval
bash run.sh norman --skip_eval
```

Display all pipeline options with:

```bash
bash run.sh --help
```

The default experiment uses two mixture components, 5,000 highly variable genes, copula rank 32, dependence-loss weight 0.2, and a population batch size of 128 during preprocessing.

## Outputs

The pipeline writes generated artifacts under `data/processed/` and `results/`:

```text
data/processed/processed_<dataset>.pt
results/<dataset>/<run_name>/checkpoints/best_model.pt
results/<dataset>/<run_name>/inference/inference_pred.h5ad
results/<dataset>/<run_name>/inference/inference_real.h5ad
results/<dataset>/<run_name>/cell_eval/
```

These paths are excluded from Git because datasets, checkpoints, and experiment outputs can be large. Publish model weights through a GitHub Release or a model-hosting service rather than committing them directly to the repository.

## Optional pretrained gene embeddings

Both training implementations look for an optional checkpoint at:

```text
embedding_weight/best_model.pt
```

If it is absent, the code reports a warning and trains the embedding layer from its initialized values. If you release this checkpoint separately, place it at the path above or pass `--pretrin_embedding_path` to the training script.

## Running stages separately

For custom datasets or paths, inspect the available options for each stage:

```bash
python preprocess.py --help
python train.py --help
python inference.py --help
```

This is the recommended route when the AnnData schema does not match a built-in dataset preset.

## Reproducibility notes

- The training scripts initialize Python, NumPy, and PyTorch with seed 42.
- Fixed dataset splits are stored under `configs/`.
- Exact reproducibility can still depend on the GPU, CUDA/cuDNN version, dependency versions, and input dataset preprocessing.
- Save the console output and generated model configuration together with each experiment.

## Citation

If you use PopPert in your work, please cite the accompanying paper:

```text
PopPert: Population-level Joint-Distribution Modeling for
Single-Cell Perturbation Prediction.
```

The BibTeX entry will be added after publication.

## License

A software license has not yet been selected for this release. Add a `LICENSE` file before treating the repository as open source or accepting reuse and redistribution.

## Acknowledgements

Evaluation is supported through the open-source [`cell-eval`](https://github.com/ArcInstitute/cell-eval) toolkit.
