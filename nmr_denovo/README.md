# *De novo* structure generation

This project provides various models for de novo molecular generation from NMR spectra, including Transformer- and BART-based architectures that decode standardized SMILES or SELFIES sequences from continuous spectral embeddings.

## Requirements

### 1. Create Conda Environment

Navigate to the project directory:

```bash
cd NMREluBench/nmr_denovo
```

Create the environment:

```bash
conda env create -f environment.yml
```

### 2. Activate the Environment

```bash
conda activate nmrdenovo
```

## Training

To start training, run:

```bash
bash nmr_denovo_train.sh
```

Make sure training-related paths and data files are correctly configured inside the script.

## Evaluation

To evaluate the model on the test set:

```bash
bash nmr_denovo_eval.sh
```

The evaluation metrics include the overall validity rate, top-1 and top-10 success rates, MCES distance, and Tanimoto similarity.

