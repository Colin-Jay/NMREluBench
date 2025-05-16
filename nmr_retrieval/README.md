# NMR Retrieval

This project provides various methods for NMR data retrieval, including Random, Wasserstein, Vector, and CReSS methods.

## Requirements

Ensure the dependencies are installed. Install them using:

```bash
pip install -r requirements.txt
```

## Installing CReSS

To use the CReSS method, run the following script:

```bash
bash install_cress.sh
```

## Dataset & Models

Put the dataset in the `data` folder and the models in the `model` folder.

## Data Preprocessing
To preprocess the data, run the following script:

```bash
python data_preprocess.py
```

## Running Experiments from the Paper

To reproduce the experiments described in the paper, run the following script:

```bash
python run_h.py
python run_c.py
python run_hc.py
```

## Notes

- Ensure the CUDA environment is properly configured if using GPU.
- For the CReSS method, update `config_path` and `pretrain_model_path` with the correct paths.

