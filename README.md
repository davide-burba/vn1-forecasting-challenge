# VN1 Forecasting challenge

This repository contains the code used for the [VN1 Forecasting challenge](https://www.datasource.ai/en/users/davide-burba/competitions/phase-2-vn1-forecasting-accuracy-challenge/datathon_detail/description).


## Environment 

The environment and dependencies are managed with [uv](https://docs.astral.sh/uv/getting-started/installation/) (click on the link for installation instructions).

The only kind of model used is [LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html). You might need to install additional libraries like [libomp](https://formulae.brew.sh/formula/libomp) to make it work, depending on your system.

The projects makes use of [MLflow](https://mlflow.org/) to track experiments. You can visualize the MLflow UI by running inside the project folder:

```bash
uv run mlflow ui
```


## Build a submission

Submissions for both competition phases were built using the scripts available in `./scripts`. Run the script with `--help` to check the expected argument.

The final submission was generated with:
```bash
uv run python scripts/run_phase_2_ensemble.py --n_estimators 30
```
It uses the default configuration ([config.yaml](./config.yaml)).

## Content

- [scripts](./scripts/): the python scripts to run time cross-validation and generate submissions, and an additional one to ensamble multiple identical models varying the seed.
- [notebooks](./notebooks/): notebooks used for exploration, quick tests, and generating configuration files.
- [vn1](./vn1): codebase
- [config.yaml](./config.yaml): the default configuration (used to build the final submission).
- [configs](./configs/): configurations used for experiments and fine-tuning. 
- [run_folder_config.sh](./run_folder_config.sh): bash script to loop over a folder of configurations.
