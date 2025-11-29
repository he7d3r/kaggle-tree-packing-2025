# Santa 2025 - Christmas Tree Packing Challenge

Source: <https://www.kaggle.com/competitions/santa-2025/overview>.

## Setup

### First time

```shell
pyenv install 3.11.13
pyenv local 3.11.13

# Create and populate the virtual environment from `pyproject.toml`
uv venv
uv sync
```

## Activate the environment
```shell
source .venv/bin/activate
```

## Start the MLFlow UI

Run
```shell
mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns.db
```
and open the informed URL <http://127.0.0.1:5000>.

## Run
The main script supports two arguments:
- `--mlflow`: log submission score to MLflow
- `--draft`: skip saving the submission to a CSV file
This allows using any of the following:
```shell
python main.py
python main.py --draft
python main.py --draft --mlflow
python main.py --mlflow
```
