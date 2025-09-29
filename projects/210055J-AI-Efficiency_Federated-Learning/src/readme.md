
This project enhances the Federated Averaging (FedAvg) algorithm with a knowledge distillation regularizer to reduce client drift and improve global model accuracy on non-IID data.

## Setup Steps

1.  **Create Virtual Environment**
    -   Create a virtual environment if you don't have one installed:
        ```bash
        python -m venv env
        ```

2. **Activate the Virtual Environment**
    -   Activate the virtual environment using the following command (Windows):
        ```bash
        .\env\Scripts\activate
        ```
    -   For macOS/Linux, use:
        ```bash
        source env/bin/activate
        ```
        
3. **Install Dependencies**
    -   Install the required Python packages by running:
        ```bash
        pip install -r requirements.txt
        ```

4. **View MLflow Results**
    - After running an experiment, launch the MLflow UI:
        ```bash
        mlflow ui --backend-store-uri mlruns
        ```
    - Open `http://localhost:5000` in your browser to explore runs, metrics, and model artifacts.

## Running Experiments

The `algorithm` configuration flag selects between the baseline and the improved approach:

* `fedavg` – standard Federated Averaging.
* `fedavg_kd` – FedAvg with a knowledge distillation regularizer.

You can change the value in `config/config.yaml` or override it on the command line:

```bash
# Baseline FedAvg
python run_experiment.py --algorithm fedavg
```
```bash
# Improved FedAvg with knowledge distillation
python run_experiment.py --algorithm fedavg_kd
```

## Expected Behavior on Non-IID Data

Non-IID client data often causes the baseline FedAvg to drift, reducing global model accuracy. The knowledge distillation variant is designed to counter this drift by aligning client models, typically leading to higher accuracy on non-IID splits. Metrics and model artifacts for both variants are logged to the `mlruns/` directory and can be compared in the MLflow UI.
