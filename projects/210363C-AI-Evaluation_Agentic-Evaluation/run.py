from config import SEED, MODEL_ID, TEMPERATURE, MAX_TOKENS, CODE_EXEC_ENABLED
print("Using model:", MODEL_ID)  

from config import DATA_PATH
from src.utils.data_loader import load_dataset
from src.evaluator.evaluator import evaluate_dataset
from src.utils.logger import setup_logger, log_run_details
from datetime import datetime
import pandas as pd
import os

def main():
    # Setup logger for the run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = load_dataset(DATA_PATH)
    results = evaluate_dataset(df.head(3))   # or full dataset
    print(results.to_string())               # pretty print

    # Save results with run_id in the filename
    results_dir = os.path.join('results')
    os.makedirs(results_dir, exist_ok=True)
    results.to_csv(os.path.join(results_dir, f"results_{run_id}.csv"), index=False)

    # Log run details after evaluation
    logger = setup_logger(run_id)
    log_run_details(
        logger,
        run_id=run_id,
        problem_id=DATA_PATH,  
        seed=SEED,
        model=MODEL_ID,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        code_exec_enabled=CODE_EXEC_ENABLED
    )

    # Output log file name
    print(f"Log run at: logs/run_{run_id}.log")

log_dir = os.path.join('experiments', 'logs')
os.makedirs(log_dir, exist_ok=True)

if __name__ == "__main__":
    main()
