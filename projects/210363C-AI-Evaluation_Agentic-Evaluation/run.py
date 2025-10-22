from config import MODEL_ID, MAX_TOKENS, CODE_EXEC_ENABLED, DATA_PATH, SEED_VALUES, TEMPERATURE_VALUES
print("Using model:", MODEL_ID)  

from src.utils.data_loader import load_dataset
from src.evaluator.evaluator import evaluate_dataset
from src.utils.logger import setup_logger, log_run_details
from datetime import datetime
import pandas as pd
import os
import time  # Import time module for sleep functionality

def main():
    # Setup results directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', f'results_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)

    log_dir = os.path.join('experiments', 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Load dataset
    df = load_dataset(DATA_PATH)

    # Initialize a list to store summary data
    summary_data = []

    # Iterate over SEED and TEMPERATURE combinations
    for seed in SEED_VALUES:
        for temperature in TEMPERATURE_VALUES:
            # Setup logger for the current run
            run_id = f"{timestamp}_seed{seed}_temp{temperature}"
            logger = setup_logger(run_id, log_dir)

            # Evaluate the dataset
            print(f"Running evaluation with SEED={seed} and TEMPERATURE={temperature}")
            results = evaluate_dataset(df.iloc[:5])  # Evaluate the first five rows (0-based index)
            print(results.to_string())  

            # Log run details
            log_run_details(
                logger,
                run_id=run_id,
                problem_id=DATA_PATH,
                seed=seed,
                model=MODEL_ID,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                code_exec_enabled=CODE_EXEC_ENABLED
            )

            # Save results to a CSV file inside the timestamped results folder
            results_file = os.path.join(results_dir, f"results_seed{seed}_temp{temperature}.csv")
            results.to_csv(results_file, index=False)
            print(f"Results saved to: {results_file}")

            # Append required metrics to the summary data list
            for idx, row in results.iterrows():
                summary_data.append({
                    'seed': seed,
                    'temperature': temperature,
                    'gold_answer': row.get('gold_answer', None),
                    'final_answer': row.get('final_answer', None),
                    'correct': row.get('correct', None),
                    'api_tries': row.get('api_tries', None),
                    'format_retries_used': row.get('format_retries_used', None),
                    'llm_calls': row.get('llm_calls', None),
                    'total_tokens': row.get('total_tokens', None),
                    'walltime_sec': row.get('walltime_sec', None),
                    'exec_successes': row.get('exec_successes', None),
                    'TPS': row.get('TPS', None),
                    'EPS': row.get('EPS', None),
                    'TE': row.get('TE', None),
                    'CE': row.get('CE', None),
                    'LE': row.get('LE', None)
                })

                # Sleep for 1 minute after every 5 rows
                if (idx + 1) % 5 == 0:
                    print("Sleeping for 1 minute...")
                    time.sleep(60)

    # Create a summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(results_dir, "results_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
