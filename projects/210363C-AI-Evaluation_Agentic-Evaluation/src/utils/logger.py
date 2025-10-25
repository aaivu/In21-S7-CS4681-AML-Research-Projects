import logging
import os
from datetime import datetime
from config import SEED, MODEL_ID, TEMPERATURE, MAX_TOKENS, CODE_EXEC_ENABLED

def setup_logger(run_id: str, log_dir: str = os.path.join('experiments', 'logs')):
    """
    Sets up a logger to log run details.

    Args:
        run_id (str): Unique identifier for the run.
        log_dir (str): Directory to store log files. Defaults to "logs".

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log file for the run
    log_file = os.path.join(log_dir, f"run_{run_id}.log")

    # Configure the logger
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.DEBUG)

    # Formatter for log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def log_run_details(logger, run_id, problem_id, **metrics):
    """
    Logs the details of a run.

    Args:
        logger (logging.Logger): Logger instance.
        run_id (str): Unique identifier for the run.
        problem_id (str): Identifier for the problem being solved.
        **metrics: Additional metrics to log.
    """
    logger.info("Run Details:")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Problem ID: {problem_id}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Seed: {SEED}")
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Temperature: {TEMPERATURE}")
    logger.info(f"Max Tokens: {MAX_TOKENS}")
    logger.info(f"Code Execution Enabled: {CODE_EXEC_ENABLED}")

    if metrics:
        logger.info("Additional Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")