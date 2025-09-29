import argparse
from utils.experiment_runner import load_config, run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Federated Learning experiments")
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--dataset', help='Override dataset name from config')
    parser.add_argument('--model', help='Override model architecture')
    parser.add_argument('--algorithm', choices=['fedavg', 'fedavg_kd'], help='Override algorithm from config')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dataset:
        config['dataset_name'] = args.dataset
    if args.model:
        config['model_name'] = args.model
    if args.algorithm:
        config['algorithm'] = args.algorithm

    run_experiment(config)
