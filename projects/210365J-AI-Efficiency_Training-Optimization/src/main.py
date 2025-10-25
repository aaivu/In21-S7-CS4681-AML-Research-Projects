#!/usr/bin/env python3
"""
Main execution script for Compute Scaling Research
"""

import argparse
from run_experiments import run_resnet_experiments, run_gpt2_experiments
from analyze_results import analyze_resnet_results, analyze_gpt2_results

def main():
    parser = argparse.ArgumentParser(description='Run compute scaling experiments')
    parser.add_argument('--run-experiments', action='store_true', help='Run training experiments')
    parser.add_argument('--analyze', action='store_true', help='Analyze results')
    parser.add_argument('--all', action='store_true', help='Run everything')
    
    args = parser.parse_args()
    
    if args.run_experiments or args.all:
        print("Running all experiments...")
        run_resnet_experiments()
        run_gpt2_experiments()
    
    if args.analyze or args.all:
        print("Analyzing results...")
        analyze_resnet_results()
        analyze_gpt2_results()

if __name__ == "__main__":
    main()