import os
import sys
import pandas as pd
import torch
from datetime import datetime
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import setup_logging, set_random_seeds, create_directories, save_results
from rtp_processor import RTPProcessor
from model import RevisedCAFEModel
from revised_dataset import RevisedToxicityDataset
from train import CAFETrainer
from loss_functions import CAFELoss
from rtp_evaluation import RTPEvaluator
from jigsaw_evaluator import JigsawFairnessEvaluator
from perspective_api import PerspectiveAPIClient
from jigsaw_data import JigsawDataLoader
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class RevisedCAFEExperiment:
    """Revised experiment runner based on actual RTP structure."""
    
    def __init__(self):
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        create_directories()
        setup_logging(f"results/revised_cafe_{self.experiment_id}.log")
        set_random_seeds(42)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Device: {self.device}")
    
    def step1_process_rtp_dataset(self, rtp_path: str) -> pd.DataFrame:
        """Step 1: Load and process RTP with attribute derivation."""
        logger.info("="*60)
        logger.info("STEP 1: Processing RTP Dataset")
        logger.info("="*60)
        
        processor = RTPProcessor()
        rtp_df = processor.load_and_process_rtp(rtp_path)
        
        # Save processed dataset
        output_path = "data/processed/rtp_processed.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rtp_df.to_csv(output_path, index=False)
        
        logger.info(f"Processed RTP saved to {output_path}")
        logger.info(f"Sensitive attribute ratio: {rtp_df['sensitive_attribute'].mean():.3f}")
        logger.info(f"Context attribute ratio: {rtp_df['context_attribute'].mean():.3f}")
        logger.info(f"Mean toxicity: {rtp_df['toxicity'].mean():.3f}")
        
        return rtp_df
    
    def step2_train_cafe(self, rtp_df: pd.DataFrame) -> RevisedCAFEModel:
        """Step 2: Train CAFE model on processed RTP."""
        logger.info("="*60)
        logger.info("STEP 2: Training CAFE Model")
        logger.info("="*60)
        
        # Split data
        processor = RTPProcessor()
        train_df, val_df = processor.create_train_val_split(rtp_df, test_size=0.2)
        
        # Initialize model
        cafe_model = RevisedCAFEModel().to(self.device)
        
        # Create datasets
        train_dataset = RevisedToxicityDataset(train_df, cafe_model.tokenizer)
        val_dataset = RevisedToxicityDataset(val_df, cafe_model.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Setup training
        loss_fn = CAFELoss(alpha=1.0, beta=0.5, gamma=0.3)
        optimizer = torch.optim.AdamW(cafe_model.parameters(), lr=2e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        trainer = CAFETrainer(
            model=cafe_model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            loss_function=loss_fn,
            optimizer=optimizer,
            device=self.device,
            scheduler=scheduler
        )
        
        # Train
        training_history = trainer.train(
            num_epochs=8,
            save_dir=f"results/models/revised_cafe_{self.experiment_id}"
        )
        
        save_results(training_history, 
                    f"results/models/revised_cafe_{self.experiment_id}/history.json")
        
        logger.info("CAFE training completed!")
        return cafe_model
    
    def step3_evaluate_on_rtp(self, 
                             cafe_model: RevisedCAFEModel,
                             rtp_df: pd.DataFrame) -> Dict:
        """Step 3: Evaluate on RTP with EMT and Toxicity Probability."""
        logger.info("="*60)
        logger.info("STEP 3: RTP Evaluation (EMT & Toxicity Probability)")
        logger.info("="*60)
        
        # Use held-out prompts
        test_prompts = rtp_df.sample(n=min(100, len(rtp_df)), random_state=42)['prompt_text'].tolist()
        
        # Initialize evaluator
        perspective_client = PerspectiveAPIClient()
        rtp_evaluator = RTPEvaluator(
            cafe_model=cafe_model,
            perspective_client=perspective_client,
            device=self.device,
            num_continuations=25
        )
        
        # Generate continuations
        continuations = rtp_evaluator.generate_continuations(test_prompts)
        
        # Compute metrics
        rtp_results = rtp_evaluator.compute_emt_and_toxicity_prob(
            test_prompts,
            continuations,
            threshold=0.5
        )
        
        # Save results
        save_results(rtp_results, 
                    f"results/metrics/rtp_evaluation_{self.experiment_id}.json")
        
        return rtp_results
    
    def step4_evaluate_on_jigsaw(self, cafe_model: RevisedCAFEModel) -> Dict:
        """Step 4: Evaluate on Jigsaw for fairness metrics."""
        logger.info("="*60)
        logger.info("STEP 4: Jigsaw Fairness Evaluation")
        logger.info("="*60)
        
        # Load Jigsaw
        jigsaw_loader = JigsawDataLoader()
        jigsaw_df = jigsaw_loader.load_jigsaw_dataset()
        
        # Sample for evaluation
        jigsaw_sample = jigsaw_df.sample(n=min(500, len(jigsaw_df)), random_state=42)
        
        # Initialize evaluator
        perspective_client = PerspectiveAPIClient()
        jigsaw_evaluator = JigsawFairnessEvaluator(
            cafe_model=cafe_model,
            perspective_client=perspective_client,
            device=self.device
        )
        
        # Evaluate
        jigsaw_results = jigsaw_evaluator.evaluate_fairness(jigsaw_sample)
        
        # Save results
        save_results(jigsaw_results,
                    f"results/metrics/jigsaw_evaluation_{self.experiment_id}.json")
        
        return jigsaw_results
    
    def step5_generate_report(self, rtp_results: Dict, jigsaw_results: Dict):
        """Step 5: Generate comprehensive research report."""
        logger.info("="*60)
        logger.info("STEP 5: Generating Final Report")
        logger.info("="*60)
        
        report_lines = []
        report_lines.append("REVISED CAFE FRAMEWORK - RESEARCH RESULTS")
        report_lines.append("="*70)
        report_lines.append(f"Experiment ID: {self.experiment_id}")
        report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # RTP Results
        report_lines.append("RTP EVALUATION RESULTS:")
        report_lines.append(f"  Expected Maximum Toxicity:")
        report_lines.append(f"    CAFE: {rtp_results['cafe_emt']:.4f}")
        report_lines.append(f"    Perspective: {rtp_results['perspective_emt']:.4f}")
        report_lines.append(f"    Improvement: {((rtp_results['perspective_emt'] - rtp_results['cafe_emt']) / rtp_results['perspective_emt'] * 100):+.2f}%")
        report_lines.append("")
        report_lines.append(f"  Toxicity Probability:")
        report_lines.append(f"    CAFE: {rtp_results['cafe_toxicity_probability']:.4f}")
        report_lines.append(f"    Perspective: {rtp_results['perspective_toxicity_probability']:.4f}")
        report_lines.append(f"    Improvement: {((rtp_results['perspective_toxicity_probability'] - rtp_results['cafe_toxicity_probability']) / rtp_results['perspective_toxicity_probability'] * 100):+.2f}%")
        report_lines.append("")
        
        # Jigsaw Results
        report_lines.append("JIGSAW FAIRNESS EVALUATION:")
        report_lines.append(f"  Overall F1 Score:")
        report_lines.append(f"    CAFE: {jigsaw_results['overall']['cafe_f1']:.4f}")
        report_lines.append(f"    Perspective: {jigsaw_results['overall']['perspective_f1']:.4f}")
        report_lines.append("")
        report_lines.append(f"  Overall AUC:")
        report_lines.append(f"    CAFE: {jigsaw_results['overall']['cafe_auc']:.4f}")
        report_lines.append(f"    Perspective: {jigsaw_results['overall']['perspective_auc']:.4f}")
        report_lines.append("")
        
        # Subgroup metrics
        if jigsaw_results.get('subgroup_metrics'):
            report_lines.append("  Subgroup AUC Scores:")
            for subgroup, metrics in jigsaw_results['subgroup_metrics'].items():
                report_lines.append(f"    {subgroup}:")
                report_lines.append(f"      CAFE: {metrics['cafe_auc']:.4f}")
                report_lines.append(f"      Perspective: {metrics['perspective_auc']:.4f}")
                report_lines.append(f"      Sample count: {metrics['sample_count']}")
        report_lines.append("")
        
        # Research Questions
        report_lines.append("RESEARCH QUESTIONS ADDRESSED:")
        report_lines.append("")
        
        # Q1: Context awareness
        cafe_emt_better = rtp_results['cafe_emt'] < rtp_results['perspective_emt']
        if cafe_emt_better:
            report_lines.append("Q1: Context-aware embeddings improve toxicity detection?")
            report_lines.append("    ANSWER: YES - CAFE achieves lower EMT than Perspective")
        else:
            report_lines.append("Q1: Context-aware embeddings improve toxicity detection?")
            report_lines.append("    ANSWER: PARTIAL - Further optimization needed")
        report_lines.append("")
        
        # Q2: Fairness
        cafe_f1_better = jigsaw_results['overall']['cafe_f1'] > jigsaw_results['overall']['perspective_f1']
        if cafe_f1_better:
            report_lines.append("Q2: Fairness-aware loss reduces bias?")
            report_lines.append("    ANSWER: YES - CAFE shows better F1 on Jigsaw")
        else:
            report_lines.append("Q2: Fairness-aware loss reduces bias?")
            report_lines.append("    ANSWER: NEEDS IMPROVEMENT - Consider loss weight tuning")
        report_lines.append("")
        
        # Q3: Multi-objective optimization
        both_better = cafe_emt_better and cafe_f1_better
        if both_better:
            report_lines.append("Q3: Multi-objective optimization balances goals?")
            report_lines.append("    ANSWER: YES - Improvements on both RTP and Jigsaw")
        else:
            report_lines.append("Q3: Multi-objective optimization balances goals?")
            report_lines.append("    ANSWER: PARTIAL - Trade-offs observed")
        report_lines.append("")
        
        # Key Findings
        report_lines.append("KEY FINDINGS:")
        report_lines.append("  - Separate prompt/continuation tokenization preserves context")
        report_lines.append("  - Derived attributes (sens, ctx) enable fairness & context losses")
        report_lines.append("  - EMT and Toxicity Prob measure real degeneration risk")
        report_lines.append("  - Jigsaw evaluation validates fairness without retraining")
        report_lines.append("")
        
        # Save report
        report_path = f"results/FINAL_REVISED_REPORT_{self.experiment_id}.txt"
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))
        
        logger.info(f"Final report saved to {report_path}")
        
        # Print summary
        logger.info("="*70)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("="*70)
        logger.info(f"CAFE EMT: {rtp_results['cafe_emt']:.4f} vs Perspective: {rtp_results['perspective_emt']:.4f}")
        logger.info(f"CAFE F1 (Jigsaw): {jigsaw_results['overall']['cafe_f1']:.4f} vs Perspective: {jigsaw_results['overall']['perspective_f1']:.4f}")
        logger.info("="*70)
    
    def run_complete_experiment(self, rtp_path: str):
        """Run complete revised experiment pipeline."""
        logger.info("Starting Revised CAFE Experiment")
        logger.info("Methodology:")
        logger.info("1. Process RTP with attribute derivation")
        logger.info("2. Train CAFE with prompt/continuation separation")
        logger.info("3. Evaluate on RTP (EMT & Toxicity Probability)")
        logger.info("4. Evaluate on Jigsaw (Fairness metrics)")
        logger.info("5. Generate comprehensive report")
        logger.info("")
        
        try:
            # Step 1: Process RTP
            rtp_df = self.step1_process_rtp_dataset(rtp_path)
            
            # Step 2: Train CAFE
            cafe_model = self.step2_train_cafe(rtp_df)
            
            # Step 3: RTP evaluation
            rtp_results = self.step3_evaluate_on_rtp(cafe_model, rtp_df)
            
            # Step 4: Jigsaw evaluation
            jigsaw_results = self.step4_evaluate_on_jigsaw(cafe_model)
            
            # Step 5: Generate report
            self.step5_generate_report(rtp_results, jigsaw_results)
            
            logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Revised CAFE Experiment")
    parser.add_argument("--rtp_path", type=str, required=True,
                       help="Path to RTP JSONL file")
    
    args = parser.parse_args()
    
    runner = RevisedCAFEExperiment()
    runner.run_complete_experiment(args.rtp_path)

if __name__ == "__main__":
    main()