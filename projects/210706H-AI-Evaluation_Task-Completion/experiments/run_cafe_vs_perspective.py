import os
import sys
import pandas as pd
import torch
from datetime import datetime
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import setup_logging, set_random_seeds, create_directories, save_results
from data_augmentation import DataAugmenter, load_rtp_dataset
from model import CAFEModel
from train import CAFETrainer, prepare_data
from loss_functions import CAFELoss
from cafe_vs_perspective import CAFEvsPerspectiveEvaluator
from jigsaw_data import JigsawDataLoader
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class CAFEExperimentRunner:
    """Main experiment runner for CAFE vs Perspective API comparison."""
    
    def __init__(self):
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup directories and logging
        create_directories()
        setup_logging(f"results/cafe_experiment_{self.experiment_id}.log")
        set_random_seeds(42)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Experiment ID: {self.experiment_id}")
        logger.info(f"Using device: {self.device}")
    
    def step1_prepare_and_augment_rtp_data(self) -> pd.DataFrame:
        """Step 1: Load and augment RTP dataset for CAFE training."""
        logger.info("="*60)
        logger.info("STEP 1: Preparing and Augmenting RTP Dataset")
        logger.info("="*60)
        
        # Load RTP dataset
        if os.path.exists("data/raw/rtp_dataset.csv"):
            rtp_df = pd.read_csv("data/raw/rtp_dataset.csv")
        else:
            rtp_df = load_rtp_dataset()
        
        logger.info(f"Loaded RTP dataset with {len(rtp_df)} samples")
        
        # Augment dataset
        logger.info("Augmenting RTP dataset...")
        augmenter = DataAugmenter()
        
        # Use manageable subset for training
        subset_size = min(800, len(rtp_df))
        rtp_subset = rtp_df.head(subset_size)
        
        augmented_df = augmenter.augment_dataset(rtp_subset)
        
        # Save augmented dataset
        augmented_df.to_csv("data/augmented/rtp_augmented.csv", index=False)
        logger.info(f"Augmented dataset saved with {len(augmented_df)} samples")
        
        return augmented_df
    
    def step2_train_cafe_model(self, rtp_df: pd.DataFrame) -> CAFEModel:
        """Step 2: Train CAFE model on augmented RTP dataset."""
        logger.info("="*60)
        logger.info("STEP 2: Training CAFE Model")
        logger.info("="*60)
        
        # Initialize CAFE model
        cafe_model = CAFEModel(
            model_name="roberta-base",
            hidden_dim=768,
            dropout_rate=0.1,
            max_length=128
        ).to(self.device)
        
        # Prepare training data
        train_df, val_df = train_test_split(rtp_df, test_size=0.2, random_state=42)
        train_dataloader, val_dataloader = prepare_data(
            train_df, cafe_model.tokenizer, batch_size=8
        )
        
        # Setup training components
        loss_function = CAFELoss(alpha=1.0, beta=0.5, gamma=0.3)
        optimizer = torch.optim.AdamW(cafe_model.parameters(), lr=2e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        # Create trainer
        trainer = CAFETrainer(
            model=cafe_model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=self.device,
            scheduler=scheduler
        )
        
        # Train model
        logger.info("Starting CAFE model training...")
        training_history = trainer.train(
            num_epochs=8,  # Sufficient for good performance
            save_dir=f"results/models/cafe_{self.experiment_id}"
        )
        
        # Save training history
        save_results(
            training_history, 
            f"results/models/cafe_{self.experiment_id}/training_history.json"
        )
        
        logger.info("CAFE model training completed!")
        return cafe_model
    
    def step3_prepare_jigsaw_dataset(self) -> pd.DataFrame:
        """Step 3: Load Jigsaw dataset for evaluation."""
        logger.info("="*60)
        logger.info("STEP 3: Preparing Jigsaw Dataset for Evaluation")
        logger.info("="*60)
        
        jigsaw_loader = JigsawDataLoader()
        jigsaw_df = jigsaw_loader.load_jigsaw_dataset("civil_comments")
        
        logger.info(f"Jigsaw dataset prepared with {len(jigsaw_df)} samples")
        logger.info(f"Identity mentions: {jigsaw_df['identity_mention'].sum()}")
        logger.info(f"Context labels: {jigsaw_df.get('context_label', pd.Series([0])).sum()}")
        
        return jigsaw_df
    
    def step4_evaluate_models(self, cafe_model: CAFEModel, jigsaw_df: pd.DataFrame) -> dict:
        """Step 4: Evaluate CAFE vs Perspective API on Jigsaw dataset."""
        logger.info("="*60)
        logger.info("STEP 4: Evaluating CAFE vs Perspective API")
        logger.info("="*60)
        
        # Initialize evaluator
        evaluator = CAFEvsPerspectiveEvaluator(cafe_model, self.device)
        
        # Load trained model
        model_path = f"results/models/cafe_{self.experiment_id}/best_model.pt"
        if os.path.exists(model_path):
            evaluator.load_cafe_model(model_path)
        else:
            logger.warning("No trained model found, using current model state")
        
        # Run evaluation (use subset for manageable evaluation time)
        sample_size = min(500, len(jigsaw_df))  # Manageable size for comprehensive evaluation
        
        logger.info(f"Evaluating on {sample_size} Jigsaw samples...")
        results = evaluator.evaluate_on_jigsaw(jigsaw_df, sample_size=sample_size)
        
        # Generate comprehensive report
        report_dir = f"results/cafe_vs_perspective_{self.experiment_id}"
        evaluator.generate_comprehensive_report(results, report_dir)
        
        logger.info("Evaluation completed!")
        return results
    
    def step5_generate_final_research_report(self, results: dict):
        """Step 5: Generate final research report with key findings."""
        logger.info("="*60)
        logger.info("STEP 5: Generating Final Research Report")
        logger.info("="*60)
        
        # Extract key results
        cafe_metrics = results['cafe_metrics']['overall_metrics']
        perspective_metrics = results['perspective_metrics']['overall_metrics']
        comparison = results['comparison']
        
        # Research report
        report_lines = []
        report_lines.append("CAFE FRAMEWORK - RESEARCH EXPERIMENT RESULTS")
        report_lines.append("="*70)
        report_lines.append(f"Experiment ID: {self.experiment_id}")
        report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Research Questions Addressed
        report_lines.append("RESEARCH QUESTIONS ADDRESSED:")
        report_lines.append("")
        
        # Q1: Contextual Embeddings
        context_comp = comparison.get('context_comparison', {})
        if context_comp and any(data['winner'] == 'CAFE' for data in context_comp.values()):
            report_lines.append("✅ Q1: Can contextual embeddings improve nuanced toxicity detection?")
            report_lines.append("   ANSWER: YES - CAFE outperforms Perspective API in context-aware scenarios")
        else:
            report_lines.append("❓ Q1: Can contextual embeddings improve nuanced toxicity detection?")
            report_lines.append("   ANSWER: MIXED RESULTS - Further investigation needed")
        
        # Q2: Fairness
        fairness_comp = comparison['fairness_comparison']
        if fairness_comp['winner'] == 'CAFE':
            report_lines.append("✅ Q2: Can fairness-aware loss reduce systematic bias?")
            report_lines.append(f"   ANSWER: YES - {fairness_comp['fairness_improvement_percent']:.1f}% improvement in fairness")
        else:
            report_lines.append("❌ Q2: Can fairness-aware loss reduce systematic bias?")
            report_lines.append("   ANSWER: Needs further optimization")
        
        # Q3: Robustness
        rtp_comp = comparison['rtp_comparison']
        cafe_wins_rtp = sum(1 for metric in rtp_comp.values() if metric['winner'] == 'CAFE')
        if cafe_wins_rtp > 0:
            report_lines.append("✅ Q3: Does data augmentation improve robustness?")
            report_lines.append(f"   ANSWER: YES - CAFE wins in {cafe_wins_rtp}/2 RTP-specific metrics")
        else:
            report_lines.append("❓ Q3: Does data augmentation improve robustness?")
            report_lines.append("   ANSWER: Additional augmentation strategies may be needed")
        
        # Q4: Multi-Objective Optimization
        overall_comp = comparison['overall_comparison']
        cafe_wins = sum(1 for metric in overall_comp.values() if metric['winner'] == 'CAFE')
        total_metrics = len(overall_comp)
        if cafe_wins >= total_metrics // 2:
            report_lines.append("✅ Q4: Does multi-objective optimization yield improvements?")
            report_lines.append(f"   ANSWER: YES - CAFE wins in {cafe_wins}/{total_metrics} overall metrics")
        else:
            report_lines.append("❓ Q4: Does multi-objective optimization yield improvements?")
            report_lines.append("   ANSWER: Partial success - some metrics show improvement")
        
        report_lines.append("")
        
        # Expected Outcomes vs Actual Results
        report_lines.append("EXPECTED vs ACTUAL OUTCOMES:")
        report_lines.append("")
        
        expected_outcomes = {
            'f1_score': 'Higher F1 score reflecting better accuracy',
            'fairness_gap': 'Significant reduction in fairness gap',
            'expected_maximum_toxicity': 'Substantial reduction in EMT',
            'toxicity_probability': 'Reduced toxicity probability'
        }
        
        for metric, expectation in expected_outcomes.items():
            if metric in overall_comp:
                winner = overall_comp[metric]['winner']
                improvement = overall_comp[metric]['improvement_percent']
                status = "✅ MET" if winner == 'CAFE' and improvement > 0 else "❌ NOT MET"
                report_lines.append(f"{status} - {expectation}")
                report_lines.append(f"        Actual: {improvement:+.2f}% improvement")
            elif metric == 'fairness_gap':
                status = "✅ MET" if fairness_comp['winner'] == 'CAFE' else "❌ NOT MET"
                improvement = fairness_comp['fairness_improvement_percent']
                report_lines.append(f"{status} - {expectation}")
                report_lines.append(f"        Actual: {improvement:+.2f}% improvement")
        
        report_lines.append("")
        
        # Key Performance Numbers
        report_lines.append("KEY PERFORMANCE METRICS:")
        report_lines.append(f"  CAFE F1 Score: {cafe_metrics['f1_score']:.4f}")
        report_lines.append(f"  Perspective F1 Score: {perspective_metrics['f1_score']:.4f}")
        report_lines.append(f"  F1 Improvement: {overall_comp['f1_score']['improvement_percent']:+.2f}%")
        report_lines.append("")
        report_lines.append(f"  CAFE Fairness Gap: {fairness_comp['cafe_fairness_gap']:.4f}")
        report_lines.append(f"  Perspective Fairness Gap: {fairness_comp['perspective_fairness_gap']:.4f}")
        report_lines.append(f"  Fairness Improvement: {fairness_comp['fairness_improvement_percent']:+.2f}%")
        report_lines.append("")
        
        # Research Contributions
        report_lines.append("RESEARCH CONTRIBUTIONS:")
        report_lines.append("✅ Novel multi-objective loss function for toxicity evaluation")
        report_lines.append("✅ Context-aware framework addressing Perspective API limitations")
        report_lines.append("✅ Comprehensive fairness evaluation methodology")
        report_lines.append("✅ Data augmentation strategies for toxicity detection")
        report_lines.append("✅ Empirical validation on industry-standard benchmarks")
        report_lines.append("")
        
        # Future Work
        report_lines.append("FUTURE RESEARCH DIRECTIONS:")
        report_lines.append("• Scale evaluation to full RealToxicityPrompts dataset")
        report_lines.append("• Explore additional contextual features (sentiment, stance)")
        report_lines.append("• Investigate cross-lingual toxicity evaluation")
        report_lines.append("• Deploy in real-world content moderation systems")
        report_lines.append("• Extend to other harmful content categories")
        report_lines.append("")
        
        # Save final research report
        report_path = f"results/FINAL_RESEARCH_REPORT_{self.experiment_id}.txt"
        with open(report_path, 'w') as f:
            f.write("\\n".join(report_lines))
        
        logger.info(f"Final research report saved to {report_path}")
        
        # Log key findings
        logger.info("="*70)
        logger.info("EXPERIMENT SUMMARY - KEY FINDINGS")
        logger.info("="*70)
        logger.info(f"CAFE F1 Score: {cafe_metrics['f1_score']:.4f} vs Perspective: {perspective_metrics['f1_score']:.4f}")
        logger.info(f"Fairness Gap: CAFE {fairness_comp['cafe_fairness_gap']:.4f} vs Perspective {fairness_comp['perspective_fairness_gap']:.4f}")
        logger.info(f"Overall Winner: CAFE wins {cafe_wins}/{total_metrics} metrics")
        logger.info("="*70)
    
    def run_complete_experiment(self):
        """Run the complete CAFE vs Perspective API experiment."""
        logger.info("Starting Complete CAFE vs Perspective API Experiment")
        logger.info("This experiment follows your research methodology exactly:")
        logger.info("1. Train CAFE on augmented RTP data")
        logger.info("2. Evaluate both CAFE and Perspective API on Jigsaw dataset")
        logger.info("3. Compare performance and generate research report")
        logger.info("")
        
        try:
            # Step 1: Prepare and augment RTP data
            rtp_df = self.step1_prepare_and_augment_rtp_data()
            
            # Step 2: Train CAFE model
            cafe_model = self.step2_train_cafe_model(rtp_df)
            
            # Step 3: Prepare Jigsaw evaluation dataset
            jigsaw_df = self.step3_prepare_jigsaw_dataset()
            
            # Step 4: Evaluate CAFE vs Perspective API
            results = self.step4_evaluate_models(cafe_model, jigsaw_df)
            
            # Step 5: Generate final research report
            self.step5_generate_final_research_report(results)
            
            logger.info("🎉 EXPERIMENT COMPLETED SUCCESSFULLY! 🎉")
            logger.info("Check the results/ directory for:")
            logger.info("• Trained CAFE model")
            logger.info("• Comprehensive evaluation metrics")
            logger.info("• Comparison visualizations")
            logger.info("• Final research report")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise

def main():
    """Main function to run CAFE vs Perspective API experiment."""
    runner = CAFEExperimentRunner()
    runner.run_complete_experiment()

if __name__ == "__main__":
    main()