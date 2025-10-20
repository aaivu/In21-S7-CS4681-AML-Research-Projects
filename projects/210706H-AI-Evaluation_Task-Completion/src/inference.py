"""
Quick Inference Script
Load trained ensemble model and make predictions on new text
"""

import torch
import numpy as np
from transformers import DebertaV2Tokenizer
from DeBERTaForToxicity import DeBERTaForToxicity
from preprocessing import preprocess_text
import warnings
warnings.filterwarnings('ignore')


class ToxicityPredictor:
    """
    Easy-to-use wrapper for making toxicity predictions.
    Loads the ensemble of 5 models and provides a simple predict interface.
    """
    
    def __init__(self, model_dir='models', model_name='microsoft/deberta-v3-large', device=None):
        """
        Initialize the predictor.
        
        Args:
            model_dir (str): Directory containing model checkpoints
            model_name (str): HuggingFace model identifier
            device: PyTorch device (None = auto-detect)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        
        # Load ensemble of models
        print("Loading ensemble models...")
        self.models = []
        for i in range(5):
            model_path = f"{model_dir}/deberta_fold{i}_best.pt"
            try:
                model = DeBERTaForToxicity(model_name=model_name)
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.device)
                model.eval()
                self.models.append(model)
                print(f"  ✓ Loaded fold {i}")
            except FileNotFoundError:
                print(f"  ✗ Model {model_path} not found")
                raise
        
        print(f"✓ Loaded {len(self.models)} models successfully\n")
    
    def preprocess(self, text):
        """
        Preprocess text using the same pipeline as training.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Preprocessed text
        """
        return preprocess_text(text)
    
    def predict_single(self, text, return_all_tasks=False, use_preprocessing=True):
        """
        Predict toxicity for a single text.
        
        Args:
            text (str): Input text
            return_all_tasks (bool): If True, return predictions for all 7 tasks
            use_preprocessing (bool): If True, apply preprocessing pipeline
            
        Returns:
            float or dict: Toxicity score (0-1), or dict with all task scores
        """
        # Preprocess if requested
        if use_preprocessing:
            text = self.preprocess(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get predictions from all models
        all_predictions = {
            'toxicity': [],
            'severe_toxicity': [],
            'obscene': [],
            'identity_attack': [],
            'insult': [],
            'threat': [],
            'sexual_explicit': []
        }
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(input_ids, attention_mask)
                
                for task in all_predictions.keys():
                    prob = torch.sigmoid(outputs[task]).item()
                    all_predictions[task].append(prob)
        
        # Ensemble with power-weighted sum
        ensemble_predictions = {}
        power = 3.5
        
        for task, preds in all_predictions.items():
            preds_array = np.array(preds)
            powered_preds = np.power(preds_array, power)
            ensemble_pred = np.mean(powered_preds)
            ensemble_pred = np.power(ensemble_pred, 1/power)
            ensemble_predictions[task] = float(ensemble_pred)
        
        if return_all_tasks:
            return ensemble_predictions
        else:
            return ensemble_predictions['toxicity']
    
    def predict_batch(self, texts, use_preprocessing=True):
        """
        Predict toxicity for multiple texts.
        
        Args:
            texts (list): List of input texts
            use_preprocessing (bool): If True, apply preprocessing pipeline
            
        Returns:
            list: List of toxicity scores (0-1)
        """
        return [self.predict_single(text, use_preprocessing=use_preprocessing) for text in texts]
    
    def predict_with_explanation(self, text, use_preprocessing=True):
        """
        Predict toxicity and provide a human-readable explanation.
        
        Args:
            text (str): Input text
            use_preprocessing (bool): If True, apply preprocessing pipeline
            
        Returns:
            dict: Prediction with interpretation
        """
        all_scores = self.predict_single(text, return_all_tasks=True, use_preprocessing=use_preprocessing)
        toxicity_score = all_scores['toxicity']
        
        # Interpret score
        if toxicity_score < 0.3:
            interpretation = "Non-toxic"
            severity = "Low"
        elif toxicity_score < 0.6:
            interpretation = "Potentially toxic"
            severity = "Medium"
        else:
            interpretation = "Toxic"
            severity = "High"
        
        # Find top contributing factors
        aux_scores = {k: v for k, v in all_scores.items() if k != 'toxicity'}
        top_factors = sorted(aux_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'text': text,
            'toxicity_score': toxicity_score,
            'interpretation': interpretation,
            'severity': severity,
            'top_factors': [
                {'type': factor[0], 'score': factor[1]}
                for factor in top_factors
            ],
            'all_scores': all_scores
        }


def interactive_demo():
    """Run an interactive demo of the toxicity predictor."""
    print("="*80)
    print(" CAFE Toxicity Predictor - Interactive Demo")
    print("="*80)
    
    # Load predictor
    try:
        predictor = ToxicityPredictor()
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please ensure you have trained the models first (run train.py)")
        return
    
    print("\nEnter text to analyze (or 'quit' to exit)\n")
    
    while True:
        # Get input
        text = input("Text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            continue
        
        # Get prediction with explanation
        result = predictor.predict_with_explanation(text)
        
        # Display results
        print("\n" + "-"*80)
        print(f"Toxicity Score: {result['toxicity_score']:.4f}")
        print(f"Interpretation: {result['interpretation']} (Severity: {result['severity']})")
        
        print("\nTop Contributing Factors:")
        for factor in result['top_factors']:
            print(f"  - {factor['type']}: {factor['score']:.4f}")
        
        print("\nAll Scores:")
        for task, score in result['all_scores'].items():
            print(f"  {task}: {score:.4f}")
        
        print("-"*80 + "\n")


def batch_demo():
    """Demo with predefined examples."""
    print("="*80)
    print(" CAFE Toxicity Predictor - Batch Demo")
    print("="*80)
    
    # Load predictor
    try:
        predictor = ToxicityPredictor()
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Example texts
    examples = [
        "I love this beautiful day!",
        "You are an idiot and should be ashamed.",
        "I disagree with your opinion, but I respect your right to express it.",
        "All people from that country are terrible.",
        "This movie was boring and poorly made.",
        "You're so stupid, just kill yourself.",
        "I think we need to have a thoughtful discussion about this issue.",
        "Women belong in the kitchen."
    ]
    
    print("\nAnalyzing example texts...\n")
    
    for i, text in enumerate(examples, 1):
        result = predictor.predict_with_explanation(text)
        
        print(f"{i}. Text: {text}")
        print(f"   Score: {result['toxicity_score']:.4f} - {result['interpretation']}")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        batch_demo()
    else:
        # Check if we should run batch or interactive
        try:
            # Try batch first to show examples
            batch_demo()
            
            # Then ask if user wants interactive mode
            response = input("\nRun interactive mode? (y/n): ")
            if response.lower() == 'y':
                interactive_demo()
        except KeyboardInterrupt:
            print("\n\nExiting...")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()