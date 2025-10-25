import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
from tqdm import tqdm
import json
from torchvision import transforms
import random
from peft import LoraConfig, get_peft_model
import time

# -----------------------------
# Centralized Configuration
# -----------------------------
class ExperimentConfig:
    """Centralized configuration for all hyperparameters."""
    
    def __init__(self):
        # Model parameters 
        self.model_name = "coca_ViT-L-14"
        self.pretrained = "mscoco_finetuned_laion2B-s13B-b90k"
        
        # Experiment parameters
        self.n_shot_values = [5, 10, 20]  # Added more values for comprehensive testing
        self.loss_functions = ["contrastive"]
        self.augmentation_options = [False]
        
        # LoRA configurations based on n_shot 
        self.lora_configs = {
            "low_shot": {"r": 4, "lora_alpha": 16, "target_modules": ["attn.out_proj"], "lora_dropout": 0.15},
            "medium_shot": {"r": 8, "lora_alpha": 32, "target_modules": ["attn.out_proj"], "lora_dropout": 0.1},
            "high_shot": {"r": 16, "lora_alpha": 64, "target_modules": ["attn.out_proj", "mlp.c_fc", "mlp.c_proj"], "lora_dropout": 0.05}
        }
        
        # Hyperparameter sets 
        self.hyperparam_sets = {
            "default": {
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "batch_size": 32,
                "num_epochs": 100,
                "label_smoothing": 0.1,
                "temperature": 0.1,
                "patience": 20,
                "contrastive_weight": 1.0,  # NEW: Weight for contrastive loss
                "ce_weight": 1.0,           # NEW: Weight for cross-entropy loss
                "use_hybrid": True          # NEW: Whether to use hybrid loss for metric learning
            },
            # "high_lr": {
            #     "learning_rate": 5e-4,
            #     "weight_decay": 0.05,
            #     "batch_size": 16,
            #     "num_epochs": 150,
            #     "label_smoothing": 0.2,
            #     "temperature": 0.2,
            #     "patience": 25,
            #     "contrastive_weight": 1.0,
            #     "ce_weight": 1.0,
            #     "use_hybrid": True
            # },
            # "low_lr": {
            #     "learning_rate": 5e-5,
            #     "weight_decay": 0.001,
            #     "batch_size": 64,
            #     "num_epochs": 80,
            #     "label_smoothing": 0.05,
            #     "temperature": 0.05,
            #     "patience": 15,
            #     "contrastive_weight": 1.0,
            #     "ce_weight": 1.0,
            #     "use_hybrid": True
            # },
            # "pure_contrastive": {  # NEW: Configuration for pure contrastive learning
            #     "learning_rate": 1e-4,
            #     "weight_decay": 0.01,
            #     "batch_size": 32,
            #     "num_epochs": 100,
            #     "label_smoothing": 0.1,
            #     "temperature": 0.1,
            #     "patience": 20,
            #     "contrastive_weight": 1.0,
            #     "ce_weight": 0.0,    # Zero weight for CE loss
            #     "use_hybrid": False  # Pure contrastive learning
            # },
            # "pure_prototypical": {  # NEW: Configuration for pure prototypical learning
            #     "learning_rate": 1e-4,
            #     "weight_decay": 0.01,
            #     "batch_size": 32,
            #     "num_epochs": 100,
            #     "label_smoothing": 0.1,
            #     "temperature": 0.1,
            #     "patience": 20,
            #     "contrastive_weight": 1.0,
            #     "ce_weight": 0.0,    # Zero weight for CE loss
            #     "use_hybrid": False  # Pure prototypical learning
            # }
        }


class ExperimentTracker:
    """Track experiment results across epochs and configurations."""
    
    def __init__(self):
        self.results = {}
        self.current_config_key = None
        
    def start_experiment(self, config_key: str, config: Dict):
        """Start tracking a new experiment."""
        self.current_config_key = config_key
        self.results[config_key] = {
            "config": config,
            "epoch_history": {
                "train_loss": [],
                "val_accuracy": [],
                "learning_rates": [],
                "train_accuracy": []
            },
            "final_metrics": {},
            "training_time": 0,
            "best_epoch": 0
        }
    
    def update_epoch(self, epoch: int, train_loss: float, val_accuracy: float, 
                   lr: float, train_accuracy: Optional[float] = None):
        """Update epoch-wise metrics."""
        if self.current_config_key not in self.results:
            return
            
        self.results[self.current_config_key]["epoch_history"]["train_loss"].append(float(train_loss))
        self.results[self.current_config_key]["epoch_history"]["val_accuracy"].append(float(val_accuracy))
        self.results[self.current_config_key]["epoch_history"]["learning_rates"].append(float(lr))
        
        if train_accuracy is not None:
            self.results[self.current_config_key]["epoch_history"]["train_accuracy"].append(float(train_accuracy))
        
        # Update best epoch
        current_best = max(self.results[self.current_config_key]["epoch_history"]["val_accuracy"])
        if val_accuracy >= current_best:
            self.results[self.current_config_key]["best_epoch"] = epoch
    
    def finalize_experiment(self, test_results: Dict, training_time: float):
        """Finalize experiment with test results."""
        if self.current_config_key not in self.results:
            return
            
        self.results[self.current_config_key]["final_metrics"] = {
            "overall_accuracy": float(test_results["overall_accuracy"]),
            "mean_per_class_accuracy": float(test_results["mean_per_class_accuracy"]),
            "total_samples": test_results["total_samples"],
            "correct_predictions": test_results["correct_predictions"]
        }
        self.results[self.current_config_key]["training_time"] = training_time


class LinearClassifier(nn.Module):
    """Simple linear classifier head."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)


class FewShotDataset(Dataset):
    """Dataset for few-shot learning."""
    def __init__(self, data_dir: Path, transform=None, augment=False):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {}
        
        class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        
        for idx, class_dir in enumerate(class_dirs):
            self.class_to_idx[class_dir.name] = idx
            image_paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.JPEG"))
            for img_path in image_paths:
                self.samples.append((img_path, idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class LoRACoCaFinetune:
    def __init__(
        self,
        config: ExperimentConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.config = config
        self.device = device
        print(f"Loading CoCa model on {device}...")
        
        # Load model and transforms
        self.model, _, self.transform = open_clip.create_model_and_transforms(
            model_name=config.model_name,
            pretrained=config.pretrained
        )
        self.model = self.model.to(device)
        
        # Store original model for reference
        self.original_model = self.model
        self.lora_model = None
        self.classifier = None
        self.class_names = None
    def setup_lora_for_fewshot(self, n_shot: int) -> LoraConfig:
        """Configure LoRA based on number of shots using ExperimentConfig."""
        
        # Determine config key based on n_shot
        if n_shot <= 2:
            config_key = "low_shot"
        elif n_shot <= 10:
            config_key = "medium_shot"
        else:
            config_key = "high_shot"
        
        # Get config from ExperimentConfig
        lora_params = self.config.lora_configs[config_key]
        
        config = LoraConfig(
            r=lora_params["r"],
            lora_alpha=lora_params["lora_alpha"],
            target_modules=lora_params["target_modules"],
            lora_dropout=lora_params["lora_dropout"],
            bias="none",
        )
        
        print(f"LoRA Config ({n_shot}-shot, {config_key}): "
              f"r={config.r}, alpha={config.lora_alpha}, "
              f"dropout={config.lora_dropout}, modules={config.target_modules}")
        
        return config

    def apply_lora(self, n_shot: int):
        """Apply LoRA to the entire CoCa model, not just the visual encoder."""
        lora_config = self.setup_lora_for_fewshot(n_shot)
        
        # Apply LoRA to the entire model, not just visual encoder
        self.lora_model = get_peft_model(self.model, lora_config)
        self.lora_model.print_trainable_parameters()

    def extract_visual_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract visual features from the LoRA-enabled model."""
        # For CoCa model, we need to extract visual features properly
        if hasattr(self.lora_model, 'encode_image'):
            # Standard CLIP/CoCa interface
            features = self.lora_model.encode_image(images)
        else:
            # Alternative: access visual directly
            features = self.lora_model.visual(images)
        
        # Handle different output formats
        if isinstance(features, tuple):
            features = features[0]  # Take the first element if it's a tuple
        elif isinstance(features, torch.Tensor):
            features = features
        else:
            raise ValueError(f"Unexpected feature type: {type(features)}")
        
        return features

    def train_with_lora(
        self,
        train_dir: Path,
        n_shot: int,
        hyperparams: Dict,
        loss_type: str,
        use_augmentation: bool,
        experiment_tracker: ExperimentTracker
    ) -> Dict:
        """Train with LoRA fine-tuning with proper gradient flow."""
        start_time = time.time()
        
        # Apply LoRA to entire model
        self.apply_lora(n_shot)
        
        # Create dataset with n-shot sampling
        train_samples = []
        class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
        self.class_names = [d.name for d in class_dirs]
        num_classes = len(self.class_names)
        class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        print(f"Preparing {n_shot}-shot training data...")
        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = class_to_idx[class_name]
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.JPEG"))
            image_paths = sorted(image_files)[:n_shot]
            
            for img_path in image_paths:
                train_samples.append((img_path, class_idx))
        
        print(f"Total training samples: {len(train_samples)}")
        
        # Create dataset
        class TempDataset(Dataset):
            def __init__(self, samples, transform, augment):
                self.samples = samples
                self.transform = transform
                self.augment = augment
                
                if self.augment:
                    self.augment_transform = transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                        transforms.RandomGrayscale(p=0.1),
                        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    ])
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                img_path, label = self.samples[idx]
                image = Image.open(img_path).convert("RGB")
                
                if self.augment:
                    image = self.augment_transform(image)
                
                if self.transform:
                    image = self.transform(image)
                
                return image, label
        
        train_dataset = TempDataset(train_samples, self.transform, use_augmentation)
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(hyperparams["batch_size"], len(train_dataset)),
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Get feature dimension and setup classifier
        with torch.no_grad():
            sample_img = train_dataset[0][0].unsqueeze(0).to(self.device)
            sample_feat = self.extract_visual_features(sample_img)
            feature_dim = sample_feat.shape[-1]
        
        self.classifier = LinearClassifier(feature_dim, num_classes).to(self.device)
        
        # Setup optimizer
        lora_params = [p for p in self.lora_model.parameters() if p.requires_grad]
        classifier_params = [p for p in self.classifier.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            [
                {'params': lora_params, 'lr': hyperparams["learning_rate"]},
                {'params': classifier_params, 'lr': hyperparams["learning_rate"] * 10}
            ],
            weight_decay=hyperparams["weight_decay"]
        )
        
        # Test gradient flow before training
        print("\n=== Testing Gradient Flow Before Training ===")
        self.lora_model.train()
        self.classifier.train()
        
        test_images, test_labels = next(iter(train_loader))
        test_images, test_labels = test_images.to(self.device), test_labels.to(self.device)
        
        optimizer.zero_grad()
        
        # Forward pass
        test_features = self.extract_visual_features(test_images)
        test_logits = self.classifier(test_features)
        test_loss = F.cross_entropy(test_logits, test_labels)
        
        # Backward pass
        test_loss.backward()
        
        # Check gradients
        lora_grad_norm = 0
        lora_grad_count = 0
        for name, param in self.lora_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                lora_grad_norm += grad_norm
                lora_grad_count += 1
        
        classifier_grad_norm = 0
        classifier_grad_count = 0
        for name, param in self.classifier.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                classifier_grad_norm += grad_norm
                classifier_grad_count += 1
        
        print(f"LoRA gradients: {lora_grad_count} parameters, total norm: {lora_grad_norm:.6f}")
        print(f"Classifier gradients: {classifier_grad_count} parameters, total norm: {classifier_grad_norm:.6f}")
        print(f"Features require grad: {test_features.requires_grad}")
        print(f"Features grad_fn: {test_features.grad_fn}")
        
        if lora_grad_count == 0:
            print("❌ CRITICAL: No gradients flowing to LoRA parameters!")
            # Try alternative approach
            return self._train_with_alternative_approach(
                train_samples, n_shot, hyperparams, loss_type, use_augmentation, experiment_tracker
            )
        else:
            print("✅ Gradients flowing to LoRA parameters!")
        
        optimizer.zero_grad()  # Reset after test
        
        # Continue with normal training...
        criterion = self._get_loss_function(loss_type, hyperparams, n_shot)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyperparams["num_epochs"])
        
        best_val_acc = 0.0
        patience_counter = 0
        best_lora_state = None
        best_classifier_state = None
        
        print(f"\nTraining with LoRA and {loss_type} loss...")
        
        for epoch in range(hyperparams["num_epochs"]):
            self.lora_model.train()
            self.classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                features = self.extract_visual_features(images)
                loss = criterion(features, labels, self.classifier)
                
                # Calculate accuracy
                with torch.no_grad():
                    logits = self.classifier(features)
                    preds = logits.argmax(dim=1)
                    train_correct += (preds == labels).sum().item()
                    train_total += len(labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = (train_correct / train_total) * 100
            
            # Validation
            val_accuracy = self._validate_on_train(train_samples, hyperparams["batch_size"])
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            experiment_tracker.update_epoch(epoch, avg_train_loss, val_accuracy, current_lr, train_accuracy)
            
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
                best_lora_state = self.lora_model.state_dict()
                best_classifier_state = self.classifier.state_dict()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{hyperparams['num_epochs']}] "
                      f"Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                      f"Val Acc: {val_accuracy:.2f}% | LR: {current_lr:.6f}")
            
            if patience_counter >= hyperparams["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_lora_state is not None:
            self.lora_model.load_state_dict(best_lora_state)
        if best_classifier_state is not None:
            self.classifier.load_state_dict(best_classifier_state)
        
        training_time = time.time() - start_time
        print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Training time: {training_time:.2f} seconds")
        
        return {"train_loss": [avg_train_loss], "val_accuracy": [val_accuracy]}

    def _train_with_alternative_approach(self, train_samples, n_shot, hyperparams, loss_type, use_augmentation, experiment_tracker):
        """Alternative training approach when LoRA gradients don't flow."""
        print("🔄 Using alternative training approach...")
        
        # Fallback: Fine-tune the entire visual encoder without LoRA
        self.lora_model = self.model.visual  # Use visual encoder directly
        for param in self.lora_model.parameters():
            param.requires_grad = True
        
        # Continue with training as before...
        # [Rest of the training code remains the same]
        
        # This is a simplified version - you'd need to adapt the rest of the training code
        return {"train_loss": [0.0], "val_accuracy": [0.0]}  # Placeholder
    def prototypical_loss(self, features: torch.Tensor, labels: torch.Tensor, n_support: int = 2) -> torch.Tensor:
        """
        Prototypical loss for few-shot learning - same as working code.
        """
        unique_labels = torch.unique(labels)
        n_classes = len(unique_labels)
        
        # Create label to index mapping
        label_to_idx = {label.item(): idx for idx, label in enumerate(unique_labels)}
        
        # Create prototypes for each class
        prototypes = []
        query_features_list = []
        query_labels_list = []
        
        for label in unique_labels:
            mask = labels == label
            class_features = features[mask]
            
            if len(class_features) <= n_support:
                # Not enough samples for this class, use all as support
                prototype = class_features.mean(dim=0)
                prototypes.append(prototype)
            else:
                # Split into support and query
                support_features = class_features[:n_support]
                query_features = class_features[n_support:]
                
                prototype = support_features.mean(dim=0)
                prototypes.append(prototype)
                
                query_features_list.append(query_features)
                query_labels_list.extend([label_to_idx[label.item()]] * len(query_features))
        
        if len(query_features_list) == 0:
            # No query samples, fallback to cross-entropy with prototypes
            distances = torch.cdist(features, torch.stack(prototypes), p=2)
            logits = -distances
            mapped_labels = torch.tensor([label_to_idx[l.item()] for l in labels], device=features.device)
            return F.cross_entropy(logits, mapped_labels)
        
        prototypes = torch.stack(prototypes)
        query_features = torch.cat(query_features_list)
        query_labels = torch.tensor(query_labels_list, device=features.device)
        
        # Calculate distances to prototypes
        distances = torch.cdist(query_features, prototypes, p=2)
        logits = -distances
        
        return F.cross_entropy(logits, query_labels)
    
    def contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """
        Supervised contrastive loss (SupCon) - same as working code.
        """
        batch_size = features.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device)
        
        # Normalize features
        features = F.normalize(features, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature
        
        # Create mask for positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        # Remove diagonal
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        
        loss = -mean_log_prob_pos.mean()
        return loss


    def _get_loss_function(self, loss_type, hyperparams, n_shot):
        """Get loss function without manual requires_grad manipulation."""
        if loss_type == "cross_entropy":
            ce_criterion = nn.CrossEntropyLoss(label_smoothing=hyperparams["label_smoothing"])
            
            def cross_entropy_loss(features, labels, classifier):
                logits = classifier(features)
                return ce_criterion(logits, labels)
            return cross_entropy_loss
            
        elif loss_type == "prototypical":
            def prototypical_loss_wrapper(features, labels, classifier):
                metric_loss = self.prototypical_loss(features, labels, n_support=max(1, n_shot//2))
                
                if hyperparams.get("use_hybrid", True) and hyperparams["ce_weight"] > 0:
                    logits = classifier(features)
                    ce_loss = F.cross_entropy(logits, labels, label_smoothing=hyperparams["label_smoothing"])
                    return (hyperparams["contrastive_weight"] * metric_loss + 
                            hyperparams["ce_weight"] * ce_loss)
                return metric_loss
            return prototypical_loss_wrapper
            
        elif loss_type == "contrastive":
            def contrastive_loss_wrapper(features, labels, classifier):
                metric_loss = self.contrastive_loss(features, labels, hyperparams["temperature"])
                
                if hyperparams.get("use_hybrid", True) and hyperparams["ce_weight"] > 0:
                    logits = classifier(features)
                    ce_loss = F.cross_entropy(logits, labels, label_smoothing=hyperparams["label_smoothing"])
                    return (hyperparams["contrastive_weight"] * metric_loss + 
                            hyperparams["ce_weight"] * ce_loss)
                return metric_loss
            return contrastive_loss_wrapper
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def _validate_on_train(self, train_samples: List, batch_size: int) -> float:
        """Validate on training data."""
        class TempValDataset(Dataset):
            def __init__(self, samples, transform):
                self.samples = samples
                self.transform = transform
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                img_path, label = self.samples[idx]
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, label
        
        val_dataset = TempValDataset(train_samples, self.transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        correct = 0
        total = 0
        
        self.lora_model.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                features = self.extract_visual_features(images)
                logits = self.classifier(features)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        
        return (correct / total) * 100
    def evaluate(self, test_dir: Path, batch_size: int = 32) -> Dict[str, float]:
        """Evaluate the fine-tuned model on test set - same as working code."""
        if self.lora_model is None or self.classifier is None:
            raise ValueError("Model not trained. Call train_with_lora first.")
        
        self.lora_model.eval()
        self.classifier.eval()
        
        # Create test dataset
        test_dataset = FewShotDataset(test_dir, transform=self.transform, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        all_preds = []
        all_labels = []
        
        print("\nEvaluating on test set...")
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(self.device)
                
                with torch.cuda.amp.autocast():
                    features = self.lora_model(images)[0]  # This matches your working code
                    logits = self.classifier(features)
                    preds = logits.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        overall_accuracy = (all_preds == all_labels).mean() * 100
        
        # Per-class accuracy
        per_class_accuracy = {}
        for class_idx, class_name in enumerate(self.class_names):
            class_mask = all_labels == class_idx
            if class_mask.sum() > 0:
                class_acc = (all_preds[class_mask] == class_idx).mean() * 100
                per_class_accuracy[class_name] = class_acc
        
        mean_per_class_accuracy = np.mean(list(per_class_accuracy.values()))
        
        return {
            "overall_accuracy": float(overall_accuracy),
            "mean_per_class_accuracy": float(mean_per_class_accuracy),
            "total_samples": len(all_labels),
            "correct_predictions": int((all_preds == all_labels).sum()),
            "per_class_accuracy": per_class_accuracy
        }

    def run_comprehensive_experiments(
    self, 
    dataset_dir: Path, 
    output_file: str = "comprehensive_lora_results.json"
) -> Dict:
        """Run experiments across all configurations."""
        config = self.config
        tracker = ExperimentTracker()
        
        experiment_id = 0
        total_experiments = (
            len(config.n_shot_values) * 
            len(config.loss_functions) * 
            len(config.augmentation_options) * 
            len(config.hyperparam_sets)
        )
        
        print(f"Starting comprehensive experiments: {total_experiments} total configurations")
        
        for n_shot in config.n_shot_values:
            for loss_type in config.loss_functions:
                for aug in config.augmentation_options:
                    for hp_set_name, hyperparams in config.hyperparam_sets.items():
                        
                        config_key = f"exp_{experiment_id:03d}_{n_shot}shot_{loss_type}_aug{aug}_{hp_set_name}"
                        
                        # Store configuration
                        experiment_config = {
                            "n_shot": n_shot,
                            "loss_type": loss_type,
                            "use_augmentation": aug,
                            "hyperparams": hyperparams,
                            "model": config.model_name,
                            "pretrained": config.pretrained
                        }
                        
                        tracker.start_experiment(config_key, experiment_config)
                        
                        print(f"\n{'='*80}")
                        print(f"Running experiment {experiment_id + 1}/{total_experiments}: {config_key}")
                        print(f"{'='*80}")
                        
                        try:
                            start_time = time.time()
                            
                            # Training
                            train_dir = dataset_dir / "train"
                            if not train_dir.exists():
                                raise FileNotFoundError(f"Training directory not found: {train_dir}")
                            
                            training_history = self.train_with_lora(
                                train_dir=train_dir,
                                n_shot=n_shot,
                                hyperparams=hyperparams,
                                loss_type=loss_type,
                                use_augmentation=aug,
                                experiment_tracker=tracker
                            )
                            
                            # Evaluation
                            test_dir = dataset_dir / "test"
                            if not test_dir.exists():
                                raise FileNotFoundError(f"Test directory not found: {test_dir}")
                            
                            test_results = self.evaluate(test_dir)
                            
                            # Finalize experiment
                            training_time = time.time() - start_time
                            tracker.finalize_experiment(test_results, training_time)
                            
                            print(f"✓ Completed: Overall Acc: {test_results['overall_accuracy']:.2f}% | "
                                  f"Time: {training_time:.2f}s")
                            
                        except Exception as e:
                            print(f"✗ Experiment {config_key} failed: {str(e)}")
                            tracker.results[config_key]["error"] = str(e)
                        
                        experiment_id += 1
        
        # Save all results
        self._save_results(tracker.results, output_file)
        
        # Print summary
        self._print_experiment_summary(tracker.results)
        
        return tracker.results

    def _save_results(self, results: Dict, output_file: str):
        """Save results to JSON file."""
        # Convert to serializable format
        serializable_results = {}
        for key, data in results.items():
            serializable_results[key] = {
                "config": data["config"],
                "epoch_history": data["epoch_history"],
                "final_metrics": data["final_metrics"],
                "training_time": data.get("training_time", 0),
                "best_epoch": data.get("best_epoch", 0)
            }
            if "error" in data:
                serializable_results[key]["error"] = data["error"]
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nAll results saved to {output_file}")

    def _print_experiment_summary(self, results: Dict):
        """Print a summary of all experiments."""
        print(f"\n{'='*100}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'='*100}")
        print(f"{'Config':<60} {'Epochs':<8} {'Best Val Acc':<12} {'Test Acc':<10} {'Time (s)':<10}")
        print(f"{'-'*100}")
        
        successful_experiments = 0
        for config_key, data in results.items():
            if "final_metrics" in data and data["final_metrics"]:
                epochs_trained = len(data["epoch_history"]["val_accuracy"])
                best_val_acc = max(data["epoch_history"]["val_accuracy"]) if data["epoch_history"]["val_accuracy"] else 0
                test_acc = data["final_metrics"]["overall_accuracy"]
                training_time = data.get("training_time", 0)
                
                print(f"{config_key:<60} {epochs_trained:<8} {best_val_acc:<12.2f} {test_acc:<10.2f} {training_time:<10.1f}")
                successful_experiments += 1
            else:
                print(f"{config_key:<60} {'FAILED':<8} {'-':<12} {'-':<10} {'-':<10}")
        
        print(f"{'='*100}")
        print(f"Successful experiments: {successful_experiments}/{len(results)}")


def main():
    """Main function to run the comprehensive experiments."""
    # Configuration
    config = ExperimentConfig()
    
    # Initialize the fine-tuner
    finetuner = LoRACoCaFinetune(config)
    
    # Set your dataset path here
    DATASET_DIR = Path("/kaggle/working/mini_imagenet_fewshot_renamed")  # Update this path
    
    # Run all experiments
    results = finetuner.run_comprehensive_experiments(
        dataset_dir=DATASET_DIR,
        output_file="comprehensive_lora_results.json"
    )
    
    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()