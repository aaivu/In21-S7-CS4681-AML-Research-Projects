#!/usr/bin/env python3
"""
Few-Shot Learning Evaluation for Pre-trained TNC Encoder
========================================================

This script evaluates the few-shot learning capabilities of a pre-trained TNC encoder
using various few-shot learning approaches including Prototypical Networks.

Usage:
    python -m evaluations.few_shot_test --data simulation --shots 1,3,5,10 --trials 50
    python -m evaluations.few_shot_test --data simulation --method prototypical --shots 5
    python -m evaluations.few_shot_test --data simulation --method nearest_neighbor --shots 1,3,5
"""

import os
import torch
import numpy as np
import pickle
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Simple progress bar replacement for tqdm
def tqdm(iterable, desc="Progress"):
    total = len(iterable) if hasattr(iterable, '__len__') else None
    for i, item in enumerate(iterable):
        if total and i % max(1, total // 10) == 0:
            print(f"{desc}: {i}/{total}")
        yield item

from tnc.models import RnnEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Import improved methods
try:
    from evaluations.improved_few_shot import (
        LinearPrototypicalNetwork, MetricPrototypicalNetwork, 
        HybridFewShotClassifier, AdaptivePrototypicalNetwork
    )
    IMPROVED_METHODS_AVAILABLE = True
except ImportError:
    IMPROVED_METHODS_AVAILABLE = False
    print("⚠️  Improved methods not available - using baseline methods only")

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class FewShotEvaluator:
    """Few-shot learning evaluator for pre-trained TNC encoder"""
    
    def __init__(self, encoder_path, data_path, window_size=50):
        self.encoder_path = encoder_path
        self.data_path = data_path
        self.window_size = window_size
        self.device = device
        
        # Load pre-trained encoder
        self.encoder = self._load_encoder()
        
        # Load and prepare data
        self.X_train, self.y_train, self.X_test, self.y_test = self._load_data()
        
        print(f"✅ Loaded pre-trained TNC encoder from: {encoder_path}")
        print(f"📊 Dataset: {len(self.X_train)} train, {len(self.X_test)} test samples")
        print(f"🎯 Classes: {len(np.unique(self.y_train))} ({np.unique(self.y_train)})")
        print(f"🔧 Feature dimension: {self.encoder.encoding_size}D")
        
        # Initialize improved methods if available
        if IMPROVED_METHODS_AVAILABLE:
            self._init_improved_methods()
        
    def _init_improved_methods(self):
        """Initialize improved few-shot learning models"""
        input_dim = self.encoder.encoding_size
        
        self.improved_models = {
            'linear_prototypical': LinearPrototypicalNetwork(input_dim=input_dim, output_dim=16).to(self.device),
            'metric_prototypical': MetricPrototypicalNetwork(input_dim=input_dim).to(self.device),
            'hybrid': HybridFewShotClassifier(input_dim=input_dim, n_classes=4).to(self.device),
            'adaptive': AdaptivePrototypicalNetwork(input_dim=input_dim).to(self.device)
        }
        print("🚀 Improved few-shot methods initialized!")
        
    def _load_encoder(self):
        """Load and initialize pre-trained TNC encoder"""
        # Load checkpoint
        checkpoint = torch.load(self.encoder_path, map_location=self.device)
        
        # Initialize encoder with same architecture as training
        encoder = RnnEncoder(
            hidden_size=100,
            in_channel=3, 
            encoding_size=10,
            device=self.device
        )
        
        # Load pre-trained weights
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder.eval()
        encoder.to(self.device)
        
        return encoder
    
    def _load_data(self):
        """Load and preprocess simulated data"""
        # Load raw data
        with open(os.path.join(self.data_path, 'x_train.pkl'), 'rb') as f:
            x_train = pickle.load(f)
        with open(os.path.join(self.data_path, 'state_train.pkl'), 'rb') as f:
            y_train = pickle.load(f)
        with open(os.path.join(self.data_path, 'x_test.pkl'), 'rb') as f:
            x_test = pickle.load(f)
        with open(os.path.join(self.data_path, 'state_test.pkl'), 'rb') as f:
            y_test = pickle.load(f)
        
        # Extract features using pre-trained encoder
        print("🔄 Extracting features from training data...")
        X_train_features = self._extract_features(x_train)
        print("🔄 Extracting features from test data...")
        X_test_features = self._extract_features(x_test)
        
        # Create labels (majority vote per sample)
        y_train_labels = self._create_labels(y_train)
        y_test_labels = self._create_labels(y_test)
        
        return X_train_features, y_train_labels, X_test_features, y_test_labels
    
    def _extract_features(self, data):
        """Extract features using pre-trained TNC encoder"""
        features = []
        
        with torch.no_grad():
            for i in tqdm(range(len(data)), desc="Extracting features"):
                sample = data[i]
                T = sample.shape[-1]
                
                # Extract multiple overlapping windows from each sample
                windows = []
                step = self.window_size // 4  # 75% overlap
                
                for t in range(self.window_size//2, T - self.window_size//2, step):
                    window = sample[:, t-self.window_size//2:t+self.window_size//2]
                    windows.append(torch.tensor(window, dtype=torch.float32))
                
                if windows:
                    windows_tensor = torch.stack(windows).to(self.device)
                    encoded = self.encoder(windows_tensor)
                    # Average the encodings from multiple windows
                    avg_encoding = torch.mean(encoded, dim=0)
                    features.append(avg_encoding.cpu().numpy())
                else:
                    # Fallback: use the middle window
                    mid_t = T // 2
                    window = sample[:, mid_t-self.window_size//2:mid_t+self.window_size//2]
                    window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
                    encoded = self.encoder(window_tensor)
                    features.append(encoded.squeeze(0).cpu().numpy())
        
        return np.array(features)
    
    def _create_labels(self, states):
        """Create labels using majority vote"""
        labels = []
        for state_sequence in states:
            # Take the most frequent state as the label
            unique, counts = np.unique(state_sequence, return_counts=True)
            majority_label = unique[np.argmax(counts)]
            labels.append(int(majority_label))
        return np.array(labels)
    
    def prototypical_networks(self, n_way, n_shot, n_query=15, n_episodes=100):
        """
        Prototypical Networks evaluation
        
        Args:
            n_way: Number of classes per episode
            n_shot: Number of support examples per class
            n_query: Number of query examples per class
            n_episodes: Number of episodes to run
        """
        print(f"\n🎯 Running Prototypical Networks: {n_way}-way {n_shot}-shot")
        
        accuracies = []
        
        for episode in tqdm(range(n_episodes), desc=f"{n_way}-way {n_shot}-shot"):
            # Sample classes for this episode
            available_classes = np.unique(self.y_train)
            if len(available_classes) < n_way:
                continue
                
            episode_classes = np.random.choice(available_classes, n_way, replace=False)
            
            # Sample support and query sets
            support_x, support_y, query_x, query_y = [], [], [], []
            
            for class_idx, class_label in enumerate(episode_classes):
                # Get all samples for this class
                class_indices = np.where(self.y_train == class_label)[0]
                
                if len(class_indices) < n_shot + n_query:
                    continue
                
                # Sample support and query examples
                selected_indices = np.random.choice(
                    class_indices, n_shot + n_query, replace=False
                )
                
                # Support set
                support_indices = selected_indices[:n_shot]
                support_x.extend(self.X_train[support_indices])
                support_y.extend([class_idx] * n_shot)
                
                # Query set  
                query_indices = selected_indices[n_shot:n_shot + n_query]
                query_x.extend(self.X_train[query_indices])
                query_y.extend([class_idx] * n_query)
            
            if len(support_x) == 0 or len(query_x) == 0:
                continue
                
            support_x = np.array(support_x)
            support_y = np.array(support_y)
            query_x = np.array(query_x)
            query_y = np.array(query_y)
            
            # Compute prototypes (class centroids)
            prototypes = []
            for class_idx in range(n_way):
                class_support = support_x[support_y == class_idx]
                if len(class_support) > 0:
                    prototype = np.mean(class_support, axis=0)
                    prototypes.append(prototype)
            
            if len(prototypes) != n_way:
                continue
                
            prototypes = np.array(prototypes)
            
            # Classify query examples using nearest prototype
            predictions = []
            for query_example in query_x:
                # Compute distances to all prototypes
                distances = np.linalg.norm(prototypes - query_example, axis=1)
                # Predict class with minimum distance
                pred_class = np.argmin(distances)
                predictions.append(pred_class)
            
            # Compute accuracy for this episode
            episode_accuracy = accuracy_score(query_y, predictions)
            accuracies.append(episode_accuracy)
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f"   📈 Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        return mean_acc, std_acc
    
    def improved_prototypical_networks(self, method_name, n_way, n_shot, n_query=15, n_episodes=100):
        """
        Improved Prototypical Networks evaluation
        
        Args:
            method_name: Name of improved method ('linear_prototypical', 'metric_prototypical', etc.)
            n_way: Number of classes per episode
            n_shot: Number of support examples per class
            n_query: Number of query examples per class
            n_episodes: Number of episodes to run
        """
        if not IMPROVED_METHODS_AVAILABLE or not hasattr(self, 'improved_models'):
            print(f"⚠️  Improved method {method_name} not available")
            return 0.0, 0.0
            
        print(f"\n🎯 Running Improved {method_name}: {n_way}-way {n_shot}-shot")
        
        model = self.improved_models[method_name]
        model.train()  # Enable training mode for learning
        
        # Train the model first
        self._train_improved_method(model, method_name, n_way, n_shot, n_episodes=50)
        
        # Evaluate the trained model
        model.eval()
        accuracies = []
        
        import torch
        
        for episode in tqdm(range(n_episodes), desc=f"{method_name} {n_way}-way {n_shot}-shot"):
            # Sample episode data
            support_x, support_y, query_x, query_y = self._sample_episode(n_way, n_shot, n_query)
            
            if len(support_x) == 0 or len(query_x) == 0:
                continue
            
            # Convert to tensors
            support_features = torch.tensor(support_x, dtype=torch.float32).to(self.device)
            support_labels = torch.tensor(support_y, dtype=torch.long).to(self.device)
            query_features = torch.tensor(query_x, dtype=torch.float32).to(self.device)
            query_labels = torch.tensor(query_y, dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                if method_name == 'hybrid':
                    logits, _, _, _ = model(support_features, support_labels, 
                                         query_features, n_way, n_shot)
                else:
                    logits = model(support_features, support_labels, 
                                 query_features, n_way, n_shot)
                
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == query_labels).float().mean().item()
                accuracies.append(accuracy)
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f"   📈 Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        return mean_acc, std_acc
    
    def _sample_episode(self, n_way, n_shot, n_query):
        """Sample a few-shot learning episode"""
        available_classes = np.unique(self.y_train)
        if len(available_classes) < n_way:
            return [], [], [], []
        
        episode_classes = np.random.choice(available_classes, n_way, replace=False)
        
        support_x, support_y, query_x, query_y = [], [], [], []
        
        for class_idx, class_label in enumerate(episode_classes):
            class_indices = np.where(self.y_train == class_label)[0]
            
            if len(class_indices) < n_shot + n_query:
                continue
            
            selected_indices = np.random.choice(
                class_indices, n_shot + n_query, replace=False
            )
            
            # Support set
            support_indices = selected_indices[:n_shot]
            support_x.extend(self.X_train[support_indices])
            support_y.extend([class_idx] * n_shot)
            
            # Query set
            query_indices = selected_indices[n_shot:n_shot + n_query]
            query_x.extend(self.X_train[query_indices])
            query_y.extend([class_idx] * n_query)
        
        return np.array(support_x), np.array(support_y), np.array(query_x), np.array(query_y)
    
    def _train_improved_method(self, model, method_name, n_way, n_shot, n_episodes=50, lr=0.001):
        """Train an improved method using few-shot episodes"""
        import torch
        import torch.nn as nn
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        
        for episode in range(n_episodes):
            support_x, support_y, query_x, query_y = self._sample_episode(n_way, n_shot, n_query=10)
            
            if len(support_x) == 0 or len(query_x) == 0:
                continue
            
            # Convert to tensors
            support_features = torch.tensor(support_x, dtype=torch.float32).to(self.device)
            support_labels = torch.tensor(support_y, dtype=torch.long).to(self.device)
            query_features = torch.tensor(query_x, dtype=torch.float32).to(self.device)
            query_labels = torch.tensor(query_y, dtype=torch.long).to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            if method_name == 'hybrid':
                logits, _, _, _ = model(support_features, support_labels, 
                                     query_features, n_way, n_shot)
            else:
                logits = model(support_features, support_labels, 
                             query_features, n_way, n_shot)
            
            loss = criterion(logits, query_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
    
    def nearest_neighbor_baseline(self, n_shots, n_trials=50):
        """k-Nearest Neighbors baseline for few-shot learning"""
        print(f"\n🔍 Running k-NN Baseline: {n_shots}-shot")
        
        accuracies = []
        
        for trial in range(n_trials):
            # Create stratified few-shot training set
            support_x, support_y = [], []
            
            for class_label in np.unique(self.y_train):
                class_indices = np.where(self.y_train == class_label)[0]
                if len(class_indices) >= n_shots:
                    selected = np.random.choice(class_indices, n_shots, replace=False)
                    support_x.extend(self.X_train[selected])
                    support_y.extend([class_label] * n_shots)
            
            if len(support_x) == 0:
                continue
                
            support_x = np.array(support_x)
            support_y = np.array(support_y)
            
            # Train k-NN classifier
            k = min(3, len(support_x))  # Adaptive k
            knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
            knn.fit(support_x, support_y)
            
            # Evaluate on test set
            predictions = knn.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, predictions)
            accuracies.append(accuracy)
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f"   📈 Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        return mean_acc, std_acc
    
    def linear_baseline(self, n_shots, n_trials=50):
        """Linear classifier baseline for few-shot learning"""
        print(f"\n📏 Running Linear Baseline: {n_shots}-shot")
        
        accuracies = []
        
        for trial in range(n_trials):
            # Create stratified few-shot training set
            support_x, support_y = [], []
            
            for class_label in np.unique(self.y_train):
                class_indices = np.where(self.y_train == class_label)[0]
                if len(class_indices) >= n_shots:
                    selected = np.random.choice(class_indices, n_shots, replace=False)
                    support_x.extend(self.X_train[selected])
                    support_y.extend([class_label] * n_shots)
            
            if len(support_x) == 0:
                continue
                
            support_x = np.array(support_x)
            support_y = np.array(support_y)
            
            # Train linear classifier
            clf = LogisticRegression(max_iter=1000, random_state=trial)
            clf.fit(support_x, support_y)
            
            # Evaluate on test set
            predictions = clf.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, predictions)
            accuracies.append(accuracy)
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f"   📈 Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        return mean_acc, std_acc
    
    def run_comprehensive_evaluation(self, shot_numbers=[1, 3, 5, 10, 20], n_trials=50):
        """Run comprehensive few-shot evaluation with multiple methods"""
        print("🚀 Starting Comprehensive Few-Shot Evaluation")
        print("=" * 60)
        
        results = {
            'prototypical': {'shots': [], 'mean': [], 'std': []},
            'knn': {'shots': [], 'mean': [], 'std': []},
            'linear': {'shots': [], 'mean': [], 'std': []}
        }
        
        # Add improved methods if available
        if IMPROVED_METHODS_AVAILABLE and hasattr(self, 'improved_models'):
            for method_name in self.improved_models.keys():
                results[method_name] = {'shots': [], 'mean': [], 'std': []}
        
        for n_shots in shot_numbers:
            print(f"\n🎯 Evaluating {n_shots}-shot learning...")
            
            # Prototypical Networks
            proto_mean, proto_std = self.prototypical_networks(
                n_way=4, n_shot=n_shots, n_query=10, n_episodes=n_trials
            )
            results['prototypical']['shots'].append(n_shots)
            results['prototypical']['mean'].append(proto_mean)
            results['prototypical']['std'].append(proto_std)
            
            # k-NN Baseline
            knn_mean, knn_std = self.nearest_neighbor_baseline(n_shots, n_trials)
            results['knn']['shots'].append(n_shots)
            results['knn']['mean'].append(knn_mean)
            results['knn']['std'].append(knn_std)
            
            # Linear Baseline
            linear_mean, linear_std = self.linear_baseline(n_shots, n_trials)
            results['linear']['shots'].append(n_shots)
            results['linear']['mean'].append(linear_mean)
            results['linear']['std'].append(linear_std)
            
            # Improved Methods
            if IMPROVED_METHODS_AVAILABLE and hasattr(self, 'improved_models'):
                for method_name in self.improved_models.keys():
                    improved_mean, improved_std = self.improved_prototypical_networks(
                        method_name, n_way=4, n_shot=n_shots, n_episodes=min(n_trials, 30)
                    )
                    results[method_name]['shots'].append(n_shots)
                    results[method_name]['mean'].append(improved_mean)
                    results[method_name]['std'].append(improved_std)
        
        # Print summary
        self._print_results_summary(results)
        
        # Create visualizations
        self._plot_results(results)
        
        return results
    
    def _print_results_summary(self, results):
        """Print formatted results summary"""
        print("\n" + "=" * 80)
        print("📊 FEW-SHOT LEARNING RESULTS SUMMARY")
        print("=" * 80)
        
        print(f"{'Shots':<6} {'Prototypical':<20} {'k-NN':<20} {'Linear':<20}")
        print("-" * 80)
        
        for i, n_shots in enumerate(results['prototypical']['shots']):
            proto = f"{results['prototypical']['mean'][i]:.3f}±{results['prototypical']['std'][i]:.3f}"
            knn = f"{results['knn']['mean'][i]:.3f}±{results['knn']['std'][i]:.3f}"
            linear = f"{results['linear']['mean'][i]:.3f}±{results['linear']['std'][i]:.3f}"
            
            print(f"{n_shots:<6} {proto:<20} {knn:<20} {linear:<20}")
        
        print("-" * 80)
        
        # Find best method for each shot number
        print("\n🏆 Best Method by Shot Number:")
        for i, n_shots in enumerate(results['prototypical']['shots']):
            methods = {
                'Prototypical': results['prototypical']['mean'][i],
                'k-NN': results['knn']['mean'][i],
                'Linear': results['linear']['mean'][i]
            }
            best_method = max(methods, key=methods.get)
            best_score = methods[best_method]
            print(f"   {n_shots}-shot: {best_method} ({best_score:.3f})")
    
    def _plot_results(self, results):
        """Create visualization of few-shot results"""
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Accuracy vs Shots
        plt.subplot(2, 2, 1)
        for method, data in results.items():
            plt.errorbar(data['shots'], data['mean'], yerr=data['std'], 
                        marker='o', linewidth=2, capsize=5, label=method.capitalize())
        
        plt.xlabel('Number of Shots per Class')
        plt.ylabel('Accuracy')
        plt.title('Few-Shot Learning Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Method Comparison (Bar chart for 5-shot)
        plt.subplot(2, 2, 2)
        if 5 in results['prototypical']['shots']:
            idx = results['prototypical']['shots'].index(5)
            methods = ['Prototypical', 'k-NN', 'Linear']
            means = [
                results['prototypical']['mean'][idx],
                results['knn']['mean'][idx],
                results['linear']['mean'][idx]
            ]
            stds = [
                results['prototypical']['std'][idx],
                results['knn']['std'][idx],
                results['linear']['std'][idx]
            ]
            
            bars = plt.bar(methods, means, yerr=stds, capsize=5, alpha=0.7)
            plt.ylabel('Accuracy')
            plt.title('5-Shot Learning Comparison')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean, std in zip(bars, means, stds):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                        f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Improvement over random
        plt.subplot(2, 2, 3)
        random_baseline = 0.25  # 25% for 4-class problem
        
        for method, data in results.items():
            improvement = [(acc - random_baseline) / random_baseline * 100 
                          for acc in data['mean']]
            plt.plot(data['shots'], improvement, marker='o', linewidth=2, 
                    label=f'{method.capitalize()}')
        
        plt.xlabel('Number of Shots per Class')
        plt.ylabel('Improvement over Random (%)')
        plt.title('Performance Improvement over Random Baseline')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Learning curves
        plt.subplot(2, 2, 4)
        for method, data in results.items():
            plt.plot(data['shots'], data['mean'], 'o-', linewidth=2, 
                    label=f'{method.capitalize()}')
        
        plt.axhline(y=random_baseline, color='red', linestyle='--', 
                   label='Random (25%)')
        plt.xlabel('Number of Shots per Class')
        plt.ylabel('Accuracy')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_dir = './plots/simulation'
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, 'few_shot_evaluation.png'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plot_dir, 'few_shot_evaluation.pdf'), 
                   bbox_inches='tight')
        
        print(f"\n📊 Plots saved to {plot_dir}/few_shot_evaluation.[png|pdf]")
        plt.show()
    
    def analyze_feature_quality(self):
        """Analyze the quality of extracted features"""
        print("\n🔍 Analyzing Feature Quality...")
        
        # Compute inter-class and intra-class distances
        from sklearn.metrics.pairwise import pairwise_distances
        
        distances = pairwise_distances(self.X_train, metric='euclidean')
        
        inter_class_dists = []
        intra_class_dists = []
        
        for i in range(len(self.X_train)):
            for j in range(i+1, len(self.X_train)):
                dist = distances[i, j]
                if self.y_train[i] == self.y_train[j]:
                    intra_class_dists.append(dist)
                else:
                    inter_class_dists.append(dist)
        
        inter_mean = np.mean(inter_class_dists)
        intra_mean = np.mean(intra_class_dists)
        separation_ratio = inter_mean / intra_mean
        
        print(f"   📏 Average intra-class distance: {intra_mean:.3f}")
        print(f"   📏 Average inter-class distance: {inter_mean:.3f}")
        print(f"   🎯 Separation ratio: {separation_ratio:.3f}")
        print(f"   💡 Higher separation ratio = better feature quality")
        
        # Visualize feature distribution using t-SNE
        try:
            from sklearn.manifold import TSNE
            print("   🔄 Computing t-SNE visualization...")
            
            # Sample subset for t-SNE (computational efficiency)
            if len(self.X_train) > 500:
                indices = np.random.choice(len(self.X_train), 500, replace=False)
                X_subset = self.X_train[indices]
                y_subset = self.y_train[indices]
            else:
                X_subset = self.X_train
                y_subset = self.y_train
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            X_tsne = tsne.fit_transform(X_subset)
            
            plt.figure(figsize=(10, 8))
            colors = ['red', 'blue', 'green', 'orange']
            for class_label in np.unique(y_subset):
                mask = y_subset == class_label
                plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                           c=colors[int(class_label)], alpha=0.6, 
                           label=f'State {int(class_label)}')
            
            plt.title('t-SNE Visualization of TNC Features')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save t-SNE plot
            plot_dir = './plots/simulation'
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, 'tnc_features_tsne.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"   📊 t-SNE plot saved to {plot_dir}/tnc_features_tsne.png")
            plt.show()
            
        except ImportError:
            print("   ⚠️  Install scikit-learn for t-SNE visualization")
        
        return {
            'intra_class_distance': intra_mean,
            'inter_class_distance': inter_mean,
            'separation_ratio': separation_ratio
        }


def main():
    parser = argparse.ArgumentParser(description='Few-Shot Learning Evaluation for TNC Encoder')
    parser.add_argument('--data', type=str, default='simulation', 
                       help='Dataset name (simulation, har, waveform)')
    parser.add_argument('--encoder_path', type=str, default='./ckpt/simulation/checkpoint_0.pth.tar',
                       help='Path to pre-trained TNC encoder checkpoint')
    parser.add_argument('--data_path', type=str, default='./data/simulated_data/',
                       help='Path to dataset directory')
    parser.add_argument('--method', type=str, default='all', 
                       choices=['prototypical', 'knn', 'linear', 'improved', 'all',
                               'linear_prototypical', 'metric_prototypical', 'hybrid', 'adaptive'],
                       help='Few-shot method to evaluate')
    parser.add_argument('--shots', type=str, default='1,3,5,10,20',
                       help='Comma-separated list of shot numbers')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of trials per evaluation')
    parser.add_argument('--analyze_features', action='store_true',
                       help='Analyze feature quality')
    
    args = parser.parse_args()
    
    # Parse shot numbers
    shot_numbers = [int(x.strip()) for x in args.shots.split(',')]
    
    print("🚀 TNC Encoder Few-Shot Learning Evaluation")
    print("=" * 60)
    print(f"📁 Dataset: {args.data}")
    print(f"🏹 Shot numbers: {shot_numbers}")
    print(f"🔄 Trials per evaluation: {args.trials}")
    print(f"🎯 Method: {args.method}")
    
    # Initialize evaluator
    evaluator = FewShotEvaluator(
        encoder_path=args.encoder_path,
        data_path=args.data_path
    )
    
    # Analyze feature quality
    if args.analyze_features:
        feature_stats = evaluator.analyze_feature_quality()
    
    # Run evaluations
    if args.method == 'all':
        results = evaluator.run_comprehensive_evaluation(shot_numbers, args.trials)
    elif args.method == 'prototypical':
        print("🎯 Running Prototypical Networks Only")
        for n_shots in shot_numbers:
            evaluator.prototypical_networks(n_way=4, n_shot=n_shots, n_episodes=args.trials)
    elif args.method == 'knn':
        print("🔍 Running k-NN Baseline Only")
        for n_shots in shot_numbers:
            evaluator.nearest_neighbor_baseline(n_shots, args.trials)
    elif args.method == 'linear':
        print("📏 Running Linear Baseline Only")
        for n_shots in shot_numbers:
            evaluator.linear_baseline(n_shots, args.trials)
    elif args.method == 'improved':
        if IMPROVED_METHODS_AVAILABLE and hasattr(evaluator, 'improved_models'):
            print("🚀 Running All Improved Methods")
            for method_name in evaluator.improved_models.keys():
                print(f"\n🎯 Testing {method_name}")
                for n_shots in shot_numbers:
                    evaluator.improved_prototypical_networks(method_name, n_way=4, n_shot=n_shots, n_episodes=args.trials)
        else:
            print("⚠️  Improved methods not available")
    elif args.method in ['linear_prototypical', 'metric_prototypical', 'hybrid', 'adaptive']:
        if IMPROVED_METHODS_AVAILABLE and hasattr(evaluator, 'improved_models'):
            print(f"🎯 Running {args.method} Only")
            for n_shots in shot_numbers:
                evaluator.improved_prototypical_networks(args.method, n_way=4, n_shot=n_shots, n_episodes=args.trials)
        else:
            print(f"⚠️  {args.method} method not available")
    
    print("\n✅ Few-shot evaluation completed!")
    print("📊 Check ./plots/simulation/ for visualization results")


if __name__ == '__main__':
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    main()