#!/usr/bin/env python3
"""
Basic Usage Example for Causal-Faithfulness Metric

This example demonstrates the basic usage of the causal-faithfulness metric
with a simple synthetic dataset and model.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import the faithfulness metric components
from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig
from src.explainers import SHAPWrapper, IntegratedGradientsWrapper, LIMEWrapper, RandomExplainer
from src.masking import DataModality


def create_synthetic_model_and_data():
    """Create a synthetic dataset and simple neural network model."""
    
    # Generate synthetic tabular data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Create a simple neural network
    class SimpleNN(nn.Module):
        def __init__(self, input_dim, hidden_dim=20, output_dim=2):
            super(SimpleNN, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, x):
            return self.network(x)
    
    # Initialize and train the model
    model = SimpleNN(input_dim=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Simple training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    
    return model, X_train, X_test_tensor


def main():
    """Main function demonstrating basic usage."""
    
    print("=== Causal-Faithfulness Metric: Basic Usage Example ===\n")
    
    # Step 1: Create synthetic model and data
    print("1. Creating synthetic model and data...")
    model, X_train, X_test = create_synthetic_model_and_data()
    
    # Select a single test instance for explanation
    test_instance = X_test[0:1]  # Keep batch dimension
    print(f"   Test instance shape: {test_instance.shape}")
    print(f"   Model prediction: {torch.softmax(model(test_instance), dim=1).detach().numpy()}")
    
    # Step 2: Configure the faithfulness metric
    print("\n2. Configuring faithfulness metric...")
    config = FaithfulnessConfig(
        n_samples=500,              # Reduced for faster computation
        baseline_strategy="gaussian",
        masking_strategy="mean",
        confidence_level=0.95,
        random_seed=42,
        batch_size=16
    )
    
    # Initialize the metric for tabular data
    faithfulness_metric = FaithfulnessMetric(config, modality=DataModality.TABULAR)
    print(f"   Configuration: {config.n_samples} samples, {config.baseline_strategy} baseline")
    
    # Step 3: Initialize explainers
    print("\n3. Initializing explanation methods...")
    
    # SHAP explainer (using background data)
    background_data = torch.FloatTensor(X_train[:50])  # Use 50 training samples as background
    shap_explainer = SHAPWrapper(
        explainer_type="kernel",
        n_samples=100,  # Reduced for faster computation
        random_seed=42
    )
    
    # Integrated Gradients explainer
    ig_explainer = IntegratedGradientsWrapper(
        n_steps=25,  # Reduced for faster computation
        baseline_strategy="zero",
        random_seed=42
    )
    
    # LIME explainer
    lime_explainer = LIMEWrapper(
        n_samples=100,  # Reduced for faster computation
        modality="tabular",
        random_seed=42
    )
    
    # Random explainer (baseline)
    random_explainer = RandomExplainer(
        random_seed=42,
        distribution="uniform"
    )
    
    explainers = {
        "SHAP": shap_explainer,
        "IntegratedGradients": ig_explainer,
        "LIME": lime_explainer,
        "Random": random_explainer
    }
    
    print(f"   Initialized {len(explainers)} explainers")
    
    # Step 4: Compute faithfulness scores
    print("\n4. Computing faithfulness scores...")
    results = {}
    
    for name, explainer in explainers.items():
        print(f"\n   Computing faithfulness for {name}...")
        
        try:
            # Special handling for explainers that need additional data
            if name == "SHAP":
                result = faithfulness_metric.compute_faithfulness_score(
                    model=model,
                    explainer=lambda m, d: explainer.explain(m, d, background_data=background_data),
                    data=test_instance,
                    target_class=None  # Auto-detect
                )
            elif name == "LIME":
                result = faithfulness_metric.compute_faithfulness_score(
                    model=model,
                    explainer=lambda m, d: explainer.explain(m, d, training_data=X_train),
                    data=test_instance,
                    target_class=None
                )
            else:
                result = faithfulness_metric.compute_faithfulness_score(
                    model=model,
                    explainer=explainer.explain,
                    data=test_instance,
                    target_class=None
                )
            
            results[name] = result
            
            # Print results
            print(f"      F-score: {result.f_score:.4f}")
            print(f"      95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
            print(f"      P-value: {result.p_value:.4f}")
            print(f"      Significant: {result.statistical_significance}")
            print(f"      Computation time: {result.computation_metrics['computation_time_seconds']:.2f}s")
            
        except Exception as e:
            print(f"      Error computing faithfulness for {name}: {e}")
            results[name] = None
    
    # Step 5: Compare results
    print("\n5. Comparison Summary:")
    print("   " + "="*60)
    print(f"   {'Method':<20} {'F-score':<10} {'95% CI':<20} {'Significant':<12}")
    print("   " + "="*60)
    
    for name, result in results.items():
        if result is not None:
            ci_str = f"[{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]"
            sig_str = "Yes" if result.statistical_significance else "No"
            print(f"   {name:<20} {result.f_score:<10.4f} {ci_str:<20} {sig_str:<12}")
        else:
            print(f"   {name:<20} {'Failed':<10} {'-':<20} {'-':<12}")
    
    print("   " + "="*60)
    
    # Step 6: Interpretation
    print("\n6. Interpretation:")
    
    # Find the best explainer
    valid_results = {name: result for name, result in results.items() if result is not None}
    if valid_results:
        best_explainer = max(valid_results.keys(), key=lambda x: valid_results[x].f_score)
        best_score = valid_results[best_explainer].f_score
        
        print(f"   • Best explainer: {best_explainer} (F-score: {best_score:.4f})")
        
        # Check if random explainer performed poorly (sanity check)
        if "Random" in valid_results:
            random_score = valid_results["Random"].f_score
            print(f"   • Random baseline: {random_score:.4f}")
            
            if random_score < 0.3:
                print("   ✓ Sanity check passed: Random explainer has low faithfulness")
            else:
                print("   ⚠ Warning: Random explainer has unexpectedly high faithfulness")
        
        # Statistical significance
        significant_explainers = [
            name for name, result in valid_results.items() 
            if result.statistical_significance
        ]
        
        if significant_explainers:
            print(f"   • Statistically significant explainers: {', '.join(significant_explainers)}")
        else:
            print("   • No explainers showed statistically significant faithfulness")
    
    print("\n7. Notes:")
    print("   • F-scores range from 0 (unfaithful) to 1 (perfectly faithful)")
    print("   • Higher F-scores indicate better explanation faithfulness")
    print("   • Confidence intervals provide uncertainty estimates")
    print("   • Statistical significance indicates reliable differences from baseline")
    print("   • Random explainer should have low F-scores (sanity check)")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()