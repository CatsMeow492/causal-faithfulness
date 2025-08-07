#!/usr/bin/env python3
"""
Text Classification Example for Causal-Faithfulness Metric

This example demonstrates how to use the causal-faithfulness metric
with text classification models and BERT-based explanations.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# Import the faithfulness metric components
from src.faithfulness import FaithfulnessMetric, FaithfulnessConfig
from src.explainers import SHAPWrapper, IntegratedGradientsWrapper, RandomExplainer
from src.masking import DataModality


def load_model_and_tokenizer():
    """Load a pre-trained BERT model for sentiment classification."""
    
    print("Loading BERT model and tokenizer...")
    
    # Use a small BERT model fine-tuned on SST-2
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Set to evaluation mode
    model.eval()
    
    return model, tokenizer


def prepare_text_data():
    """Prepare sample text data for explanation."""
    
    # Sample texts for explanation
    texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "The film was boring and poorly executed. Complete waste of time.",
        "An okay movie with decent acting but a weak plot.",
        "Outstanding performance by the lead actor in this masterpiece.",
        "Not the worst movie I've seen, but definitely not good either."
    ]
    
    return texts


def create_model_wrapper(model, tokenizer):
    """Create a wrapper function for the model that handles tokenization."""
    
    def model_wrapper(texts):
        """
        Wrapper function that takes raw text and returns predictions.
        
        Args:
            texts: List of text strings or single text string
            
        Returns:
            torch.Tensor: Model predictions (logits)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize the texts
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        return logits
    
    return model_wrapper


def create_tokenized_explainer_wrapper(model, tokenizer, explainer):
    """Create an explainer wrapper that works with tokenized inputs."""
    
    def tokenized_explainer(model_func, tokenized_data, **kwargs):
        """
        Explainer wrapper for tokenized data.
        
        Args:
            model_func: Model function (not used directly)
            tokenized_data: Dictionary with tokenized inputs
            
        Returns:
            Attribution object
        """
        # Extract input_ids for explanation
        input_ids = tokenized_data['input_ids']
        
        # Create a function that takes input_ids and returns predictions
        def token_model_func(token_ids):
            if isinstance(token_ids, np.ndarray):
                token_ids = torch.from_numpy(token_ids).long()
            
            # Ensure proper shape
            if len(token_ids.shape) == 1:
                token_ids = token_ids.unsqueeze(0)
            
            # Create attention mask (assume all tokens are valid)
            attention_mask = torch.ones_like(token_ids)
            
            # Get model predictions
            with torch.no_grad():
                outputs = model(input_ids=token_ids, attention_mask=attention_mask)
                return outputs.logits
        
        # Generate explanation
        attribution = explainer.explain(
            model=token_model_func,
            input_data=input_ids,
            **kwargs
        )
        
        return attribution
    
    return tokenized_explainer


def main():
    """Main function demonstrating text classification faithfulness evaluation."""
    
    print("=== Causal-Faithfulness Metric: Text Classification Example ===\n")
    
    # Step 1: Load model and data
    print("1. Loading model and preparing data...")
    model, tokenizer = load_model_and_tokenizer()
    texts = prepare_text_data()
    
    # Create model wrapper
    model_wrapper = create_model_wrapper(model, tokenizer)
    
    print(f"   Loaded model: {model.config.name_or_path}")
    print(f"   Prepared {len(texts)} sample texts")
    
    # Step 2: Select a text for explanation
    text_idx = 0
    selected_text = texts[text_idx]
    print(f"\n2. Selected text for explanation:")
    print(f"   \"{selected_text}\"")
    
    # Get model prediction
    prediction = model_wrapper([selected_text])
    predicted_class = torch.argmax(prediction, dim=1).item()
    probabilities = torch.softmax(prediction, dim=1).squeeze().tolist()
    
    class_names = ["Negative", "Positive"]
    print(f"   Predicted class: {class_names[predicted_class]}")
    print(f"   Probabilities: Negative={probabilities[0]:.3f}, Positive={probabilities[1]:.3f}")
    
    # Step 3: Tokenize the text
    print(f"\n3. Tokenizing text...")
    tokenized = tokenizer(
        selected_text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    input_ids = tokenized['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Tokens: {tokens[:10]}...")  # Show first 10 tokens
    
    # Step 4: Configure faithfulness metric
    print(f"\n4. Configuring faithfulness metric...")
    config = FaithfulnessConfig(
        n_samples=200,  # Reduced for faster computation
        baseline_strategy="random",
        masking_strategy="pad",
        confidence_level=0.95,
        random_seed=42,
        batch_size=8
    )
    
    faithfulness_metric = FaithfulnessMetric(config, modality=DataModality.TEXT)
    print(f"   Configuration: {config.n_samples} samples, text modality")
    
    # Step 5: Initialize explainers
    print(f"\n5. Initializing explainers...")
    
    # Create background data for SHAP (sample of tokenized texts)
    background_texts = texts[1:4]  # Use other texts as background
    background_tokenized = tokenizer(
        background_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # SHAP explainer
    shap_explainer = SHAPWrapper(
        explainer_type="kernel",
        n_samples=50,  # Reduced for faster computation
        random_seed=42
    )
    
    # Integrated Gradients explainer
    ig_explainer = IntegratedGradientsWrapper(
        n_steps=20,  # Reduced for faster computation
        baseline_strategy="zero",
        random_seed=42
    )
    
    # Random explainer (baseline)
    random_explainer = RandomExplainer(
        random_seed=42,
        distribution="uniform"
    )
    
    # Create explainer wrappers for tokenized data
    explainers = {
        "SHAP": create_tokenized_explainer_wrapper(model, tokenizer, shap_explainer),
        "IntegratedGradients": create_tokenized_explainer_wrapper(model, tokenizer, ig_explainer),
        "Random": create_tokenized_explainer_wrapper(model, tokenizer, random_explainer)
    }
    
    print(f"   Initialized {len(explainers)} explainers for text data")
    
    # Step 6: Compute faithfulness scores
    print(f"\n6. Computing faithfulness scores...")
    results = {}
    
    for name, explainer_wrapper in explainers.items():
        print(f"\n   Computing faithfulness for {name}...")
        
        try:
            # Special handling for SHAP (needs background data)
            if name == "SHAP":
                result = faithfulness_metric.compute_faithfulness_score(
                    model=model_wrapper,
                    explainer=lambda m, d: explainer_wrapper(
                        m, d, background_data=background_tokenized['input_ids']
                    ),
                    data=tokenized,
                    target_class=predicted_class
                )
            else:
                result = faithfulness_metric.compute_faithfulness_score(
                    model=model_wrapper,
                    explainer=explainer_wrapper,
                    data=tokenized,
                    target_class=predicted_class
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
    
    # Step 7: Analyze token-level explanations
    print(f"\n7. Token-level explanation analysis:")
    
    # Get explanation from the best performing method
    valid_results = {name: result for name, result in results.items() if result is not None}
    if valid_results:
        best_explainer_name = max(valid_results.keys(), key=lambda x: valid_results[x].f_score)
        
        print(f"   Using {best_explainer_name} for token analysis...")
        
        # Generate explanation for visualization
        try:
            if best_explainer_name == "SHAP":
                attribution = explainers[best_explainer_name](
                    model_wrapper, tokenized, background_data=background_tokenized['input_ids']
                )
            else:
                attribution = explainers[best_explainer_name](model_wrapper, tokenized)
            
            # Get top important tokens
            top_features = attribution.get_top_features(k=5)
            
            print(f"   Top 5 most important tokens:")
            for i, token_idx in enumerate(top_features):
                if token_idx < len(tokens):
                    token = tokens[token_idx]
                    importance = attribution.feature_scores[token_idx]
                    print(f"      {i+1}. '{token}' (importance: {importance:.4f})")
            
        except Exception as e:
            print(f"   Error generating token analysis: {e}")
    
    # Step 8: Compare results
    print(f"\n8. Comparison Summary:")
    print("   " + "="*70)
    print(f"   {'Method':<20} {'F-score':<10} {'95% CI':<20} {'Significant':<12} {'Time(s)':<8}")
    print("   " + "="*70)
    
    for name, result in results.items():
        if result is not None:
            ci_str = f"[{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]"
            sig_str = "Yes" if result.statistical_significance else "No"
            time_str = f"{result.computation_metrics['computation_time_seconds']:.1f}"
            print(f"   {name:<20} {result.f_score:<10.4f} {ci_str:<20} {sig_str:<12} {time_str:<8}")
        else:
            print(f"   {name:<20} {'Failed':<10} {'-':<20} {'-':<12} {'-':<8}")
    
    print("   " + "="*70)
    
    # Step 9: Interpretation
    print(f"\n9. Interpretation for Text Classification:")
    
    if valid_results:
        best_explainer = max(valid_results.keys(), key=lambda x: valid_results[x].f_score)
        best_score = valid_results[best_explainer].f_score
        
        print(f"   • Best explainer: {best_explainer} (F-score: {best_score:.4f})")
        print(f"   • Text: \"{selected_text}\"")
        print(f"   • Predicted: {class_names[predicted_class]} ({probabilities[predicted_class]:.3f})")
        
        # Interpretation based on F-score
        if best_score > 0.7:
            print(f"   • High faithfulness: The explanation method captures the model's decision logic well")
        elif best_score > 0.4:
            print(f"   • Moderate faithfulness: The explanation partially reflects the model's reasoning")
        else:
            print(f"   • Low faithfulness: The explanation may not accurately reflect the model's decisions")
        
        # Check random baseline
        if "Random" in valid_results:
            random_score = valid_results["Random"].f_score
            if random_score < 0.3:
                print(f"   ✓ Sanity check passed: Random explainer has low faithfulness ({random_score:.4f})")
            else:
                print(f"   ⚠ Warning: Random explainer has unexpectedly high faithfulness ({random_score:.4f})")
    
    print(f"\n10. Text-Specific Considerations:")
    print(f"   • Token masking uses [PAD] tokens to mask non-important features")
    print(f"   • Baseline generation uses random token sampling from vocabulary")
    print(f"   • Attention mechanisms in BERT may affect explanation quality")
    print(f"   • Longer texts may require more samples for stable estimates")
    print(f"   • Subword tokenization can affect feature-level interpretations")
    
    print(f"\n=== Text classification example completed successfully! ===")


if __name__ == "__main__":
    main()