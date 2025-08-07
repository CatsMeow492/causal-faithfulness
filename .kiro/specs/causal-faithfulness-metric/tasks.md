# Implementation Plan

- [x] 1. Set up project structure and environment
  - Create directory structure (/paper, /src, /data, /results) with proper organization
  - Set up Python environment with requirements.txt including PyTorch ≥ 2.1, transformers, numpy, pandas, scikit-learn, matplotlib, seaborn
  - Configure Mac M-series compatibility with MPS support and CPU fallbacks
  - Add MIT license and basic README
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 7.6_

- [x] 2. Implement core faithfulness metric computation
- [x] 2.1 Create base metric computation framework
  - Implement FaithfulnessConfig and FaithfulnessResult dataclasses
  - Write core F(E) formula computation with Monte-Carlo sampling
  - Add statistical analysis functions for confidence intervals and significance testing
  - _Requirements: 3.2, 5.5, 8.5_

- [x] 2.2 Build feature masking engine
  - Implement modality-specific masking strategies (PAD tokens for text, zeros for tabular, noise for images)
  - Create FeatureMasker class with support for different baseline strategies
  - Add input validation and error handling for various data types
  - _Requirements: 4.1, 8.1_

- [x] 2.3 Implement baseline generation system
  - Create BaselineGenerator class for x_rand baseline creation
  - Implement random, mean, and zero baseline strategies per modality
  - Add batch processing capabilities for efficient computation
  - _Requirements: 3.2, 8.1, 8.2_

- [x] 3. Create explanation method wrappers
- [x] 3.1 Implement SHAP wrapper
  - Create SHAPWrapper class supporting KernelSHAP, TreeSHAP, and DeepSHAP
  - Implement unified Attribution dataclass and interface
  - Add error handling for SHAP-specific edge cases
  - _Requirements: 4.2, 4.3_

- [x] 3.2 Implement Integrated Gradients wrapper
  - Create IntegratedGradientsWrapper with gradient computation
  - Handle different baseline strategies for IG (zeros, random)
  - Add GPU/CPU compatibility for gradient computations
  - _Requirements: 4.2, 4.3_

- [x] 3.3 Implement LIME wrapper
  - Create LIMEWrapper for local perturbation-based explanations
  - Configure perturbation parameters (500 samples as specified)
  - Add support for different data modalities
  - _Requirements: 4.2, 4.3_

- [x] 3.4 Create random explainer baseline
  - Implement RandomExplainer as negative control
  - Generate random attribution scores for sanity checking
  - Ensure consistent interface with other explainers
  - _Requirements: 5.3, 8.4_

- [x] 4. Build evaluation and experimental framework
- [x] 4.1 Create dataset loading and preprocessing
  - Implement SST-2 dataset loader with BERT tokenization
  - Create WikiText-2 dataset loader for GPT-2 experiments
  - Add data validation and preprocessing pipelines
  - Ensure dataset license compliance documentation
  - _Requirements: 5.1, 7.8_

- [x] 4.2 Implement model integration
  - Create model wrapper for bert-base-uncased fine-tuned on SST-2
  - Implement gpt2-small model wrapper for perplexity computation
  - Add hardware-aware model loading (MPS/CPU fallbacks)
  - _Requirements: 5.1, 1.4, 8.3_

- [x] 4.3 Build evaluation pipeline
  - Implement evaluate_explainer function with model-agnostic I/O
  - Create batch processing for 200 validation instances per task
  - Add progress tracking and intermediate result saving
  - _Requirements: 4.3, 5.2_

- [x] 5. Implement ROAR comparison and validation
- [x] 5.1 Create ROAR benchmark implementation
  - Implement feature removal and model retraining/evaluation
  - Calculate accuracy drop metrics for comparison
  - Add correlation computation with Pearson r and p-values
  - _Requirements: 5.4_

- [x] 5.2 Add statistical analysis and validation
  - Implement paired t-tests and bootstrap confidence intervals
  - Create sanity check functions (random explainer validation)
  - Add significance testing with p < 0.05 threshold
  - _Requirements: 5.4, 5.5, 8.4_

- [x] 6. Create visualization and analysis tools
- [x] 6.1 Implement result visualization
  - Create bar charts showing mean F scores by explainer
  - Generate scatter plots for F vs ROAR correlation analysis
  - Add performance tables with runtime and memory usage metrics
  - Save all plots to /figures with proper captions and metadata
  - _Requirements: 6.1, 6.2, 6.3_

- [x] 6.2 Build analysis and reporting system
  - Implement statistical summary generation
  - Create automated finding identification (highest faithfulness explainers)
  - Add correlation analysis with counter-example detection
  - _Requirements: 6.4, 6.5_

- [x] 7. Add comprehensive testing suite
- [x] 7.1 Create unit tests for core components
  - Test metric axiom satisfaction (monotonicity, normalization, bounds)
  - Validate masking strategies and baseline generation
  - Test statistical computation accuracy
  - _Requirements: 3.3, 3.4, 3.5_

- [x] 7.2 Implement integration tests
  - Test end-to-end pipeline on toy datasets
  - Validate cross-platform compatibility
  - Add memory and runtime benchmarks
  - _Requirements: 4.4_

- [x] 7.3 Add validation and sanity checks
  - Test random explainer produces lower scores
  - Validate reproducibility with fixed seeds
  - Check correlation with ROAR on known cases
  - _Requirements: 8.4, 8.5_

- [x] 8. Optimize performance and add hardware support
- [x] 8.1 Implement hardware optimization
  - Add MPS acceleration for Apple Silicon
  - Create CPU fallbacks for unsupported operations
  - Implement automatic batch size adjustment for memory management
  - _Requirements: 1.4, 8.2, 8.3_

- [x] 8.2 Add computational robustness
  - Implement memory management with OOM handling
  - Add numerical stability checks (epsilon handling, gradient clipping)
  - Create streaming computation for large datasets
  - _Requirements: 8.2_

- [x] 9. Create documentation and reproducibility assets
- [x] 9.1 Generate comprehensive documentation
  - Document all hyperparameters and configuration options
  - Create API documentation for all public interfaces
  - Add usage examples and tutorials
  - _Requirements: 7.3, 8.1_

- [x] 9.2 Ensure reproducibility
  - Fix all random seeds and document seed values
  - Create reproducible experiment scripts
  - Add version pinning for all dependencies
  - _Requirements: 7.3, 8.5_

- [x] 10. Set up continuous integration and release
- [x] 10.1 Create CI/CD pipeline
  - Set up GitHub Actions for automated testing
  - Add unit test execution and toy model F-score computation
  - Configure cross-platform testing (macOS focus)
  - _Requirements: 8.6_

- [x] 10.2 Prepare for release
  - Tag repository version v1.0 with proper release notes
  - Create Zenodo DOI for archival
  - Prepare code and data for public availability
  - _Requirements: 8.1, 8.5, 8.6_