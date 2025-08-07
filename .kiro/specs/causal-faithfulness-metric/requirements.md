# Requirements Document

## Introduction

This project aims to define, analyze, and empirically validate a new model-agnostic metric that quantifies how faithfully a feature-based explanation reflects a model's true decision logic. The causal-faithfulness score will address the current gap in explanation evaluation by providing a single scalar metric that satisfies key axioms and incorporates causal intervention semantics across different model architectures.

## Requirements

### Requirement 1: Repository Structure and Environment Setup

**User Story:** As a researcher, I want a well-organized repository with proper environment setup, so that I can efficiently develop and reproduce the causal-faithfulness metric implementation.

#### Acceptance Criteria

1. WHEN the repository is created THEN the system SHALL have folders for /paper, /src, /data, and /results
2. WHEN the environment is set up THEN the system SHALL support Python ≥ 3.10, PyTorch ≥ 2.1, and Hugging Face Transformers
3. WHEN requirements.txt is created THEN the system SHALL include all necessary dependencies (numpy, pandas, scikit-learn, matplotlib, seaborn)
4. WHEN the environment is tested THEN the system SHALL work on Mac M-series with both CPU and MPS support, including fallback strategies if 4-bit quantization is unavailable

### Requirement 2: Literature Review and Gap Analysis

**User Story:** As a researcher, I want to synthesize existing literature on explanation faithfulness, so that I can identify gaps and position my contribution appropriately.

#### Acceptance Criteria

1. WHEN literature collection is performed THEN the system SHALL gather papers on "explanation faithfulness metric", "ROAR audit", and "causal attribution in XAI"
2. WHEN key papers are analyzed THEN the system SHALL summarize Jain & Wallace 2019, Adebayo 2018, Hooker 2019, Riley 2022, Han 2023, and Doshi-Velez & Kim 2017
3. WHEN the literature review is complete THEN the system SHALL document each paper's goal, method, key definitions, and limitations in lit_review.md
4. WHEN gaps are identified THEN the system SHALL note the absence of a single scalar score satisfying axioms and causal intervention semantics

### Requirement 3: Metric Definition and Theoretical Foundation

**User Story:** As a researcher, I want to formally define the causal-faithfulness metric with theoretical backing, so that it provides a rigorous foundation for explanation evaluation.

#### Acceptance Criteria

1. WHEN axioms are defined THEN the system SHALL specify Causal Influence, Sufficiency, Monotonicity, and Normalization properties
2. WHEN the metric is formulated THEN the system SHALL implement F(E) = 1 - E[|f(x) - f(x_∖E)|] / E[|f(x) - f(x_rand)|] with clearly defined masking strategies for x_∖E and x_rand baselines per modality
3. WHEN theoretical properties are proven THEN the system SHALL demonstrate that F meets all declared axioms
4. WHEN linear model analysis is performed THEN the system SHALL derive closed form and prove equivalence to coefficient-based importance
5. WHEN bounds are established THEN the system SHALL prove F ∈ [0,1] and show sensitivity to true causal features

### Requirement 4: Metric Implementation

**User Story:** As a researcher, I want a robust implementation of the causal-faithfulness metric, so that I can evaluate different explanation methods consistently.

#### Acceptance Criteria

1. WHEN the core metric is implemented THEN the system SHALL provide functions to mask features (using [PAD] tokens for BERT, zero-vectors for numeric features), sample baselines, and batch-query models
2. WHEN explanation wrappers are created THEN the system SHALL support SHAP (KernelSHAP), Integrated Gradients, and LIME
3. WHEN the evaluation interface is built THEN the system SHALL provide evaluate_explainer(model, explainer, data) → score function with clearly defined I/O types for model-agnostic compatibility
4. WHEN the implementation is tested THEN the system SHALL handle different input types and model architectures

### Requirement 5: Empirical Validation

**User Story:** As a researcher, I want to empirically validate the metric across different tasks and models, so that I can demonstrate its effectiveness and reliability.

#### Acceptance Criteria

1. WHEN datasets are prepared THEN the system SHALL use SST-2 with bert-base-uncased and WikiText-2 with gpt2-small
2. WHEN explanations are generated THEN the system SHALL create explanations for 200 validation instances per task
3. WHEN explainers are evaluated THEN the system SHALL compute F scores for SHAP, IG, LIME, and a random explainer baseline using specified Monte-Carlo sample sizes for reproducible expectation computation
4. WHEN sanity checks are performed THEN the system SHALL compare F against ROAR drop-in-accuracy and compute Pearson correlation with significance testing (p-value < 0.05)
5. WHEN results are recorded THEN the system SHALL document means ± standard deviations with 95% confidence intervals and paired statistical tests for F score differences

### Requirement 6: Analysis and Visualization

**User Story:** As a researcher, I want comprehensive analysis and visualization of results, so that I can interpret findings and communicate insights effectively.

#### Acceptance Criteria

1. WHEN visualizations are created THEN the system SHALL generate bar charts showing mean F by explainer and save all plots under /figures with captions including seed values and dataset information
2. WHEN correlations are plotted THEN the system SHALL create scatter plots of F vs. ROAR accuracy drop
3. WHEN performance is analyzed THEN the system SHALL provide tables of runtime and model queries per method
4. WHEN findings are identified THEN the system SHALL determine which explainers have highest causal faithfulness
5. WHEN correlations are assessed THEN the system SHALL identify relationships between F and ROAR, including counter-examples

### Requirement 7: Paper Writing and Documentation

**User Story:** As a researcher, I want a complete research paper documenting the metric, so that I can share the contribution with the scientific community.

#### Acceptance Criteria

1. WHEN the paper structure is created THEN the system SHALL include sections for Introduction, Related Work, Metric Definition, Theory, Experiments, Discussion, and Conclusion
2. WHEN figures and tables are integrated THEN the system SHALL incorporate all visualizations and results from the analysis phase
3. WHEN reproducibility is ensured THEN the system SHALL provide code links, seed values, and hyperparameter documentation
4. WHEN the abstract is written THEN the system SHALL use the title "A Causal-Faithfulness Score for Post-hoc Explanations Across Model Architectures"
5. WHEN appendices are added THEN the system SHALL document masking strategies and implementation details
6. WHEN licensing is addressed THEN the system SHALL include an open-source license (MIT) for code release
7. WHEN ethics compliance is ensured THEN the system SHALL include an ethics statement in the paper template
8. WHEN dataset compliance is verified THEN the system SHALL ensure SST-2 and WikiText usage complies with their respective licenses

### Requirement 8: Computational Robustness and Baseline Strategies

**User Story:** As a researcher, I want robust computational strategies and well-defined baselines, so that the metric is reliable across different hardware configurations and modalities.

#### Acceptance Criteria

1. WHEN baseline strategies are defined THEN the system SHALL document masking approaches per modality (text: [PAD] tokens, tabular: mean/zero, images: noise/blur)
2. WHEN computational limits are addressed THEN the system SHALL implement fallback strategies for memory-intensive operations (sequence truncation, batch size reduction)
3. WHEN hardware compatibility is ensured THEN the system SHALL provide CPU-only alternatives when GPU acceleration is unavailable
4. WHEN negative controls are implemented THEN the system SHALL include random explainer baselines to validate metric discrimination
5. WHEN reproducibility is guaranteed THEN the system SHALL fix random seeds and document all hyperparameters for Monte-Carlo sampling

### Requirement 9: Release and Dissemination

**User Story:** As a researcher, I want to properly release and disseminate the work, so that it can be accessed and used by the research community.

#### Acceptance Criteria

1. WHEN the paper is finalized THEN the system SHALL ensure PDF is under 15 MB and properly formatted
2. WHEN the repository is published THEN the system SHALL push to GitHub with v1.0 tag and Zenodo DOI
3. WHEN the preprint is submitted THEN the system SHALL upload to arXiv under cs.LG category
4. WHEN outreach is performed THEN the system SHALL draft social media content with arXiv link
5. WHEN success criteria are met THEN the system SHALL have F formally defined, empirically validated, and publicly available with reproducible results
6. WHEN continuous integration is set up THEN the system SHALL include GitHub actions that run unit tests and reproduce F computation on a toy model