# Statistical Analysis and Validation Report

## Summary

- Overall Assessment: **GOOD**
- Significant Statistical Tests: 100.0%
- Passed Sanity Checks: 100.0%

## Statistical Tests

### Pairwise Comparisons

- IntegratedGradients_vs_Random: Paired t-test: statistic=25.9036, p=0.0000 (significant)
- IntegratedGradients_vs_Random_wilcoxon: Wilcoxon signed-rank test: statistic=3.0000, p=0.0000 (significant)

### Bootstrap Confidence Intervals

- IntegratedGradients: Bootstrap: mean=0.6390 ± 0.0241, CI=[0.5914, 0.6863]
- Random: Bootstrap: mean=0.0104 ± 0.0046, CI=[0.0028, 0.0207]

## Sanity Checks

- Metric bounds validation: PASSED (score=1.0000, threshold=0.9500)
- Random explainer validation: PASSED (score=0.6288, threshold=0.1000)

