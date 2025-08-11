# Statistical Analysis and Validation Report

## Summary

- Overall Assessment: **GOOD**
- Significant Statistical Tests: 100.0%
- Passed Sanity Checks: 100.0%

## Statistical Tests

### Pairwise Comparisons

- IntegratedGradients_vs_Random: Paired t-test: statistic=60.6953, p=0.0000 (significant)
- IntegratedGradients_vs_Random_wilcoxon: Wilcoxon signed-rank test: statistic=0.0000, p=0.0000 (significant)

### Bootstrap Confidence Intervals

- IntegratedGradients: Bootstrap: mean=0.9976 ± 0.0021, CI=[0.9930, 1.0000]
- Random: Bootstrap: mean=0.1279 ± 0.0141, CI=[0.1009, 0.1561]

## Sanity Checks

- Metric bounds validation: PASSED (score=1.0000, threshold=0.9500)
- Random explainer validation: PASSED (score=0.8699, threshold=0.1000)

