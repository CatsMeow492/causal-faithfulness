# Statistical Analysis and Validation Report

## Summary

- Overall Assessment: **POOR**
- Significant Statistical Tests: 0.0%
- Passed Sanity Checks: 50.0%

## Statistical Tests

### Pairwise Comparisons

- IntegratedGradients_vs_LIME: Paired t-test: statistic=nan, p=nan (not significant)
- IntegratedGradients_vs_LIME_wilcoxon: Wilcoxon signed-rank test: statistic=0.0000, p=nan (not significant)
- IntegratedGradients_vs_Random: Paired t-test: statistic=nan, p=nan (not significant)
- IntegratedGradients_vs_Random_wilcoxon: Wilcoxon signed-rank test: statistic=0.0000, p=nan (not significant)
- LIME_vs_Random: Paired t-test: statistic=nan, p=nan (not significant)
- LIME_vs_Random_wilcoxon: Wilcoxon signed-rank test: statistic=0.0000, p=nan (not significant)

### Bootstrap Confidence Intervals

- IntegratedGradients: Bootstrap: mean=0.0000 ± 0.0000, CI=[0.0000, 0.0000]
- LIME: Bootstrap: mean=0.0000 ± 0.0000, CI=[0.0000, 0.0000]
- Random: Bootstrap: mean=0.0000 ± 0.0000, CI=[0.0000, 0.0000]

## Sanity Checks

- Metric bounds validation: PASSED (score=1.0000, threshold=0.9500)
- Random explainer validation: FAILED (score=0.0000, threshold=0.1000)

