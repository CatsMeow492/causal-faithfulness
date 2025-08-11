# Statistical Analysis and Validation Report

## Summary

- Overall Assessment: **GOOD**
- Significant Statistical Tests: 0.0%
- Passed Sanity Checks: 100.0%

## Statistical Tests

### Pairwise Comparisons

- SHAP_vs_LIME: Paired t-test: statistic=-0.7437, p=0.4643 (not significant)
- SHAP_vs_LIME_wilcoxon: Wilcoxon signed-rank test: statistic=47.0000, p=0.4603 (not significant)

### Bootstrap Confidence Intervals

- SHAP: Bootstrap: mean=0.0541 ± 0.0165, CI=[0.0241, 0.0889]
- LIME: Bootstrap: mean=0.0570 ± 0.0167, CI=[0.0266, 0.0920]

## Sanity Checks

- Metric bounds validation: PASSED (score=1.0000, threshold=0.9500)

