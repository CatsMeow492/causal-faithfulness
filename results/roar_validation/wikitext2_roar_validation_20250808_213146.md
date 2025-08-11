# ROAR-Faithfulness Validation Report

## Summary Statistics

### Random
- Pearson correlation: r = nan (p = nan)
- Spearman correlation: ρ = nan (p = nan)
- Statistically significant: No
- Sample size: 2

### IntegratedGradients
- Pearson correlation: r = nan (p = nan)
- Spearman correlation: ρ = nan (p = nan)
- Statistically significant: No
- Sample size: 2

### Occlusion
- Pearson correlation: r = nan (p = nan)
- Spearman correlation: ρ = nan (p = nan)
- Statistically significant: No
- Sample size: 2

## Explainer Ranking Consistency

- ROAR ranking: ['Random', 'IntegratedGradients', 'Occlusion']
- Faithfulness ranking: ['Occlusion', 'Random', 'IntegratedGradients']
- Rank correlation: ρ = -0.5000 (p = 0.6667)
- Rankings consistent: No

## Counter-Examples (Significant Disagreements)

### Random
Found 1 counter-examples:
- Example 1: ROAR = 0.0000, Faithfulness = 0.3149

## Overall Assessment

- 0/3 explainers show significant correlation
- Ranking consistency: Poor
- **Conclusion**: Weak agreement between ROAR and faithfulness metrics
