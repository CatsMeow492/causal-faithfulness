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

- ROAR ranking: ['Occlusion', 'Random', 'IntegratedGradients']
- Faithfulness ranking: ['Occlusion', 'Random', 'IntegratedGradients']
- Rank correlation: ρ = 1.0000 (p = 0.0000)
- Rankings consistent: Yes

## Counter-Examples (Significant Disagreements)

### Random
Found 1 counter-examples:
- Example 1: ROAR = 0.0000, Faithfulness = 0.5623

### IntegratedGradients
Found 1 counter-examples:
- Example 1: ROAR = 0.0000, Faithfulness = 0.3190

### Occlusion
Found 2 counter-examples:
- Example 1: ROAR = 0.5000, Faithfulness = 1.0000
- Example 2: ROAR = 0.5000, Faithfulness = 1.0000

## Overall Assessment

- 0/3 explainers show significant correlation
- Ranking consistency: Good
- **Conclusion**: Weak agreement between ROAR and faithfulness metrics
