# ROAR-Faithfulness Validation Report

## Summary Statistics

### Random
- Pearson correlation: r = -1.0000 (p = 1.0000)
- Spearman correlation: ρ = -1.0000 (p = nan)
- Statistically significant: No
- Sample size: 2

### Occlusion
- Pearson correlation: r = nan (p = nan)
- Spearman correlation: ρ = nan (p = nan)
- Statistically significant: No
- Sample size: 2

## Explainer Ranking Consistency

- ROAR ranking: ['Occlusion', 'Random']
- Faithfulness ranking: ['Occlusion', 'Random']
- Rank correlation: ρ = 1.0000 (p = nan)
- Rankings consistent: No

## Counter-Examples (Significant Disagreements)

### Random
Found 1 counter-examples:
- Example 1: ROAR = 0.0000, Faithfulness = 0.5657

### Occlusion
Found 2 counter-examples:
- Example 1: ROAR = 0.5000, Faithfulness = 1.0000
- Example 2: ROAR = 0.5000, Faithfulness = 1.0000

## Overall Assessment

- 0/2 explainers show significant correlation
- Ranking consistency: Poor
- **Conclusion**: Weak agreement between ROAR and faithfulness metrics
