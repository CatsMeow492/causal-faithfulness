# ROAR-Faithfulness Validation Report

## Summary Statistics

### Random
- Pearson correlation: r = -0.2168 (p = 0.8609)
- Spearman correlation: ρ = 0.0000 (p = 1.0000)
- Statistically significant: No
- Sample size: 3

### Occlusion
- Pearson correlation: r = nan (p = nan)
- Spearman correlation: ρ = nan (p = nan)
- Statistically significant: No
- Sample size: 3

## Explainer Ranking Consistency

- ROAR ranking: ['Occlusion', 'Random']
- Faithfulness ranking: ['Occlusion', 'Random']
- Rank correlation: ρ = 1.0000 (p = nan)
- Rankings consistent: No

## Counter-Examples (Significant Disagreements)

### Random
Found 1 counter-examples:
- Example 1: ROAR = 0.0500, Faithfulness = 0.5586

### Occlusion
Found 3 counter-examples:
- Example 1: ROAR = 0.4375, Faithfulness = 1.0000
- Example 2: ROAR = 0.4375, Faithfulness = 1.0000
- Example 3: ROAR = 0.4375, Faithfulness = 1.0000

## Overall Assessment

- 0/2 explainers show significant correlation
- Ranking consistency: Poor
- **Conclusion**: Weak agreement between ROAR and faithfulness metrics
