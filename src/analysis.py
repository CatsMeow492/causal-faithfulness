"""
Analysis and reporting system for causal-faithfulness metric experiments.
Provides statistical summary generation, automated finding identification, and correlation analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import warnings
from datetime import datetime
from scipy import stats
import json
import os

from .faithfulness import FaithfulnessResult
from .roar import ROARResult, ROARCorrelationResult
from .evaluation import ExperimentResult


@dataclass
class StatisticalSummary:
    """Statistical summary for a single explainer."""
    explainer_name: str
    n_samples: int
    mean_f_score: float
    std_f_score: float
    median_f_score: float
    q25_f_score: float
    q75_f_score: float
    min_f_score: float
    max_f_score: float
    confidence_interval: Tuple[float, float]
    significant_results: int
    total_results: int
    significance_rate: float
    mean_p_value: float
    effect_size: Optional[float] = None  # Cohen's d
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class FindingReport:
    """Automated finding identification results."""
    highest_faithfulness_explainer: str
    highest_faithfulness_score: float
    lowest_faithfulness_explainer: str
    lowest_faithfulness_score: float
    most_consistent_explainer: str
    most_consistent_std: float
    most_significant_explainer: str
    most_significant_rate: float
    performance_ranking: List[Tuple[str, float]]  # (explainer, score) sorted by performance
    statistical_significance: Dict[str, bool]  # explainer -> is significantly better than random
    effect_sizes: Dict[str, float]  # explainer -> Cohen's d
    key_insights: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class CorrelationAnalysis:
    """Correlation analysis between faithfulness and ROAR metrics."""
    overall_correlation: float
    overall_p_value: float
    explainer_correlations: Dict[str, Tuple[float, float]]  # explainer -> (r, p)
    strong_correlations: List[str]  # explainers with r > 0.5 and p < 0.05
    weak_correlations: List[str]  # explainers with r < 0.3 or p >= 0.05
    counter_examples: Dict[str, List[Tuple[float, float]]]  # explainer -> [(roar, faith)]
    correlation_consistency: float  # how consistent correlations are across explainers
    ranking_agreement: float  # Spearman correlation between rankings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class StatisticalAnalyzer:
    """
    Statistical analysis engine for faithfulness experiment results.
    Provides comprehensive statistical summaries and significance testing.
    """
    
    def __init__(self, confidence_level: float = 0.95, random_seed: int = 42):
        """Initialize analyzer with configuration."""
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
    
    def compute_explainer_summary(
        self, 
        results: List[FaithfulnessResult], 
        explainer_name: str
    ) -> StatisticalSummary:
        """
        Compute comprehensive statistical summary for a single explainer.
        
        Args:
            results: List of faithfulness results for the explainer
            explainer_name: Name of the explainer
            
        Returns:
            Statistical summary with all metrics
        """
        if not results:
            return StatisticalSummary(
                explainer_name=explainer_name,
                n_samples=0,
                mean_f_score=0.0,
                std_f_score=0.0,
                median_f_score=0.0,
                q25_f_score=0.0,
                q75_f_score=0.0,
                min_f_score=0.0,
                max_f_score=0.0,
                confidence_interval=(0.0, 0.0),
                significant_results=0,
                total_results=0,
                significance_rate=0.0,
                mean_p_value=1.0
            )
        
        # Extract F-scores and p-values
        f_scores = np.array([r.f_score for r in results])
        p_values = np.array([r.p_value for r in results])
        
        # Basic statistics
        n_samples = len(f_scores)
        mean_f = np.mean(f_scores)
        std_f = np.std(f_scores, ddof=1) if n_samples > 1 else 0.0
        median_f = np.median(f_scores)
        q25_f = np.percentile(f_scores, 25)
        q75_f = np.percentile(f_scores, 75)
        min_f = np.min(f_scores)
        max_f = np.max(f_scores)
        
        # Confidence interval for mean
        if n_samples > 1:
            sem = std_f / np.sqrt(n_samples)
            t_critical = stats.t.ppf((1 + self.confidence_level) / 2, n_samples - 1)
            margin_error = t_critical * sem
            ci_lower = mean_f - margin_error
            ci_upper = mean_f + margin_error
        else:
            ci_lower = ci_upper = mean_f
        
        # Significance analysis
        significant_results = np.sum(p_values < 0.05)
        significance_rate = significant_results / n_samples if n_samples > 0 else 0.0
        mean_p_value = np.mean(p_values)
        
        # Effect size (Cohen's d) - comparing against null hypothesis of F=0
        if std_f > 0:
            effect_size = mean_f / std_f
        else:
            effect_size = 0.0
        
        return StatisticalSummary(
            explainer_name=explainer_name,
            n_samples=n_samples,
            mean_f_score=mean_f,
            std_f_score=std_f,
            median_f_score=median_f,
            q25_f_score=q25_f,
            q75_f_score=q75_f,
            min_f_score=min_f,
            max_f_score=max_f,
            confidence_interval=(ci_lower, ci_upper),
            significant_results=significant_results,
            total_results=n_samples,
            significance_rate=significance_rate,
            mean_p_value=mean_p_value,
            effect_size=effect_size
        )
    
    def compare_explainers(
        self,
        results_dict: Dict[str, List[FaithfulnessResult]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform pairwise statistical comparisons between explainers.
        
        Args:
            results_dict: Dictionary mapping explainer names to results
            
        Returns:
            Dictionary of pairwise comparison results
        """
        explainer_names = list(results_dict.keys())
        comparisons = {}
        
        for i, explainer1 in enumerate(explainer_names):
            for j, explainer2 in enumerate(explainer_names[i+1:], i+1):
                results1 = results_dict[explainer1]
                results2 = results_dict[explainer2]
                
                if not results1 or not results2:
                    continue
                
                # Extract F-scores
                scores1 = np.array([r.f_score for r in results1])
                scores2 = np.array([r.f_score for r in results2])
                
                # Perform statistical tests
                comparison_key = f"{explainer1}_vs_{explainer2}"
                
                try:
                    # Independent t-test
                    t_stat, t_p = stats.ttest_ind(scores1, scores2)
                    
                    # Mann-Whitney U test (non-parametric)
                    u_stat, u_p = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                                         (len(scores2) - 1) * np.var(scores2, ddof=1)) / 
                                        (len(scores1) + len(scores2) - 2))
                    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std > 0 else 0.0
                    
                    # Determine winner
                    winner = explainer1 if np.mean(scores1) > np.mean(scores2) else explainer2
                    
                    comparisons[comparison_key] = {
                        'explainer1': explainer1,
                        'explainer2': explainer2,
                        'mean1': np.mean(scores1),
                        'mean2': np.mean(scores2),
                        'std1': np.std(scores1, ddof=1),
                        'std2': np.std(scores2, ddof=1),
                        't_statistic': t_stat,
                        't_p_value': t_p,
                        'u_statistic': u_stat,
                        'u_p_value': u_p,
                        'cohens_d': cohens_d,
                        'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d)),
                        'winner': winner,
                        'significant_difference': min(t_p, u_p) < 0.05,
                        'practical_significance': abs(cohens_d) > 0.5
                    }
                    
                except Exception as e:
                    warnings.warn(f"Comparison failed for {comparison_key}: {e}")
                    continue
        
        return comparisons
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"


class FindingIdentifier:
    """
    Automated finding identification system.
    Analyzes results to identify key insights and patterns.
    """
    
    def __init__(self, significance_threshold: float = 0.05):
        """Initialize with significance threshold."""
        self.significance_threshold = significance_threshold
    
    def identify_findings(
        self,
        experiment_result: ExperimentResult,
        statistical_summaries: Dict[str, StatisticalSummary]
    ) -> FindingReport:
        """
        Identify key findings from experiment results.
        
        Args:
            experiment_result: Complete experiment results
            statistical_summaries: Statistical summaries by explainer
            
        Returns:
            Comprehensive finding report
        """
        if not statistical_summaries:
            return self._empty_finding_report()
        
        # Find highest and lowest performing explainers
        explainer_scores = {name: summary.mean_f_score 
                           for name, summary in statistical_summaries.items()}
        
        highest_explainer = max(explainer_scores.keys(), key=lambda x: explainer_scores[x])
        lowest_explainer = min(explainer_scores.keys(), key=lambda x: explainer_scores[x])
        
        # Find most consistent explainer (lowest std)
        explainer_stds = {name: summary.std_f_score 
                         for name, summary in statistical_summaries.items()}
        most_consistent = min(explainer_stds.keys(), key=lambda x: explainer_stds[x])
        
        # Find most significant explainer (highest significance rate)
        explainer_sig_rates = {name: summary.significance_rate 
                              for name, summary in statistical_summaries.items()}
        most_significant = max(explainer_sig_rates.keys(), key=lambda x: explainer_sig_rates[x])
        
        # Create performance ranking
        performance_ranking = sorted(explainer_scores.items(), 
                                   key=lambda x: x[1], reverse=True)
        
        # Determine statistical significance vs random baseline
        statistical_significance = {}
        effect_sizes = {}
        
        for name, summary in statistical_summaries.items():
            # Consider significant if mean F-score is significantly > 0 with good effect size
            is_significant = (summary.mean_f_score > 0.1 and 
                            summary.significance_rate > 0.5 and
                            summary.effect_size and summary.effect_size > 0.3)
            statistical_significance[name] = is_significant
            effect_sizes[name] = summary.effect_size or 0.0
        
        # Generate key insights
        key_insights = self._generate_insights(
            statistical_summaries, performance_ranking, 
            statistical_significance, effect_sizes
        )
        
        return FindingReport(
            highest_faithfulness_explainer=highest_explainer,
            highest_faithfulness_score=explainer_scores[highest_explainer],
            lowest_faithfulness_explainer=lowest_explainer,
            lowest_faithfulness_score=explainer_scores[lowest_explainer],
            most_consistent_explainer=most_consistent,
            most_consistent_std=explainer_stds[most_consistent],
            most_significant_explainer=most_significant,
            most_significant_rate=explainer_sig_rates[most_significant],
            performance_ranking=performance_ranking,
            statistical_significance=statistical_significance,
            effect_sizes=effect_sizes,
            key_insights=key_insights
        )
    
    def _empty_finding_report(self) -> FindingReport:
        """Create empty finding report for edge cases."""
        return FindingReport(
            highest_faithfulness_explainer="",
            highest_faithfulness_score=0.0,
            lowest_faithfulness_explainer="",
            lowest_faithfulness_score=0.0,
            most_consistent_explainer="",
            most_consistent_std=0.0,
            most_significant_explainer="",
            most_significant_rate=0.0,
            performance_ranking=[],
            statistical_significance={},
            effect_sizes={},
            key_insights=["No valid results found for analysis."]
        )
    
    def _generate_insights(
        self,
        summaries: Dict[str, StatisticalSummary],
        ranking: List[Tuple[str, float]],
        significance: Dict[str, bool],
        effect_sizes: Dict[str, float]
    ) -> List[str]:
        """Generate automated insights from analysis."""
        insights = []
        
        if not ranking:
            return ["No explainers to analyze."]
        
        # Performance insights
        best_explainer, best_score = ranking[0]
        worst_explainer, worst_score = ranking[-1]
        
        insights.append(f"{best_explainer} achieved the highest faithfulness score ({best_score:.4f})")
        
        if len(ranking) > 1:
            score_gap = best_score - worst_score
            insights.append(f"Performance gap between best and worst explainers: {score_gap:.4f}")
        
        # Significance insights
        significant_explainers = [name for name, is_sig in significance.items() if is_sig]
        if significant_explainers:
            insights.append(f"{len(significant_explainers)}/{len(summaries)} explainers show statistically significant faithfulness")
            if len(significant_explainers) == 1:
                insights.append(f"Only {significant_explainers[0]} demonstrates reliable faithfulness")
        else:
            insights.append("No explainers show strong statistical significance")
        
        # Effect size insights
        large_effects = [name for name, effect in effect_sizes.items() if effect > 0.8]
        if large_effects:
            insights.append(f"Large effect sizes observed for: {', '.join(large_effects)}")
        
        # Consistency insights
        std_values = [summary.std_f_score for summary in summaries.values()]
        if std_values:
            avg_std = np.mean(std_values)
            if avg_std < 0.1:
                insights.append("All explainers show high consistency (low variance)")
            elif avg_std > 0.3:
                insights.append("High variance observed across explainers - results may be unstable")
        
        # Practical insights
        high_performers = [name for name, score in ranking if score > 0.7]
        if high_performers:
            insights.append(f"High-performing explainers (F > 0.7): {', '.join(high_performers)}")
        
        poor_performers = [name for name, score in ranking if score < 0.3]
        if poor_performers:
            insights.append(f"Poor-performing explainers (F < 0.3): {', '.join(poor_performers)}")
        
        return insights


class CorrelationAnalyzer:
    """
    Correlation analysis between faithfulness and ROAR metrics.
    Detects counter-examples and analyzes agreement between metrics.
    """
    
    def __init__(self, strong_correlation_threshold: float = 0.5):
        """Initialize with correlation thresholds."""
        self.strong_threshold = strong_correlation_threshold
    
    def analyze_correlations(
        self,
        correlation_results: List[ROARCorrelationResult]
    ) -> CorrelationAnalysis:
        """
        Analyze correlations between faithfulness and ROAR metrics.
        
        Args:
            correlation_results: List of correlation results by explainer
            
        Returns:
            Comprehensive correlation analysis
        """
        if not correlation_results:
            return self._empty_correlation_analysis()
        
        # Extract correlation data
        explainer_correlations = {}
        all_correlations = []
        all_p_values = []
        
        for result in correlation_results:
            r = result.pearson_r
            p = result.p_value
            explainer_correlations[result.explainer_name] = (r, p)
            all_correlations.append(r)
            all_p_values.append(p)
        
        # Overall correlation (average)
        overall_correlation = np.mean(all_correlations)
        overall_p_value = np.mean(all_p_values)  # Simplified - could use meta-analysis
        
        # Categorize correlations
        strong_correlations = []
        weak_correlations = []
        
        for explainer, (r, p) in explainer_correlations.items():
            if abs(r) >= self.strong_threshold and p < 0.05:
                strong_correlations.append(explainer)
            else:
                weak_correlations.append(explainer)
        
        # Detect counter-examples
        counter_examples = self._detect_counter_examples(correlation_results)
        
        # Correlation consistency
        correlation_consistency = 1.0 - np.std(all_correlations) if len(all_correlations) > 1 else 1.0
        
        # Ranking agreement
        ranking_agreement = self._compute_ranking_agreement(correlation_results)
        
        return CorrelationAnalysis(
            overall_correlation=overall_correlation,
            overall_p_value=overall_p_value,
            explainer_correlations=explainer_correlations,
            strong_correlations=strong_correlations,
            weak_correlations=weak_correlations,
            counter_examples=counter_examples,
            correlation_consistency=correlation_consistency,
            ranking_agreement=ranking_agreement
        )
    
    def _empty_correlation_analysis(self) -> CorrelationAnalysis:
        """Create empty correlation analysis for edge cases."""
        return CorrelationAnalysis(
            overall_correlation=0.0,
            overall_p_value=1.0,
            explainer_correlations={},
            strong_correlations=[],
            weak_correlations=[],
            counter_examples={},
            correlation_consistency=0.0,
            ranking_agreement=0.0
        )
    
    def _detect_counter_examples(
        self,
        correlation_results: List[ROARCorrelationResult],
        disagreement_threshold: float = 0.4
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Detect cases where ROAR and faithfulness strongly disagree."""
        counter_examples = {}
        
        for result in correlation_results:
            explainer_counter_examples = []
            
            # Look for points where metrics disagree significantly
            for roar_score, faith_score in zip(result.roar_scores, result.faithfulness_scores):
                # Normalize ROAR score to [0, 1] for comparison
                # Higher ROAR drop should correspond to higher faithfulness
                roar_normalized = min(1.0, max(0.0, roar_score))
                
                # Check for disagreement
                disagreement = abs(roar_normalized - faith_score)
                if disagreement > disagreement_threshold:
                    explainer_counter_examples.append((roar_score, faith_score))
            
            if explainer_counter_examples:
                counter_examples[result.explainer_name] = explainer_counter_examples
        
        return counter_examples
    
    def _compute_ranking_agreement(
        self,
        correlation_results: List[ROARCorrelationResult]
    ) -> float:
        """Compute agreement between ROAR and faithfulness rankings."""
        if len(correlation_results) < 2:
            return 1.0
        
        # Get average scores for each explainer
        roar_averages = {}
        faith_averages = {}
        
        for result in correlation_results:
            roar_averages[result.explainer_name] = np.mean(result.roar_scores)
            faith_averages[result.explainer_name] = np.mean(result.faithfulness_scores)
        
        # Create rankings
        explainers = list(roar_averages.keys())
        roar_ranking = sorted(explainers, key=lambda x: roar_averages[x], reverse=True)
        faith_ranking = sorted(explainers, key=lambda x: faith_averages[x], reverse=True)
        
        # Convert to ranks for correlation
        roar_ranks = {explainer: i for i, explainer in enumerate(roar_ranking)}
        faith_ranks = {explainer: i for i, explainer in enumerate(faith_ranking)}
        
        # Compute Spearman correlation between rankings
        roar_rank_values = [roar_ranks[explainer] for explainer in explainers]
        faith_rank_values = [faith_ranks[explainer] for explainer in explainers]
        
        try:
            correlation, _ = stats.spearmanr(roar_rank_values, faith_rank_values)
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0


class ComprehensiveAnalyzer:
    """
    Main analysis engine that combines all analysis components.
    Provides unified interface for complete result analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with configuration."""
        config = config or {}
        
        self.statistical_analyzer = StatisticalAnalyzer(
            confidence_level=config.get('confidence_level', 0.95),
            random_seed=config.get('random_seed', 42)
        )
        
        self.finding_identifier = FindingIdentifier(
            significance_threshold=config.get('significance_threshold', 0.05)
        )
        
        self.correlation_analyzer = CorrelationAnalyzer(
            strong_correlation_threshold=config.get('strong_correlation_threshold', 0.5)
        )
    
    def analyze_experiment(
        self,
        experiment_result: ExperimentResult,
        correlation_results: Optional[List[ROARCorrelationResult]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of experiment results.
        
        Args:
            experiment_result: Complete experiment results
            correlation_results: Optional ROAR correlation results
            
        Returns:
            Dictionary containing all analysis results
        """
        analysis_results = {
            'experiment_info': {
                'name': experiment_result.experiment_config.experiment_name,
                'dataset': experiment_result.dataset_info['name'],
                'model': experiment_result.model_info['name'],
                'timestamp': experiment_result.timestamp,
                'total_runtime': experiment_result.total_runtime
            }
        }
        
        # Statistical analysis
        statistical_summaries = {}
        for explainer_name, results in experiment_result.explainer_results.items():
            summary = self.statistical_analyzer.compute_explainer_summary(results, explainer_name)
            statistical_summaries[explainer_name] = summary
        
        analysis_results['statistical_summaries'] = {
            name: summary.to_dict() for name, summary in statistical_summaries.items()
        }
        
        # Pairwise comparisons
        comparisons = self.statistical_analyzer.compare_explainers(
            experiment_result.explainer_results
        )
        analysis_results['pairwise_comparisons'] = comparisons
        
        # Finding identification
        findings = self.finding_identifier.identify_findings(
            experiment_result, statistical_summaries
        )
        analysis_results['findings'] = findings.to_dict()
        
        # Correlation analysis (if available)
        if correlation_results:
            correlation_analysis = self.correlation_analyzer.analyze_correlations(
                correlation_results
            )
            analysis_results['correlation_analysis'] = correlation_analysis.to_dict()
        
        return analysis_results
    
    def generate_report(
        self,
        analysis_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive analysis report.
        
        Args:
            analysis_results: Results from analyze_experiment
            output_path: Optional path to save report
            
        Returns:
            Formatted report as string
        """
        report = self._format_analysis_report(analysis_results)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report
    
    def _format_analysis_report(self, analysis_results: Dict[str, Any]) -> str:
        """Format analysis results into readable report."""
        report = "# Comprehensive Faithfulness Analysis Report\n\n"
        
        # Experiment info
        exp_info = analysis_results['experiment_info']
        report += f"**Experiment**: {exp_info['name']}\n"
        report += f"**Dataset**: {exp_info['dataset']}\n"
        report += f"**Model**: {exp_info['model']}\n"
        report += f"**Generated**: {exp_info['timestamp']}\n"
        report += f"**Runtime**: {exp_info['total_runtime']:.2f} seconds\n\n"
        
        # Key findings
        if 'findings' in analysis_results:
            findings = analysis_results['findings']
            report += "## Key Findings\n\n"
            
            report += f"- **Best Performer**: {findings['highest_faithfulness_explainer']} "
            report += f"(F = {findings['highest_faithfulness_score']:.4f})\n"
            
            report += f"- **Most Consistent**: {findings['most_consistent_explainer']} "
            report += f"(σ = {findings['most_consistent_std']:.4f})\n"
            
            report += f"- **Most Significant**: {findings['most_significant_explainer']} "
            report += f"({findings['most_significant_rate']:.1%} significant results)\n\n"
            
            report += "### Performance Ranking\n"
            for i, (explainer, score) in enumerate(findings['performance_ranking'], 1):
                report += f"{i}. {explainer}: {score:.4f}\n"
            report += "\n"
            
            report += "### Key Insights\n"
            for insight in findings['key_insights']:
                report += f"- {insight}\n"
            report += "\n"
        
        # Statistical summaries
        if 'statistical_summaries' in analysis_results:
            report += "## Statistical Summaries\n\n"
            
            for explainer, summary in analysis_results['statistical_summaries'].items():
                report += f"### {explainer}\n"
                report += f"- Mean F-score: {summary['mean_f_score']:.4f} ± {summary['std_f_score']:.4f}\n"
                report += f"- 95% CI: [{summary['confidence_interval'][0]:.4f}, {summary['confidence_interval'][1]:.4f}]\n"
                report += f"- Median: {summary['median_f_score']:.4f}\n"
                report += f"- Range: [{summary['min_f_score']:.4f}, {summary['max_f_score']:.4f}]\n"
                report += f"- Significant results: {summary['significant_results']}/{summary['total_results']} ({summary['significance_rate']:.1%})\n"
                if summary['effect_size']:
                    report += f"- Effect size (Cohen's d): {summary['effect_size']:.3f}\n"
                report += "\n"
        
        # Correlation analysis
        if 'correlation_analysis' in analysis_results:
            corr_analysis = analysis_results['correlation_analysis']
            report += "## ROAR-Faithfulness Correlation Analysis\n\n"
            
            report += f"- **Overall correlation**: r = {corr_analysis['overall_correlation']:.4f} "
            report += f"(p = {corr_analysis['overall_p_value']:.4f})\n"
            
            report += f"- **Strong correlations**: {', '.join(corr_analysis['strong_correlations']) or 'None'}\n"
            report += f"- **Weak correlations**: {', '.join(corr_analysis['weak_correlations']) or 'None'}\n"
            
            report += f"- **Correlation consistency**: {corr_analysis['correlation_consistency']:.3f}\n"
            report += f"- **Ranking agreement**: {corr_analysis['ranking_agreement']:.3f}\n"
            
            if corr_analysis['counter_examples']:
                report += "\n### Counter-Examples Detected\n"
                for explainer, examples in corr_analysis['counter_examples'].items():
                    report += f"- {explainer}: {len(examples)} disagreements\n"
            report += "\n"
        
        # Pairwise comparisons
        if 'pairwise_comparisons' in analysis_results:
            report += "## Pairwise Comparisons\n\n"
            
            significant_comparisons = [
                comp for comp in analysis_results['pairwise_comparisons'].values()
                if comp['significant_difference']
            ]
            
            if significant_comparisons:
                report += "### Significant Differences\n"
                for comp in significant_comparisons:
                    winner = comp['winner']
                    loser = comp['explainer1'] if winner == comp['explainer2'] else comp['explainer2']
                    report += f"- {winner} > {loser} "
                    report += f"(p = {min(comp['t_p_value'], comp['u_p_value']):.4f}, "
                    report += f"d = {comp['cohens_d']:.3f})\n"
            else:
                report += "No statistically significant differences found between explainers.\n"
            report += "\n"
        
        report += "---\n"
        report += f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return report


# Convenience functions
def analyze_experiment_results(
    experiment_result: ExperimentResult,
    correlation_results: Optional[List[ROARCorrelationResult]] = None,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Convenience function for complete experiment analysis.
    
    Args:
        experiment_result: Complete experiment results
        correlation_results: Optional ROAR correlation results
        output_dir: Output directory for reports
        
    Returns:
        Complete analysis results
    """
    analyzer = ComprehensiveAnalyzer()
    analysis_results = analyzer.analyze_experiment(experiment_result, correlation_results)
    
    # Save analysis results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_path = os.path.join(output_dir, f"analysis_{experiment_result.experiment_config.experiment_name}_{timestamp}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    # Save formatted report
    report_path = os.path.join(output_dir, f"report_{experiment_result.experiment_config.experiment_name}_{timestamp}.md")
    analyzer.generate_report(analysis_results, report_path)
    
    return analysis_results


def identify_best_explainers(
    experiment_result: ExperimentResult,
    criteria: str = "mean_f_score"
) -> List[Tuple[str, float]]:
    """
    Identify best explainers based on specified criteria.
    
    Args:
        experiment_result: Complete experiment results
        criteria: Criteria for ranking ("mean_f_score", "consistency", "significance")
        
    Returns:
        List of (explainer_name, score) tuples sorted by performance
    """
    analyzer = StatisticalAnalyzer()
    
    explainer_scores = []
    
    for explainer_name, results in experiment_result.explainer_results.items():
        summary = analyzer.compute_explainer_summary(results, explainer_name)
        
        if criteria == "mean_f_score":
            score = summary.mean_f_score
        elif criteria == "consistency":
            score = -summary.std_f_score  # Lower std is better
        elif criteria == "significance":
            score = summary.significance_rate
        else:
            score = summary.mean_f_score  # Default
        
        explainer_scores.append((explainer_name, score))
    
    return sorted(explainer_scores, key=lambda x: x[1], reverse=True)