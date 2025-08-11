"""
Statistical analysis and validation functions for causal-faithfulness evaluation.
Implements paired t-tests, bootstrap confidence intervals, and sanity checks.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import warnings
from scipy import stats
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .faithfulness import FaithfulnessResult, FaithfulnessConfig
from .roar import ROARResult, ROARCorrelationResult
from .explainers import ExplainerWrapper, Attribution, RandomExplainer


@dataclass
class StatisticalTestResult:
    """Result from statistical significance testing."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """String representation of test result."""
        significance = "significant" if self.is_significant else "not significant"
        return (f"{self.test_name}: statistic={self.statistic:.4f}, "
                f"p={self.p_value:.4f} ({significance})")


@dataclass
class BootstrapResult:
    """Result from bootstrap analysis."""
    original_statistic: float
    bootstrap_mean: float
    bootstrap_std: float
    confidence_interval: Tuple[float, float]
    n_bootstrap_samples: int
    confidence_level: float
    
    def __str__(self) -> str:
        """String representation of bootstrap result."""
        return (f"Bootstrap: mean={self.bootstrap_mean:.4f} ± {self.bootstrap_std:.4f}, "
                f"CI=[{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]")


@dataclass
class SanityCheckResult:
    """Result from sanity check validation."""
    check_name: str
    passed: bool
    score: float
    threshold: float
    message: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """String representation of sanity check result."""
        status = "PASSED" if self.passed else "FAILED"
        return f"{self.check_name}: {status} (score={self.score:.4f}, threshold={self.threshold:.4f})"


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for faithfulness evaluation.
    Provides significance testing, bootstrap analysis, and validation.
    """
    
    def __init__(self, alpha: float = 0.05, random_seed: int = 42):
        """
        Initialize statistical analyzer.
        
        Args:
            alpha: Significance level for hypothesis testing
            random_seed: Random seed for reproducibility
        """
        self.alpha = alpha
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
    def paired_t_test(
        self,
        group1_scores: List[float],
        group2_scores: List[float],
        alternative: str = "two-sided"
    ) -> StatisticalTestResult:
        """
        Perform paired t-test between two groups of scores.
        
        Args:
            group1_scores: First group of scores
            group2_scores: Second group of scores
            alternative: Alternative hypothesis ("two-sided", "less", "greater")
            
        Returns:
            Statistical test result
        """
        if len(group1_scores) != len(group2_scores):
            raise ValueError("Groups must have equal length for paired t-test")
        
        if len(group1_scores) < 2:
            warnings.warn("Insufficient samples for t-test")
            return StatisticalTestResult(
                test_name="Paired t-test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                metadata={'error': 'insufficient_samples'}
            )
        
        try:
            # Convert to numpy arrays
            group1 = np.array(group1_scores)
            group2 = np.array(group2_scores)
            
            # Perform paired t-test
            statistic, p_value = stats.ttest_rel(group1, group2, alternative=alternative)
            
            # Compute effect size (Cohen's d for paired samples)
            differences = group1 - group2
            effect_size = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences) > 0 else 0.0
            
            # Compute confidence interval for the difference
            n = len(differences)
            se = stats.sem(differences)
            t_critical = stats.t.ppf(1 - self.alpha/2, n - 1)
            ci_lower = np.mean(differences) - t_critical * se
            ci_upper = np.mean(differences) + t_critical * se
            
            return StatisticalTestResult(
                test_name="Paired t-test",
                statistic=statistic,
                p_value=p_value,
                is_significant=p_value < self.alpha,
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                metadata={
                    'alternative': alternative,
                    'n_samples': n,
                    'mean_difference': np.mean(differences),
                    'std_difference': np.std(differences, ddof=1)
                }
            )
            
        except Exception as e:
            warnings.warn(f"Paired t-test failed: {e}")
            return StatisticalTestResult(
                test_name="Paired t-test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                metadata={'error': str(e)}
            )
    
    def bootstrap_confidence_interval(
        self,
        data: List[float],
        statistic_func: Callable[[np.ndarray], float] = np.mean,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95
    ) -> BootstrapResult:
        """
        Compute bootstrap confidence interval for a statistic.
        
        Args:
            data: Data to bootstrap
            statistic_func: Function to compute statistic
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for interval
            
        Returns:
            Bootstrap analysis result
        """
        if len(data) < 2:
            warnings.warn("Insufficient data for bootstrap")
            return BootstrapResult(
                original_statistic=0.0,
                bootstrap_mean=0.0,
                bootstrap_std=0.0,
                confidence_interval=(0.0, 0.0),
                n_bootstrap_samples=0,
                confidence_level=confidence_level
            )
        
        try:
            data_array = np.array(data)
            original_statistic = statistic_func(data_array)
            
            # Perform bootstrap resampling
            bootstrap_statistics = []
            for _ in range(n_bootstrap):
                # Resample with replacement
                bootstrap_sample = self.rng.choice(data_array, size=len(data_array), replace=True)
                bootstrap_stat = statistic_func(bootstrap_sample)
                bootstrap_statistics.append(bootstrap_stat)
            
            bootstrap_statistics = np.array(bootstrap_statistics)
            
            # Compute confidence interval
            alpha = 1 - confidence_level
            ci_lower = np.percentile(bootstrap_statistics, (alpha/2) * 100)
            ci_upper = np.percentile(bootstrap_statistics, (1 - alpha/2) * 100)
            
            return BootstrapResult(
                original_statistic=original_statistic,
                bootstrap_mean=np.mean(bootstrap_statistics),
                bootstrap_std=np.std(bootstrap_statistics),
                confidence_interval=(ci_lower, ci_upper),
                n_bootstrap_samples=n_bootstrap,
                confidence_level=confidence_level
            )
            
        except Exception as e:
            warnings.warn(f"Bootstrap analysis failed: {e}")
            return BootstrapResult(
                original_statistic=0.0,
                bootstrap_mean=0.0,
                bootstrap_std=0.0,
                confidence_interval=(0.0, 0.0),
                n_bootstrap_samples=0,
                confidence_level=confidence_level
            )
    
    def wilcoxon_signed_rank_test(
        self,
        group1_scores: List[float],
        group2_scores: List[float],
        alternative: str = "two-sided"
    ) -> StatisticalTestResult:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
        
        Args:
            group1_scores: First group of scores
            group2_scores: Second group of scores
            alternative: Alternative hypothesis
            
        Returns:
            Statistical test result
        """
        if len(group1_scores) != len(group2_scores):
            raise ValueError("Groups must have equal length for Wilcoxon test")
        
        if len(group1_scores) < 3:
            warnings.warn("Insufficient samples for Wilcoxon test")
            return StatisticalTestResult(
                test_name="Wilcoxon signed-rank test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                metadata={'error': 'insufficient_samples'}
            )
        
        try:
            group1 = np.array(group1_scores)
            group2 = np.array(group2_scores)
            
            # Perform Wilcoxon signed-rank test
            statistic, p_value = stats.wilcoxon(group1, group2, alternative=alternative)
            
            return StatisticalTestResult(
                test_name="Wilcoxon signed-rank test",
                statistic=statistic,
                p_value=p_value,
                is_significant=p_value < self.alpha,
                metadata={
                    'alternative': alternative,
                    'n_samples': len(group1)
                }
            )
            
        except Exception as e:
            warnings.warn(f"Wilcoxon test failed: {e}")
            return StatisticalTestResult(
                test_name="Wilcoxon signed-rank test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                metadata={'error': str(e)}
            )
    
    def multiple_comparisons_correction(
        self,
        p_values: List[float],
        method: str = "bonferroni"
    ) -> Tuple[List[bool], List[float]]:
        """
        Apply multiple comparisons correction to p-values.
        
        Args:
            p_values: List of p-values to correct
            method: Correction method ("bonferroni", "holm", "fdr_bh")
            
        Returns:
            Tuple of (rejected hypotheses, corrected p-values)
        """
        # Prefer statsmodels if available
        try:
            from statsmodels.stats.multitest import multipletests  # type: ignore
            method_map = {
                "bonferroni": 'bonferroni',
                "holm": 'holm',
                "fdr_bh": 'fdr_bh',
            }
            m = method_map.get(method, 'bonferroni')
            rejected, corrected_p, _, _ = multipletests(p_values, alpha=self.alpha, method=m)
            return rejected.tolist(), corrected_p.tolist()
        except Exception as e:
            # Fallback: implement simple Bonferroni manually
            warnings.warn(f"Multiple comparisons correction fallback (manual Bonferroni): {e}")
            m = len(p_values) if p_values else 1
            corrected = [min(1.0, float(p) * m) for p in p_values]
            rejected = [cp < self.alpha for cp in corrected]
            return rejected, corrected


class SanityChecker:
    """
    Sanity check validator for explanation methods and faithfulness metrics.
    Implements various validation tests to ensure metric reliability.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize sanity checker with random seed."""
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
    
    def random_explainer_validation(
        self,
        random_explainer_results: List[FaithfulnessResult],
        informed_explainer_results: List[FaithfulnessResult],
        threshold_difference: float = 0.1
    ) -> SanityCheckResult:
        """
        Validate that random explainer produces lower faithfulness scores than informed explainers.
        
        Args:
            random_explainer_results: Results from random explainer
            informed_explainer_results: Results from informed explainers
            threshold_difference: Minimum expected difference
            
        Returns:
            Sanity check result
        """
        if not random_explainer_results or not informed_explainer_results:
            return SanityCheckResult(
                check_name="Random explainer validation",
                passed=False,
                score=0.0,
                threshold=threshold_difference,
                message="Insufficient results for validation",
                metadata={'error': 'insufficient_data'}
            )
        
        try:
            # Compute average scores
            random_avg = np.mean([r.f_score for r in random_explainer_results])
            informed_avg = np.mean([r.f_score for r in informed_explainer_results])
            
            # Check if informed explainers perform better
            difference = informed_avg - random_avg
            passed = difference >= threshold_difference
            
            message = (f"Random explainer avg: {random_avg:.4f}, "
                      f"Informed explainer avg: {informed_avg:.4f}, "
                      f"Difference: {difference:.4f}")
            
            return SanityCheckResult(
                check_name="Random explainer validation",
                passed=passed,
                score=difference,
                threshold=threshold_difference,
                message=message,
                metadata={
                    'random_avg': random_avg,
                    'informed_avg': informed_avg,
                    'n_random_results': len(random_explainer_results),
                    'n_informed_results': len(informed_explainer_results)
                }
            )
            
        except Exception as e:
            return SanityCheckResult(
                check_name="Random explainer validation",
                passed=False,
                score=0.0,
                threshold=threshold_difference,
                message=f"Validation failed: {str(e)}",
                metadata={'error': str(e)}
            )
    
    def metric_bounds_validation(
        self,
        faithfulness_results: List[FaithfulnessResult]
    ) -> SanityCheckResult:
        """
        Validate that faithfulness scores are within expected bounds [0, 1].
        
        Args:
            faithfulness_results: Faithfulness results to validate
            
        Returns:
            Sanity check result
        """
        if not faithfulness_results:
            return SanityCheckResult(
                check_name="Metric bounds validation",
                passed=False,
                score=0.0,
                threshold=1.0,
                message="No results to validate"
            )
        
        try:
            scores = [r.f_score for r in faithfulness_results]
            
            # Check bounds
            min_score = min(scores)
            max_score = max(scores)
            
            within_bounds = min_score >= 0.0 and max_score <= 1.0
            
            # Compute fraction of scores within bounds
            valid_scores = sum(1 for s in scores if 0.0 <= s <= 1.0)
            fraction_valid = valid_scores / len(scores)
            
            message = (f"Score range: [{min_score:.4f}, {max_score:.4f}], "
                      f"Valid fraction: {fraction_valid:.4f}")
            
            return SanityCheckResult(
                check_name="Metric bounds validation",
                passed=within_bounds and fraction_valid >= 0.95,
                score=fraction_valid,
                threshold=0.95,
                message=message,
                metadata={
                    'min_score': min_score,
                    'max_score': max_score,
                    'n_valid_scores': valid_scores,
                    'n_total_scores': len(scores)
                }
            )
            
        except Exception as e:
            return SanityCheckResult(
                check_name="Metric bounds validation",
                passed=False,
                score=0.0,
                threshold=0.95,
                message=f"Validation failed: {str(e)}",
                metadata={'error': str(e)}
            )
    
    def reproducibility_validation(
        self,
        results1: List[FaithfulnessResult],
        results2: List[FaithfulnessResult],
        tolerance: float = 1e-6
    ) -> SanityCheckResult:
        """
        Validate reproducibility by comparing results from identical runs.
        
        Args:
            results1: First set of results
            results2: Second set of results (should be identical)
            tolerance: Numerical tolerance for comparison
            
        Returns:
            Sanity check result
        """
        if len(results1) != len(results2):
            return SanityCheckResult(
                check_name="Reproducibility validation",
                passed=False,
                score=0.0,
                threshold=1.0,
                message=f"Result lengths differ: {len(results1)} vs {len(results2)}"
            )
        
        if not results1:
            return SanityCheckResult(
                check_name="Reproducibility validation",
                passed=True,
                score=1.0,
                threshold=1.0,
                message="No results to compare (trivially reproducible)"
            )
        
        try:
            scores1 = [r.f_score for r in results1]
            scores2 = [r.f_score for r in results2]
            
            # Compute maximum absolute difference
            max_diff = max(abs(s1 - s2) for s1, s2 in zip(scores1, scores2))
            
            # Check if all differences are within tolerance
            reproducible = max_diff <= tolerance
            
            # Compute fraction of reproducible results
            reproducible_count = sum(1 for s1, s2 in zip(scores1, scores2) 
                                   if abs(s1 - s2) <= tolerance)
            fraction_reproducible = reproducible_count / len(scores1)
            
            message = (f"Max difference: {max_diff:.8f}, "
                      f"Reproducible fraction: {fraction_reproducible:.4f}")
            
            return SanityCheckResult(
                check_name="Reproducibility validation",
                passed=reproducible,
                score=fraction_reproducible,
                threshold=1.0,
                message=message,
                metadata={
                    'max_difference': max_diff,
                    'tolerance': tolerance,
                    'n_reproducible': reproducible_count,
                    'n_total': len(scores1)
                }
            )
            
        except Exception as e:
            return SanityCheckResult(
                check_name="Reproducibility validation",
                passed=False,
                score=0.0,
                threshold=1.0,
                message=f"Validation failed: {str(e)}",
                metadata={'error': str(e)}
            )
    
    def monotonicity_validation(
        self,
        faithfulness_results: List[FaithfulnessResult],
        feature_importance_rankings: List[List[int]]
    ) -> SanityCheckResult:
        """
        Validate monotonicity property: removing more important features should decrease faithfulness.
        
        Args:
            faithfulness_results: Faithfulness results for different feature removal levels
            feature_importance_rankings: Rankings of features by importance
            
        Returns:
            Sanity check result
        """
        # This is a simplified monotonicity check
        # In practice, this would require more sophisticated analysis
        
        if len(faithfulness_results) < 2:
            return SanityCheckResult(
                check_name="Monotonicity validation",
                passed=False,
                score=0.0,
                threshold=0.8,
                message="Insufficient results for monotonicity check"
            )
        
        try:
            scores = [r.f_score for r in faithfulness_results]
            
            # Check if scores generally decrease (allowing some noise)
            monotonic_pairs = 0
            total_pairs = 0
            
            for i in range(len(scores) - 1):
                for j in range(i + 1, len(scores)):
                    if scores[i] >= scores[j]:  # Should be decreasing
                        monotonic_pairs += 1
                    total_pairs += 1
            
            monotonicity_score = monotonic_pairs / total_pairs if total_pairs > 0 else 0.0
            
            return SanityCheckResult(
                check_name="Monotonicity validation",
                passed=monotonicity_score >= 0.8,
                score=monotonicity_score,
                threshold=0.8,
                message=f"Monotonic pairs: {monotonic_pairs}/{total_pairs}",
                metadata={
                    'monotonic_pairs': monotonic_pairs,
                    'total_pairs': total_pairs,
                    'scores': scores
                }
            )
            
        except Exception as e:
            return SanityCheckResult(
                check_name="Monotonicity validation",
                passed=False,
                score=0.0,
                threshold=0.8,
                message=f"Validation failed: {str(e)}",
                metadata={'error': str(e)}
            )


class ValidationSuite:
    """
    Comprehensive validation suite combining statistical analysis and sanity checks.
    """
    
    def __init__(self, alpha: float = 0.05, random_seed: int = 42):
        """Initialize validation suite."""
        self.alpha = alpha
        self.random_seed = random_seed
        self.statistical_analyzer = StatisticalAnalyzer(alpha, random_seed)
        self.sanity_checker = SanityChecker(random_seed)
    
    def run_full_validation(
        self,
        faithfulness_results: Dict[str, List[FaithfulnessResult]],
        roar_results: Optional[Dict[str, List[ROARResult]]] = None,
        random_explainer_results: Optional[List[FaithfulnessResult]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive validation including statistical tests and sanity checks.
        
        Args:
            faithfulness_results: Faithfulness results by explainer
            roar_results: ROAR results by explainer (optional)
            random_explainer_results: Random explainer results for sanity checks
            
        Returns:
            Comprehensive validation report
        """
        validation_report = {
            'statistical_tests': {},
            'sanity_checks': {},
            'summary': {}
        }
        
        # Statistical tests between explainers
        explainer_names = list(faithfulness_results.keys())
        
        if len(explainer_names) >= 2:
            # Pairwise comparisons
            pairwise_tests = {}
            p_values_for_correction = []
            
            for i, explainer1 in enumerate(explainer_names):
                for j, explainer2 in enumerate(explainer_names[i+1:], i+1):
                    scores1 = [r.f_score for r in faithfulness_results[explainer1]]
                    scores2 = [r.f_score for r in faithfulness_results[explainer2]]
                    
                    if len(scores1) == len(scores2) and len(scores1) > 1:
                        # Paired t-test
                        t_test = self.statistical_analyzer.paired_t_test(scores1, scores2)
                        pairwise_tests[f"{explainer1}_vs_{explainer2}"] = t_test
                        p_values_for_correction.append(t_test.p_value)
                        
                        # Wilcoxon test as non-parametric alternative
                        wilcoxon_test = self.statistical_analyzer.wilcoxon_signed_rank_test(scores1, scores2)
                        pairwise_tests[f"{explainer1}_vs_{explainer2}_wilcoxon"] = wilcoxon_test
            
            validation_report['statistical_tests']['pairwise_comparisons'] = pairwise_tests
            
            # Multiple comparisons correction
            if p_values_for_correction:
                rejected, corrected_p = self.statistical_analyzer.multiple_comparisons_correction(
                    p_values_for_correction, method="bonferroni"
                )
                validation_report['statistical_tests']['multiple_comparisons'] = {
                    'original_p_values': p_values_for_correction,
                    'corrected_p_values': corrected_p,
                    'rejected_hypotheses': rejected
                }
        
        # Bootstrap confidence intervals for each explainer
        bootstrap_results = {}
        for explainer_name, results in faithfulness_results.items():
            scores = [r.f_score for r in results]
            if len(scores) > 1:
                bootstrap = self.statistical_analyzer.bootstrap_confidence_interval(scores)
                bootstrap_results[explainer_name] = bootstrap
        
        validation_report['statistical_tests']['bootstrap_intervals'] = bootstrap_results
        
        # Sanity checks
        sanity_results = {}
        
        # Metric bounds validation
        all_results = []
        for results in faithfulness_results.values():
            all_results.extend(results)
        
        if all_results:
            bounds_check = self.sanity_checker.metric_bounds_validation(all_results)
            sanity_results['bounds_validation'] = bounds_check
        
        # Random explainer validation
        if random_explainer_results:
            informed_results = []
            for explainer_name, results in faithfulness_results.items():
                if 'random' not in explainer_name.lower():
                    informed_results.extend(results)
            
            if informed_results:
                random_check = self.sanity_checker.random_explainer_validation(
                    random_explainer_results, informed_results
                )
                sanity_results['random_explainer_validation'] = random_check
        
        validation_report['sanity_checks'] = sanity_results
        
        # Summary statistics
        summary = {}
        
        # Count significant tests
        significant_tests = 0
        total_tests = 0
        
        if 'pairwise_comparisons' in validation_report['statistical_tests']:
            for test_result in validation_report['statistical_tests']['pairwise_comparisons'].values():
                if hasattr(test_result, 'is_significant'):
                    total_tests += 1
                    if test_result.is_significant:
                        significant_tests += 1
        
        summary['significant_tests_fraction'] = significant_tests / total_tests if total_tests > 0 else 0.0
        
        # Count passed sanity checks
        passed_checks = 0
        total_checks = 0
        
        for check_result in sanity_results.values():
            if hasattr(check_result, 'passed'):
                total_checks += 1
                if check_result.passed:
                    passed_checks += 1
        
        summary['passed_sanity_checks_fraction'] = passed_checks / total_checks if total_checks > 0 else 0.0
        
        # Overall assessment
        if summary['passed_sanity_checks_fraction'] >= 0.8:
            summary['overall_assessment'] = 'GOOD'
        elif summary['passed_sanity_checks_fraction'] >= 0.6:
            summary['overall_assessment'] = 'MODERATE'
        else:
            summary['overall_assessment'] = 'POOR'
        
        validation_report['summary'] = summary
        
        return validation_report
    
    def generate_validation_report_text(self, validation_report: Dict[str, Any]) -> str:
        """Generate human-readable validation report."""
        report = "# Statistical Analysis and Validation Report\n\n"
        
        # Summary
        summary = validation_report.get('summary', {})
        report += "## Summary\n\n"
        report += f"- Overall Assessment: **{summary.get('overall_assessment', 'UNKNOWN')}**\n"
        report += f"- Significant Statistical Tests: {summary.get('significant_tests_fraction', 0):.1%}\n"
        report += f"- Passed Sanity Checks: {summary.get('passed_sanity_checks_fraction', 0):.1%}\n\n"
        
        # Statistical tests
        if 'statistical_tests' in validation_report:
            report += "## Statistical Tests\n\n"
            
            # Pairwise comparisons
            if 'pairwise_comparisons' in validation_report['statistical_tests']:
                report += "### Pairwise Comparisons\n\n"
                for test_name, test_result in validation_report['statistical_tests']['pairwise_comparisons'].items():
                    if hasattr(test_result, '__str__'):
                        report += f"- {test_name}: {str(test_result)}\n"
                report += "\n"
            
            # Bootstrap intervals
            if 'bootstrap_intervals' in validation_report['statistical_tests']:
                report += "### Bootstrap Confidence Intervals\n\n"
                for explainer, bootstrap_result in validation_report['statistical_tests']['bootstrap_intervals'].items():
                    if hasattr(bootstrap_result, '__str__'):
                        report += f"- {explainer}: {str(bootstrap_result)}\n"
                report += "\n"
        
        # Sanity checks
        if 'sanity_checks' in validation_report:
            report += "## Sanity Checks\n\n"
            for check_name, check_result in validation_report['sanity_checks'].items():
                if hasattr(check_result, '__str__'):
                    report += f"- {str(check_result)}\n"
            report += "\n"
        
        return report


# Convenience functions
def run_statistical_analysis(
    faithfulness_results: Dict[str, List[FaithfulnessResult]],
    alpha: float = 0.05
) -> Dict[str, StatisticalTestResult]:
    """
    Run statistical analysis on faithfulness results.
    
    Args:
        faithfulness_results: Results by explainer name
        alpha: Significance level
        
    Returns:
        Statistical test results
    """
    analyzer = StatisticalAnalyzer(alpha=alpha)
    results = {}
    
    explainer_names = list(faithfulness_results.keys())
    
    # Pairwise comparisons
    for i, explainer1 in enumerate(explainer_names):
        for j, explainer2 in enumerate(explainer_names[i+1:], i+1):
            scores1 = [r.f_score for r in faithfulness_results[explainer1]]
            scores2 = [r.f_score for r in faithfulness_results[explainer2]]
            
            if len(scores1) == len(scores2) and len(scores1) > 1:
                test_result = analyzer.paired_t_test(scores1, scores2)
                results[f"{explainer1}_vs_{explainer2}"] = test_result
    
    return results


def run_sanity_checks(
    faithfulness_results: Dict[str, List[FaithfulnessResult]],
    random_explainer_results: Optional[List[FaithfulnessResult]] = None
) -> Dict[str, SanityCheckResult]:
    """
    Run sanity checks on faithfulness results.
    
    Args:
        faithfulness_results: Results by explainer name
        random_explainer_results: Random explainer results for validation
        
    Returns:
        Sanity check results
    """
    checker = SanityChecker()
    results = {}
    
    # Collect all results for bounds validation
    all_results = []
    for explainer_results in faithfulness_results.values():
        all_results.extend(explainer_results)
    
    if all_results:
        results['bounds_validation'] = checker.metric_bounds_validation(all_results)
    
    # Random explainer validation
    if random_explainer_results:
        informed_results = []
        for explainer_name, explainer_results in faithfulness_results.items():
            if 'random' not in explainer_name.lower():
                informed_results.extend(explainer_results)
        
        if informed_results:
            results['random_explainer_validation'] = checker.random_explainer_validation(
                random_explainer_results, informed_results
            )
    
    return results