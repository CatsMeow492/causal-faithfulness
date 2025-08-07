"""
Visualization and analysis tools for causal-faithfulness metric results.
Creates bar charts, scatter plots, and performance tables with proper metadata.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
from datetime import datetime

from .faithfulness import FaithfulnessResult
from .roar import ROARResult, ROARCorrelationResult
from .evaluation import ExperimentResult


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""
    output_dir: str = "figures"
    figure_format: str = "png"
    figure_dpi: int = 300
    figure_size: Tuple[float, float] = (10, 6)
    color_palette: str = "Set2"
    font_size: int = 12
    save_data: bool = True  # Save underlying data as CSV
    include_metadata: bool = True
    random_seed: int = 42
    
    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)


class ResultVisualizer:
    """
    Main class for creating visualizations of faithfulness and ROAR results.
    Generates publication-ready figures with proper captions and metadata.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        """Initialize visualizer with configuration."""
        self.config = config or VisualizationConfig()
        
        # Set up matplotlib and seaborn styling
        plt.style.use('default')
        sns.set_palette(self.config.color_palette)
        plt.rcParams.update({
            'font.size': self.config.font_size,
            'figure.figsize': self.config.figure_size,
            'figure.dpi': self.config.figure_dpi,
            'savefig.dpi': self.config.figure_dpi,
            'savefig.bbox': 'tight',
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        
        np.random.seed(self.config.random_seed)
    
    def create_faithfulness_bar_chart(
        self,
        results: Dict[str, List[FaithfulnessResult]],
        title: str = "Mean Faithfulness Scores by Explainer",
        filename: str = "faithfulness_scores_bar_chart"
    ) -> str:
        """
        Create bar chart showing mean F-scores by explainer with error bars.
        
        Args:
            results: Dictionary mapping explainer names to faithfulness results
            title: Chart title
            filename: Output filename (without extension)
            
        Returns:
            Path to saved figure
        """
        # Prepare data
        explainer_names = []
        mean_scores = []
        std_scores = []
        n_samples = []
        
        for explainer_name, explainer_results in results.items():
            if not explainer_results:
                continue
                
            scores = [r.f_score for r in explainer_results]
            explainer_names.append(explainer_name)
            mean_scores.append(np.mean(scores))
            std_scores.append(np.std(scores))
            n_samples.append(len(scores))
        
        if not explainer_names:
            warnings.warn("No valid results for bar chart")
            return ""
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Create bar chart with error bars
        bars = ax.bar(explainer_names, mean_scores, yerr=std_scores, 
                     capsize=5, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize chart
        ax.set_title(title, fontsize=self.config.font_size + 2, fontweight='bold')
        ax.set_xlabel('Explanation Method', fontsize=self.config.font_size)
        ax.set_ylabel('Mean Faithfulness Score (F)', fontsize=self.config.font_size)
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bar, mean_val, std_val, n in zip(bars, mean_scores, std_scores, n_samples):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                   f'{mean_val:.3f}±{std_val:.3f}\n(n={n})',
                   ha='center', va='bottom', fontsize=self.config.font_size - 2)
        
        # Rotate x-axis labels if needed
        if len(explainer_names) > 4:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure
        filepath = self._save_figure(fig, filename, title)
        
        # Save underlying data
        if self.config.save_data:
            data_df = pd.DataFrame({
                'explainer': explainer_names,
                'mean_f_score': mean_scores,
                'std_f_score': std_scores,
                'n_samples': n_samples
            })
            data_path = filepath.replace(f'.{self.config.figure_format}', '_data.csv')
            data_df.to_csv(data_path, index=False)
        
        plt.close(fig)
        return filepath
    
    def create_roar_correlation_scatter(
        self,
        correlation_results: List[ROARCorrelationResult],
        title: str = "Faithfulness vs ROAR Correlation Analysis",
        filename: str = "roar_faithfulness_correlation"
    ) -> str:
        """
        Create scatter plot showing correlation between F-scores and ROAR accuracy drops.
        
        Args:
            correlation_results: List of correlation results
            title: Chart title
            filename: Output filename (without extension)
            
        Returns:
            Path to saved figure
        """
        if not correlation_results:
            warnings.warn("No correlation results for scatter plot")
            return ""
        
        # Create figure with subplots for each explainer
        n_explainers = len(correlation_results)
        n_cols = min(3, n_explainers)
        n_rows = (n_explainers + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_explainers == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten() if n_explainers > 1 else [axes[0]]
        
        for i, result in enumerate(correlation_results):
            ax = axes_flat[i]
            
            # Create scatter plot
            scatter = ax.scatter(result.roar_scores, result.faithfulness_scores, 
                               alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
            
            # Add trend line
            if len(result.roar_scores) > 1:
                z = np.polyfit(result.roar_scores, result.faithfulness_scores, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(result.roar_scores), max(result.roar_scores), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
            
            # Customize subplot
            ax.set_title(f'{result.explainer_name}\nr={result.pearson_r:.3f}, p={result.p_value:.3f}',
                        fontsize=self.config.font_size)
            ax.set_xlabel('ROAR Accuracy Drop', fontsize=self.config.font_size - 1)
            ax.set_ylabel('Faithfulness Score (F)', fontsize=self.config.font_size - 1)
            ax.set_ylim(0, 1.0)
            
            # Add significance indicator
            if result.is_significant:
                ax.text(0.05, 0.95, '* Significant', transform=ax.transAxes,
                       fontsize=self.config.font_size - 2, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Hide unused subplots
        for i in range(n_explainers, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.suptitle(title, fontsize=self.config.font_size + 2, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        filepath = self._save_figure(fig, filename, title)
        
        # Save underlying data
        if self.config.save_data:
            all_data = []
            for result in correlation_results:
                for roar_score, faith_score in zip(result.roar_scores, result.faithfulness_scores):
                    all_data.append({
                        'explainer': result.explainer_name,
                        'roar_score': roar_score,
                        'faithfulness_score': faith_score,
                        'pearson_r': result.pearson_r,
                        'p_value': result.p_value,
                        'is_significant': result.is_significant
                    })
            
            data_df = pd.DataFrame(all_data)
            data_path = filepath.replace(f'.{self.config.figure_format}', '_data.csv')
            data_df.to_csv(data_path, index=False)
        
        plt.close(fig)
        return filepath
    
    def create_performance_table(
        self,
        experiment_result: ExperimentResult,
        filename: str = "performance_metrics_table"
    ) -> str:
        """
        Create performance table with runtime and memory usage metrics.
        
        Args:
            experiment_result: Complete experiment results
            filename: Output filename (without extension)
            
        Returns:
            Path to saved table (as image and CSV)
        """
        # Prepare performance data
        performance_data = []
        
        for explainer_name in experiment_result.explainer_results.keys():
            # Get summary statistics
            summary = experiment_result.summary_statistics.get(explainer_name, {})
            computation = experiment_result.computation_metrics.get(explainer_name, {})
            
            # Calculate additional metrics from individual results
            explainer_results = experiment_result.explainer_results[explainer_name]
            if explainer_results:
                # Runtime metrics
                total_explanation_time = sum(
                    r.computation_metrics.get('explanation_time', 0) for r in explainer_results
                )
                total_faithfulness_time = sum(
                    r.computation_metrics.get('faithfulness_time', 0) for r in explainer_results
                )
                total_model_queries = sum(
                    r.computation_metrics.get('n_model_queries', 0) for r in explainer_results
                )
                
                performance_data.append({
                    'Explainer': explainer_name,
                    'Mean F-Score': f"{summary.get('mean_f_score', 0):.4f}",
                    'Std F-Score': f"{summary.get('std_f_score', 0):.4f}",
                    'Samples': summary.get('total_results', 0),
                    'Significant Results': f"{summary.get('significant_results', 0)}/{summary.get('total_results', 0)}",
                    'Total Runtime (s)': f"{computation.get('total_runtime', 0):.2f}",
                    'Avg Time/Sample (s)': f"{computation.get('avg_time_per_sample', 0):.3f}",
                    'Explanation Time (s)': f"{total_explanation_time:.2f}",
                    'Faithfulness Time (s)': f"{total_faithfulness_time:.2f}",
                    'Total Model Queries': total_model_queries,
                    'Queries/Sample': f"{total_model_queries / max(len(explainer_results), 1):.1f}"
                })
        
        if not performance_data:
            warnings.warn("No performance data available for table")
            return ""
        
        # Create DataFrame
        df = pd.DataFrame(performance_data)
        
        # Save as CSV
        csv_path = os.path.join(self.config.output_dir, f"{filename}.csv")
        df.to_csv(csv_path, index=False)
        
        # Create visual table
        fig, ax = plt.subplots(figsize=(16, max(6, len(performance_data) * 0.5 + 2)))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(self.config.font_size - 2)
        table.scale(1, 2)
        
        # Color header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(performance_data) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title(f"Performance Metrics - {experiment_result.experiment_config.experiment_name}",
                 fontsize=self.config.font_size + 2, fontweight='bold', pad=20)
        
        # Save figure
        img_path = self._save_figure(fig, filename, "Performance Metrics Table")
        
        plt.close(fig)
        return csv_path
    
    def create_comparison_heatmap(
        self,
        results: Dict[str, List[FaithfulnessResult]],
        title: str = "Faithfulness Score Comparison Heatmap",
        filename: str = "faithfulness_comparison_heatmap"
    ) -> str:
        """
        Create heatmap comparing faithfulness scores across explainers and samples.
        
        Args:
            results: Dictionary mapping explainer names to faithfulness results
            title: Chart title
            filename: Output filename (without extension)
            
        Returns:
            Path to saved figure
        """
        # Prepare data matrix
        explainer_names = list(results.keys())
        max_samples = max(len(results[name]) for name in explainer_names)
        
        # Create matrix with F-scores
        score_matrix = np.full((len(explainer_names), max_samples), np.nan)
        
        for i, explainer_name in enumerate(explainer_names):
            explainer_results = results[explainer_name]
            for j, result in enumerate(explainer_results):
                if j < max_samples:
                    score_matrix[i, j] = result.f_score
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(8, max_samples * 0.3), len(explainer_names) * 0.8))
        
        # Use a colormap that handles NaN values
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='lightgray')
        
        im = ax.imshow(score_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(min(max_samples, 50)))  # Limit x-axis labels
        ax.set_yticks(range(len(explainer_names)))
        ax.set_xticklabels([f'S{i+1}' for i in range(min(max_samples, 50))])
        ax.set_yticklabels(explainer_names)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Faithfulness Score (F)', rotation=270, labelpad=20)
        
        # Set title and labels
        ax.set_title(title, fontsize=self.config.font_size + 2, fontweight='bold')
        ax.set_xlabel('Sample Index', fontsize=self.config.font_size)
        ax.set_ylabel('Explanation Method', fontsize=self.config.font_size)
        
        plt.tight_layout()
        
        # Save figure
        filepath = self._save_figure(fig, filename, title)
        
        plt.close(fig)
        return filepath
    
    def _save_figure(self, fig: plt.Figure, filename: str, title: str) -> str:
        """
        Save figure with metadata and proper formatting.
        
        Args:
            fig: Matplotlib figure
            filename: Base filename
            title: Figure title for metadata
            
        Returns:
            Path to saved file
        """
        # Create full filepath
        filepath = os.path.join(self.config.output_dir, f"{filename}.{self.config.figure_format}")
        
        # Add metadata if requested
        if self.config.include_metadata:
            metadata = {
                'Title': title,
                'Generated': datetime.now().isoformat(),
                'Software': 'Causal-Faithfulness Metric Implementation',
                'Random Seed': str(self.config.random_seed)
            }
            
            # Save with metadata
            fig.savefig(filepath, format=self.config.figure_format, 
                       dpi=self.config.figure_dpi, bbox_inches='tight',
                       metadata=metadata)
        else:
            fig.savefig(filepath, format=self.config.figure_format,
                       dpi=self.config.figure_dpi, bbox_inches='tight')
        
        return filepath
    
    def create_summary_dashboard(
        self,
        experiment_result: ExperimentResult,
        correlation_results: Optional[List[ROARCorrelationResult]] = None,
        filename: str = "summary_dashboard"
    ) -> str:
        """
        Create comprehensive summary dashboard with multiple visualizations.
        
        Args:
            experiment_result: Complete experiment results
            correlation_results: Optional ROAR correlation results
            filename: Output filename (without extension)
            
        Returns:
            Path to saved dashboard
        """
        # Create figure with subplots
        if correlation_results:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.3, wspace=0.3)
        else:
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Faithfulness bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        self._create_bar_chart_subplot(ax1, experiment_result.explainer_results, 
                                     "Mean Faithfulness Scores")
        
        # 2. Performance comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._create_performance_subplot(ax2, experiment_result)
        
        # 3. Score distribution
        ax3 = fig.add_subplot(gs[1, :])
        self._create_distribution_subplot(ax3, experiment_result.explainer_results)
        
        # 4. ROAR correlation (if available)
        if correlation_results:
            ax4 = fig.add_subplot(gs[2, :])
            self._create_correlation_subplot(ax4, correlation_results)
        
        # Add main title
        fig.suptitle(f"Faithfulness Analysis Dashboard - {experiment_result.experiment_config.experiment_name}",
                    fontsize=self.config.font_size + 4, fontweight='bold')
        
        # Save dashboard
        filepath = self._save_figure(fig, filename, "Summary Dashboard")
        
        plt.close(fig)
        return filepath
    
    def _create_bar_chart_subplot(self, ax, results: Dict[str, List[FaithfulnessResult]], title: str):
        """Create bar chart in subplot."""
        explainer_names = []
        mean_scores = []
        std_scores = []
        
        for explainer_name, explainer_results in results.items():
            if explainer_results:
                scores = [r.f_score for r in explainer_results]
                explainer_names.append(explainer_name)
                mean_scores.append(np.mean(scores))
                std_scores.append(np.std(scores))
        
        if explainer_names:
            bars = ax.bar(explainer_names, mean_scores, yerr=std_scores, capsize=3, alpha=0.8)
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('F-Score')
            ax.set_ylim(0, 1.0)
            
            # Add value labels
            for bar, mean_val in zip(bars, mean_scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10)
            
            if len(explainer_names) > 3:
                ax.tick_params(axis='x', rotation=45)
    
    def _create_performance_subplot(self, ax, experiment_result: ExperimentResult):
        """Create performance comparison in subplot."""
        explainer_names = []
        runtimes = []
        
        for explainer_name, metrics in experiment_result.computation_metrics.items():
            explainer_names.append(explainer_name)
            runtimes.append(metrics.get('total_runtime', 0))
        
        if explainer_names:
            bars = ax.bar(explainer_names, runtimes, alpha=0.8, color='orange')
            ax.set_title('Runtime Comparison', fontweight='bold')
            ax.set_ylabel('Runtime (seconds)')
            
            # Add value labels
            for bar, runtime in zip(bars, runtimes):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(runtimes) * 0.01,
                       f'{runtime:.1f}s', ha='center', va='bottom', fontsize=10)
            
            if len(explainer_names) > 3:
                ax.tick_params(axis='x', rotation=45)
    
    def _create_distribution_subplot(self, ax, results: Dict[str, List[FaithfulnessResult]]):
        """Create score distribution in subplot."""
        all_scores = []
        labels = []
        
        for explainer_name, explainer_results in results.items():
            if explainer_results:
                scores = [r.f_score for r in explainer_results]
                all_scores.append(scores)
                labels.append(explainer_name)
        
        if all_scores:
            ax.boxplot(all_scores, labels=labels)
            ax.set_title('Faithfulness Score Distributions', fontweight='bold')
            ax.set_ylabel('F-Score')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)
            
            if len(labels) > 3:
                ax.tick_params(axis='x', rotation=45)
    
    def _create_correlation_subplot(self, ax, correlation_results: List[ROARCorrelationResult]):
        """Create correlation summary in subplot."""
        explainer_names = [r.explainer_name for r in correlation_results]
        correlations = [r.pearson_r for r in correlation_results]
        p_values = [r.p_value for r in correlation_results]
        
        # Create bar chart with significance indicators
        bars = ax.bar(explainer_names, correlations, alpha=0.8, color='green')
        
        # Color bars based on significance
        for bar, p_val in zip(bars, p_values):
            if p_val < 0.05:
                bar.set_color('darkgreen')
            else:
                bar.set_color('lightgreen')
        
        ax.set_title('ROAR-Faithfulness Correlations', fontweight='bold')
        ax.set_ylabel('Pearson Correlation (r)')
        ax.set_ylim(-1, 1)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add significance line
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Strong correlation')
        ax.legend()
        
        # Add value labels
        for bar, corr, p_val in zip(bars, correlations, p_values):
            height = bar.get_height()
            significance = '*' if p_val < 0.05 else ''
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + 0.05 if height >= 0 else height - 0.1,
                   f'{corr:.3f}{significance}', ha='center', va='bottom', fontsize=10)
        
        if len(explainer_names) > 3:
            ax.tick_params(axis='x', rotation=45)


# Convenience functions
def create_all_visualizations(
    experiment_result: ExperimentResult,
    correlation_results: Optional[List[ROARCorrelationResult]] = None,
    output_dir: str = "figures"
) -> Dict[str, str]:
    """
    Create all standard visualizations for an experiment.
    
    Args:
        experiment_result: Complete experiment results
        correlation_results: Optional ROAR correlation results
        output_dir: Output directory for figures
        
    Returns:
        Dictionary mapping visualization names to file paths
    """
    config = VisualizationConfig(output_dir=output_dir)
    visualizer = ResultVisualizer(config)
    
    filepaths = {}
    
    # 1. Faithfulness bar chart
    filepaths['bar_chart'] = visualizer.create_faithfulness_bar_chart(
        experiment_result.explainer_results,
        title=f"Faithfulness Scores - {experiment_result.experiment_config.experiment_name}"
    )
    
    # 2. Performance table
    filepaths['performance_table'] = visualizer.create_performance_table(
        experiment_result
    )
    
    # 3. Comparison heatmap
    filepaths['heatmap'] = visualizer.create_comparison_heatmap(
        experiment_result.explainer_results,
        title=f"Score Comparison - {experiment_result.experiment_config.experiment_name}"
    )
    
    # 4. ROAR correlation scatter plot (if available)
    if correlation_results:
        filepaths['correlation_scatter'] = visualizer.create_roar_correlation_scatter(
            correlation_results,
            title=f"ROAR Correlation - {experiment_result.experiment_config.experiment_name}"
        )
    
    # 5. Summary dashboard
    filepaths['dashboard'] = visualizer.create_summary_dashboard(
        experiment_result, correlation_results
    )
    
    return filepaths


def save_results_summary(
    experiment_result: ExperimentResult,
    correlation_results: Optional[List[ROARCorrelationResult]] = None,
    output_dir: str = "results"
) -> str:
    """
    Save comprehensive results summary as text report.
    
    Args:
        experiment_result: Complete experiment results
        correlation_results: Optional ROAR correlation results
        output_dir: Output directory
        
    Returns:
        Path to saved summary report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary report
    report = f"# Faithfulness Analysis Summary\n\n"
    report += f"**Experiment**: {experiment_result.experiment_config.experiment_name}\n"
    report += f"**Dataset**: {experiment_result.dataset_info['name']}\n"
    report += f"**Model**: {experiment_result.model_info['name']}\n"
    report += f"**Generated**: {experiment_result.timestamp}\n"
    report += f"**Total Runtime**: {experiment_result.total_runtime:.2f} seconds\n\n"
    
    # Summary statistics
    report += "## Faithfulness Scores\n\n"
    for explainer_name, stats in experiment_result.summary_statistics.items():
        report += f"### {explainer_name}\n"
        report += f"- Mean F-score: {stats['mean_f_score']:.4f} ± {stats['std_f_score']:.4f}\n"
        report += f"- Median F-score: {stats['median_f_score']:.4f}\n"
        report += f"- Range: [{stats['min_f_score']:.4f}, {stats['max_f_score']:.4f}]\n"
        report += f"- Significant results: {stats['significant_results']}/{stats['total_results']}\n"
        report += f"- Runtime: {stats['total_computation_time']:.2f}s\n\n"
    
    # ROAR correlations
    if correlation_results:
        report += "## ROAR Correlations\n\n"
        for result in correlation_results:
            report += f"### {result.explainer_name}\n"
            report += f"- Pearson r: {result.pearson_r:.4f} (p={result.p_value:.4f})\n"
            report += f"- Spearman ρ: {result.spearman_rho:.4f} (p={result.spearman_p:.4f})\n"
            report += f"- Significant: {'Yes' if result.is_significant else 'No'}\n\n"
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_result.experiment_config.experiment_name}_summary_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(report)
    
    return filepath