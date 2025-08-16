"""
Experimental Validation Framework for Continual Learning Research

Comprehensive framework for validating continual learning algorithms with statistical rigor.
Includes controlled experiments, ablation studies, and publication-ready benchmarking.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import json
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import scipy.stats as stats
from collections import defaultdict
import warnings
from sklearn.metrics import classification_report, confusion_matrix
import hashlib
import os

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experimental validation."""
    experiment_name: str
    description: str
    random_seed: int = 42
    num_runs: int = 5
    confidence_level: float = 0.95
    significance_threshold: float = 0.05
    
    # Dataset configuration
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Evaluation metrics
    primary_metric: str = "accuracy"
    secondary_metrics: List[str] = field(default_factory=lambda: ["f1_score", "precision", "recall"])
    
    # Statistical testing
    enable_statistical_tests: bool = True
    multiple_comparison_correction: str = "bonferroni"  # "bonferroni", "holm", "fdr_bh"
    
    # Reproducibility
    save_intermediate_results: bool = True
    log_detailed_metrics: bool = True
    save_model_checkpoints: bool = False


@dataclass
class ExperimentResult:
    """Single experiment run result."""
    run_id: int
    metrics: Dict[str, float]
    training_time: float
    memory_usage: float
    model_size: int
    convergence_epoch: int
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete benchmark result with statistical analysis."""
    experiment_name: str
    algorithm_name: str
    results: List[ExperimentResult]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_significance: Dict[str, Any]
    effect_sizes: Dict[str, float]


class StatisticalAnalyzer:
    """Statistical analysis tools for experimental validation."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
    
    def compute_confidence_interval(
        self, 
        values: List[float], 
        confidence_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """Compute confidence interval for a set of values."""
        
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        values = np.array(values)
        n = len(values)
        
        if n < 2:
            return (values[0], values[0]) if n == 1 else (0.0, 0.0)
        
        mean = np.mean(values)
        sem = stats.sem(values)  # Standard error of the mean
        
        # Use t-distribution for small samples
        alpha = 1.0 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n - 1)
        margin_error = t_critical * sem
        
        return (mean - margin_error, mean + margin_error)
    
    def paired_t_test(
        self, 
        results1: List[float], 
        results2: List[float]
    ) -> Dict[str, Any]:
        """Perform paired t-test between two sets of results."""
        
        if len(results1) != len(results2):
            raise ValueError("Results lists must have the same length")
        
        results1 = np.array(results1)
        results2 = np.array(results2)
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(results1, results2)
        
        # Effect size (Cohen's d for paired samples)
        differences = results1 - results2
        cohen_d = np.mean(differences) / np.std(differences, ddof=1)
        
        # Interpretation
        if abs(cohen_d) < 0.2:
            effect_size_interpretation = "negligible"
        elif abs(cohen_d) < 0.5:
            effect_size_interpretation = "small"
        elif abs(cohen_d) < 0.8:
            effect_size_interpretation = "medium"
        else:
            effect_size_interpretation = "large"
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohen_d": cohen_d,
            "effect_size_interpretation": effect_size_interpretation,
            "significant": p_value < self.alpha,
            "mean_difference": np.mean(differences),
            "std_difference": np.std(differences, ddof=1)
        }
    
    def one_way_anova(
        self, 
        *groups: List[List[float]]
    ) -> Dict[str, Any]:
        """Perform one-way ANOVA across multiple groups."""
        
        # Flatten groups for ANOVA
        group_data = [np.array(group) for group in groups]
        
        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*group_data)
        
        # Effect size (eta-squared)
        grand_mean = np.mean(np.concatenate(group_data))
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in group_data)
        ss_total = sum(np.sum((group - grand_mean)**2) for group in group_data)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
        
        return {
            "f_statistic": f_stat,
            "p_value": p_value,
            "eta_squared": eta_squared,
            "significant": p_value < self.alpha,
            "num_groups": len(groups),
            "total_samples": sum(len(group) for group in groups)
        }
    
    def multiple_comparison_correction(
        self, 
        p_values: List[float], 
        method: str = "bonferroni"
    ) -> Tuple[List[bool], List[float]]:
        """Apply multiple comparison correction."""
        
        p_values = np.array(p_values)
        n = len(p_values)
        
        if method == "bonferroni":
            corrected_alpha = self.alpha / n
            rejected = p_values < corrected_alpha
            adjusted_p = p_values * n
            
        elif method == "holm":
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            rejected = np.zeros(n, dtype=bool)
            adjusted_p = np.full(n, 1.0)
            
            for i, idx in enumerate(sorted_indices):
                corrected_alpha = self.alpha / (n - i)
                if p_values[idx] < corrected_alpha:
                    rejected[idx] = True
                    adjusted_p[idx] = p_values[idx] * (n - i)
                else:
                    break
        
        elif method == "fdr_bh":
            # Benjamini-Hochberg FDR control
            sorted_indices = np.argsort(p_values)
            rejected = np.zeros(n, dtype=bool)
            adjusted_p = np.full(n, 1.0)
            
            for i in range(n-1, -1, -1):
                idx = sorted_indices[i]
                corrected_alpha = (i + 1) / n * self.alpha
                if p_values[idx] <= corrected_alpha:
                    rejected[sorted_indices[:i+1]] = True
                    break
            
            # Adjusted p-values for FDR
            for i, idx in enumerate(sorted_indices):
                adjusted_p[idx] = min(1.0, p_values[idx] * n / (i + 1))
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Ensure adjusted p-values don't exceed 1.0
        adjusted_p = np.minimum(adjusted_p, 1.0)
        
        return rejected.tolist(), adjusted_p.tolist()


class ExperimentRunner:
    """Runs controlled experiments with proper statistical validation."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results_db = {}
        self.analyzer = StatisticalAnalyzer(config.confidence_level)
        
        # Set up reproducibility
        self._setup_reproducibility()
        
        # Create experiment directory
        self.experiment_dir = Path(f"experiments/{config.experiment_name}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Experiment runner initialized: {config.experiment_name}")
    
    def _setup_reproducibility(self):
        """Set up reproducible environment."""
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)
            torch.cuda.manual_seed_all(self.config.random_seed)
        
        # For even more reproducibility (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def run_experiment(
        self,
        algorithm_name: str,
        model_factory: Callable,
        train_function: Callable,
        evaluate_function: Callable,
        dataset: Any,
        **kwargs
    ) -> BenchmarkResult:
        """Run complete experiment with multiple runs."""
        
        logger.info(f"Starting experiment: {algorithm_name}")
        
        results = []
        
        for run_id in range(self.config.num_runs):
            logger.info(f"Run {run_id + 1}/{self.config.num_runs}")
            
            # Ensure different random state for each run
            run_seed = self.config.random_seed + run_id
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            
            # Run single experiment
            result = self._run_single_experiment(
                run_id, algorithm_name, model_factory, 
                train_function, evaluate_function, dataset, **kwargs
            )
            
            results.append(result)
            
            # Save intermediate results
            if self.config.save_intermediate_results:
                self._save_intermediate_result(algorithm_name, run_id, result)
        
        # Compute aggregate statistics
        benchmark_result = self._compute_benchmark_statistics(
            algorithm_name, results
        )
        
        # Save complete benchmark
        self._save_benchmark_result(benchmark_result)
        
        logger.info(f"Experiment completed: {algorithm_name}")
        return benchmark_result
    
    def _run_single_experiment(
        self,
        run_id: int,
        algorithm_name: str,
        model_factory: Callable,
        train_function: Callable,
        evaluate_function: Callable,
        dataset: Any,
        **kwargs
    ) -> ExperimentResult:
        """Run a single experiment instance."""
        
        start_time = time.time()
        
        # Create model
        model = model_factory(**kwargs)
        
        # Get initial memory usage
        initial_memory = self._get_memory_usage()
        
        # Training
        training_start = time.time()
        training_info = train_function(model, dataset, **kwargs)
        training_time = time.time() - training_start
        
        # Evaluation
        metrics = evaluate_function(model, dataset, **kwargs)
        
        # Get final memory usage
        final_memory = self._get_memory_usage()
        memory_usage = final_memory - initial_memory
        
        # Model size
        model_size = sum(p.numel() for p in model.parameters())
        
        # Extract convergence information
        convergence_epoch = training_info.get('convergence_epoch', -1)
        
        total_time = time.time() - start_time
        
        return ExperimentResult(
            run_id=run_id,
            metrics=metrics,
            training_time=training_time,
            memory_usage=memory_usage,
            model_size=model_size,
            convergence_epoch=convergence_epoch,
            additional_info={
                'total_time': total_time,
                'training_info': training_info,
                'algorithm_name': algorithm_name
            }
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
    
    def _compute_benchmark_statistics(
        self, 
        algorithm_name: str, 
        results: List[ExperimentResult]
    ) -> BenchmarkResult:
        """Compute comprehensive statistics for benchmark results."""
        
        # Extract metrics from all runs
        all_metrics = defaultdict(list)
        for result in results:
            for metric_name, value in result.metrics.items():
                all_metrics[metric_name].append(value)
        
        # Compute mean and standard deviation
        mean_metrics = {
            metric: np.mean(values) for metric, values in all_metrics.items()
        }
        
        std_metrics = {
            metric: np.std(values, ddof=1) for metric, values in all_metrics.items()
        }
        
        # Compute confidence intervals
        confidence_intervals = {
            metric: self.analyzer.compute_confidence_interval(values)
            for metric, values in all_metrics.items()
        }
        
        # Placeholder for statistical significance (would compare against baseline)
        statistical_significance = {
            "tested_against_baseline": False,
            "p_values": {},
            "significant_differences": []
        }
        
        # Effect sizes (placeholder - would need comparison)
        effect_sizes = {}
        
        return BenchmarkResult(
            experiment_name=self.config.experiment_name,
            algorithm_name=algorithm_name,
            results=results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            confidence_intervals=confidence_intervals,
            statistical_significance=statistical_significance,
            effect_sizes=effect_sizes
        )
    
    def compare_algorithms(
        self, 
        benchmark_results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """Compare multiple algorithms with statistical testing."""
        
        if len(benchmark_results) < 2:
            raise ValueError("Need at least 2 algorithms to compare")
        
        comparison_results = {
            "algorithms": [result.algorithm_name for result in benchmark_results],
            "pairwise_comparisons": {},
            "overall_anova": {},
            "rankings": {}
        }
        
        # Extract primary metric for comparison
        primary_metric = self.config.primary_metric
        
        # Collect results for each algorithm
        algorithm_results = {}
        for benchmark in benchmark_results:
            if primary_metric in benchmark.mean_metrics:
                algorithm_results[benchmark.algorithm_name] = [
                    result.metrics[primary_metric] for result in benchmark.results
                    if primary_metric in result.metrics
                ]
        
        if len(algorithm_results) < 2:
            logger.warning(f"Insufficient data for comparison on {primary_metric}")
            return comparison_results
        
        # Pairwise comparisons
        algorithm_names = list(algorithm_results.keys())
        p_values = []
        
        for i, alg1 in enumerate(algorithm_names):
            for j, alg2 in enumerate(algorithm_names[i+1:], i+1):
                
                results1 = algorithm_results[alg1]
                results2 = algorithm_results[alg2]
                
                # Ensure same number of runs
                min_runs = min(len(results1), len(results2))
                results1 = results1[:min_runs]
                results2 = results2[:min_runs]
                
                # Paired t-test
                test_result = self.analyzer.paired_t_test(results1, results2)
                
                comparison_key = f"{alg1}_vs_{alg2}"
                comparison_results["pairwise_comparisons"][comparison_key] = test_result
                p_values.append(test_result["p_value"])
        
        # Multiple comparison correction
        if self.config.enable_statistical_tests and p_values:
            rejected, adjusted_p = self.analyzer.multiple_comparison_correction(
                p_values, self.config.multiple_comparison_correction
            )
            
            # Update significance after correction
            comparison_idx = 0
            for i, alg1 in enumerate(algorithm_names):
                for j, alg2 in enumerate(algorithm_names[i+1:], i+1):
                    comparison_key = f"{alg1}_vs_{alg2}"
                    comparison_results["pairwise_comparisons"][comparison_key]["adjusted_p_value"] = adjusted_p[comparison_idx]
                    comparison_results["pairwise_comparisons"][comparison_key]["significant_after_correction"] = rejected[comparison_idx]
                    comparison_idx += 1
        
        # Overall ANOVA
        if len(algorithm_results) > 2:
            anova_groups = list(algorithm_results.values())
            anova_result = self.analyzer.one_way_anova(*anova_groups)
            comparison_results["overall_anova"] = anova_result
        
        # Rankings
        mean_scores = {
            alg: np.mean(results) for alg, results in algorithm_results.items()
        }
        
        # Rank from best to worst (assuming higher is better)
        ranked_algorithms = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
        comparison_results["rankings"] = {
            "by_mean_score": ranked_algorithms,
            "score_differences": {
                alg: mean_scores[ranked_algorithms[0][0]] - score
                for alg, score in mean_scores.items()
            }
        }
        
        return comparison_results
    
    def _save_intermediate_result(
        self, 
        algorithm_name: str, 
        run_id: int, 
        result: ExperimentResult
    ):
        """Save intermediate result for single run."""
        
        filename = self.experiment_dir / f"{algorithm_name}_run_{run_id}.json"
        
        result_dict = {
            "run_id": result.run_id,
            "metrics": result.metrics,
            "training_time": result.training_time,
            "memory_usage": result.memory_usage,
            "model_size": result.model_size,
            "convergence_epoch": result.convergence_epoch,
            "additional_info": result.additional_info
        }
        
        with open(filename, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    def _save_benchmark_result(self, benchmark_result: BenchmarkResult):
        """Save complete benchmark result."""
        
        filename = self.experiment_dir / f"{benchmark_result.algorithm_name}_benchmark.json"
        
        # Convert to serializable format
        benchmark_dict = {
            "experiment_name": benchmark_result.experiment_name,
            "algorithm_name": benchmark_result.algorithm_name,
            "mean_metrics": benchmark_result.mean_metrics,
            "std_metrics": benchmark_result.std_metrics,
            "confidence_intervals": {
                metric: {"lower": ci[0], "upper": ci[1]}
                for metric, ci in benchmark_result.confidence_intervals.items()
            },
            "statistical_significance": benchmark_result.statistical_significance,
            "effect_sizes": benchmark_result.effect_sizes,
            "num_runs": len(benchmark_result.results),
            "config": self.config.__dict__
        }
        
        with open(filename, 'w') as f:
            json.dump(benchmark_dict, f, indent=2)
        
        # Also save raw results
        results_filename = self.experiment_dir / f"{benchmark_result.algorithm_name}_raw_results.pkl"
        with open(results_filename, 'wb') as f:
            pickle.dump(benchmark_result.results, f)


class VisualizationGenerator:
    """Generate publication-ready visualizations for experimental results."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for publication quality
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_algorithm_comparison(
        self, 
        benchmark_results: List[BenchmarkResult],
        metric: str = "accuracy",
        save_name: str = "algorithm_comparison"
    ):
        """Create box plot comparing algorithms."""
        
        # Prepare data
        data = []
        for benchmark in benchmark_results:
            for result in benchmark.results:
                if metric in result.metrics:
                    data.append({
                        'Algorithm': benchmark.algorithm_name,
                        'Score': result.metrics[metric],
                        'Run': result.run_id
                    })
        
        if not data:
            logger.warning(f"No data available for metric: {metric}")
            return
        
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Box plot with individual points
        ax = sns.boxplot(data=df, x='Algorithm', y='Score', showfliers=False)
        sns.stripplot(data=df, x='Algorithm', y='Score', color='black', alpha=0.6, size=4)
        
        plt.title(f'{metric.title()} Comparison Across Algorithms', fontsize=16, fontweight='bold')
        plt.xlabel('Algorithm', fontsize=14)
        plt.ylabel(f'{metric.title()}', fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.output_dir / f"{save_name}_{metric}.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f"{save_name}_{metric}.pdf", bbox_inches='tight')
        plt.close()
    
    def plot_convergence_analysis(
        self,
        training_histories: Dict[str, List[List[float]]],
        save_name: str = "convergence_analysis"
    ):
        """Plot training convergence for different algorithms."""
        
        plt.figure(figsize=(12, 8))
        
        for algorithm_name, histories in training_histories.items():
            # Compute mean and std across runs
            max_length = max(len(history) for history in histories)
            
            # Pad histories to same length
            padded_histories = []
            for history in histories:
                padded = history + [history[-1]] * (max_length - len(history))
                padded_histories.append(padded)
            
            histories_array = np.array(padded_histories)
            mean_history = np.mean(histories_array, axis=0)
            std_history = np.std(histories_array, axis=0)
            
            epochs = range(1, len(mean_history) + 1)
            
            # Plot mean with confidence band
            plt.plot(epochs, mean_history, label=algorithm_name, linewidth=2)
            plt.fill_between(
                epochs, 
                mean_history - std_history, 
                mean_history + std_history, 
                alpha=0.2
            )
        
        plt.title('Training Convergence Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f"{save_name}.pdf", bbox_inches='tight')
        plt.close()
    
    def plot_performance_vs_efficiency(
        self,
        benchmark_results: List[BenchmarkResult],
        performance_metric: str = "accuracy",
        efficiency_metric: str = "training_time",
        save_name: str = "performance_efficiency"
    ):
        """Plot performance vs efficiency trade-off."""
        
        plt.figure(figsize=(10, 8))
        
        for benchmark in benchmark_results:
            # Extract data
            performance_scores = []
            efficiency_scores = []
            
            for result in benchmark.results:
                if performance_metric in result.metrics:
                    performance_scores.append(result.metrics[performance_metric])
                    
                    if efficiency_metric == "training_time":
                        efficiency_scores.append(result.training_time)
                    elif efficiency_metric == "memory_usage":
                        efficiency_scores.append(result.memory_usage)
                    elif efficiency_metric == "model_size":
                        efficiency_scores.append(result.model_size)
            
            if performance_scores and efficiency_scores:
                # Plot scatter with error bars
                mean_perf = np.mean(performance_scores)
                std_perf = np.std(performance_scores)
                mean_eff = np.mean(efficiency_scores)
                std_eff = np.std(efficiency_scores)
                
                plt.errorbar(
                    mean_eff, mean_perf, 
                    xerr=std_eff, yerr=std_perf,
                    fmt='o', markersize=8, capsize=5,
                    label=benchmark.algorithm_name
                )
                
                # Add algorithm name near point
                plt.annotate(
                    benchmark.algorithm_name,
                    (mean_eff, mean_perf),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10
                )
        
        plt.title(f'{performance_metric.title()} vs {efficiency_metric.replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel(f'{efficiency_metric.replace("_", " ").title()}', fontsize=14)
        plt.ylabel(f'{performance_metric.title()}', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f"{save_name}.pdf", bbox_inches='tight')
        plt.close()


def create_experimental_validation_suite(
    experiment_name: str,
    description: str,
    num_runs: int = 5,
    confidence_level: float = 0.95,
    **kwargs
) -> Tuple[ExperimentRunner, VisualizationGenerator]:
    """Factory function to create experimental validation suite."""
    
    config = ExperimentConfig(
        experiment_name=experiment_name,
        description=description,
        num_runs=num_runs,
        confidence_level=confidence_level,
        **kwargs
    )
    
    runner = ExperimentRunner(config)
    visualizer = VisualizationGenerator(runner.experiment_dir / "plots")
    
    return runner, visualizer


# Example usage and demonstration
def demonstrate_experimental_validation():
    """Demonstrate experimental validation framework."""
    
    logger.info("Demonstrating Experimental Validation Framework")
    
    print("Experimental Validation Framework Features:")
    print("✓ Controlled experiments with statistical rigor")
    print("✓ Multiple run validation with confidence intervals")
    print("✓ Paired t-tests and ANOVA for algorithm comparison")
    print("✓ Multiple comparison correction (Bonferroni, Holm, FDR)")
    print("✓ Effect size computation (Cohen's d, eta-squared)")
    print("✓ Publication-ready visualizations")
    print("✓ Reproducibility controls with random seeds")
    print("✓ Comprehensive benchmarking and statistical analysis")


if __name__ == "__main__":
    demonstrate_experimental_validation()