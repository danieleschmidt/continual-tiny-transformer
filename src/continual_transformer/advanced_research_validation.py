"""
Advanced Research Validation Framework for Continual Learning

This module provides comprehensive experimental validation capabilities including:
- Statistical significance testing for continual learning research
- Reproducible benchmarking with proper experimental controls
- Publication-ready evaluation metrics and reporting
- Comparative analysis framework for novel algorithms
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from scipy import stats
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    # Create dummy classes for type hints
    plt = None
    sns = None
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import defaultdict
import time
import psutil
import gc

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Container for experimental results with statistical validation."""
    task_id: str
    method: str
    accuracy: float
    forgetting: float
    inference_time: float
    memory_usage: float
    parameters: int
    run_id: int
    timestamp: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'method': self.method,
            'accuracy': self.accuracy,
            'forgetting': self.forgetting,
            'inference_time': self.inference_time,
            'memory_usage': self.memory_usage,
            'parameters': self.parameters,
            'run_id': self.run_id,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class StatisticalTestResult:
    """Results of statistical significance testing."""
    method_a: str
    method_b: str
    metric: str
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    test_statistic: float
    test_name: str
    
    def __str__(self) -> str:
        significance = "✅ Significant" if self.is_significant else "❌ Not significant"
        return (
            f"{self.method_a} vs {self.method_b} ({self.metric}): "
            f"p={self.p_value:.4f}, effect_size={self.effect_size:.4f} - {significance}"
        )


class AdvancedResearchValidator:
    """
    Comprehensive validation framework for continual learning research.
    
    Features:
    - Multiple run experimental validation with statistical testing
    - Memory and computational efficiency analysis
    - Catastrophic forgetting quantification
    - Publication-ready benchmarking reports
    - Reproducible experimental protocols
    """
    
    def __init__(self, output_dir: str = "research_results", seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.results = []
        self.experiment_metadata = {}
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def run_controlled_experiment(
        self,
        models: Dict[str, Any],
        tasks: List[Dict[str, Any]], 
        num_runs: int = 5,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, List[ExperimentResult]]:
        """
        Run controlled experiment with multiple methods and statistical validation.
        
        Args:
            models: Dictionary of {method_name: model_instance}
            tasks: List of task configurations
            num_runs: Number of independent runs for statistical validation
            metrics: List of metrics to evaluate
            
        Returns:
            Dictionary of results grouped by method
        """
        if metrics is None:
            metrics = ['accuracy', 'forgetting', 'inference_time', 'memory_usage']
        
        logger.info(f"Starting controlled experiment with {len(models)} methods, "
                   f"{len(tasks)} tasks, {num_runs} runs")
        
        results_by_method = defaultdict(list)
        
        for run_id in range(num_runs):
            logger.info(f"Starting experimental run {run_id + 1}/{num_runs}")
            
            # Reset seeds for each run to ensure independence
            run_seed = self.seed + run_id * 1000
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            
            for method_name, model in models.items():
                logger.info(f"Evaluating method: {method_name}")
                
                # Create fresh model instance for each run
                if hasattr(model, 'reset') or hasattr(model, '__init__'):
                    try:
                        # Reset model state if possible
                        if hasattr(model, 'reset'):
                            model.reset()
                        
                        results = self._evaluate_method_on_tasks(
                            model, method_name, tasks, run_id, metrics
                        )
                        results_by_method[method_name].extend(results)
                        
                    except Exception as e:
                        logger.error(f"Error evaluating {method_name} in run {run_id}: {e}")
                        continue
            
            # Memory cleanup between runs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Store results for analysis
        self.results.extend([
            result for method_results in results_by_method.values() 
            for result in method_results
        ])
        
        logger.info(f"Experiment completed. Total results: {len(self.results)}")
        return dict(results_by_method)
    
    def _evaluate_method_on_tasks(
        self,
        model: Any,
        method_name: str,
        tasks: List[Dict[str, Any]],
        run_id: int,
        metrics: List[str]
    ) -> List[ExperimentResult]:
        """Evaluate a single method on all tasks."""
        results = []
        task_accuracies = []
        
        for task_idx, task_config in enumerate(tasks):
            task_id = task_config.get('task_id', f'task_{task_idx}')
            
            try:
                # Train on current task
                if hasattr(model, 'learn_task'):
                    model.learn_task(
                        task_id=task_id,
                        train_dataloader=task_config.get('train_data'),
                        eval_dataloader=task_config.get('eval_data'),
                        **task_config.get('train_args', {})
                    )
                
                # Evaluate current task performance
                task_result = self._measure_task_performance(
                    model, task_id, task_config, method_name, run_id, metrics
                )
                results.append(task_result)
                task_accuracies.append(task_result.accuracy)
                
                # Evaluate forgetting on previous tasks
                if task_idx > 0:
                    for prev_idx in range(task_idx):
                        prev_task_id = tasks[prev_idx].get('task_id', f'task_{prev_idx}')
                        forgetting_result = self._measure_forgetting(
                            model, prev_task_id, tasks[prev_idx], 
                            task_accuracies[prev_idx], method_name, run_id, metrics
                        )
                        results.append(forgetting_result)
                
            except Exception as e:
                logger.error(f"Error evaluating task {task_id}: {e}")
                continue
        
        return results
    
    def _measure_task_performance(
        self,
        model: Any,
        task_id: str,
        task_config: Dict[str, Any],
        method_name: str,
        run_id: int,
        metrics: List[str]
    ) -> ExperimentResult:
        """Measure comprehensive performance on a single task."""
        
        # Memory before evaluation
        memory_before = self._get_memory_usage()
        
        # Time evaluation
        start_time = time.time()
        
        # Evaluate accuracy
        eval_data = task_config.get('eval_data')
        if eval_data is None:
            # Create dummy evaluation if needed
            accuracy = 0.0
        else:
            if hasattr(model, 'evaluate_task'):
                eval_metrics = model.evaluate_task(task_id, eval_data)
                accuracy = eval_metrics.get('accuracy', 0.0)
            else:
                accuracy = self._compute_accuracy(model, task_id, eval_data)
        
        inference_time = time.time() - start_time
        
        # Memory after evaluation
        memory_after = self._get_memory_usage()
        memory_usage = max(0, memory_after - memory_before)
        
        # Parameter count
        if hasattr(model, 'get_memory_usage'):
            param_info = model.get_memory_usage()
            parameters = param_info.get('total_parameters', 0)
        else:
            parameters = sum(p.numel() for p in model.parameters() if hasattr(model, 'parameters'))
        
        return ExperimentResult(
            task_id=task_id,
            method=method_name,
            accuracy=accuracy,
            forgetting=0.0,  # No forgetting on current task
            inference_time=inference_time,
            memory_usage=memory_usage,
            parameters=parameters,
            run_id=run_id,
            timestamp=time.time(),
            metadata={
                'task_type': task_config.get('task_type', 'classification'),
                'num_samples': task_config.get('num_samples', 0),
                'num_labels': task_config.get('num_labels', 2)
            }
        )
    
    def _measure_forgetting(
        self,
        model: Any,
        task_id: str,
        task_config: Dict[str, Any],
        original_accuracy: float,
        method_name: str,
        run_id: int,
        metrics: List[str]
    ) -> ExperimentResult:
        """Measure catastrophic forgetting on a previously learned task."""
        
        # Evaluate current performance on old task
        eval_data = task_config.get('eval_data')
        if eval_data is None:
            current_accuracy = 0.0
        else:
            if hasattr(model, 'evaluate_task'):
                eval_metrics = model.evaluate_task(task_id, eval_data)
                current_accuracy = eval_metrics.get('accuracy', 0.0)
            else:
                current_accuracy = self._compute_accuracy(model, task_id, eval_data)
        
        # Calculate forgetting as performance drop
        forgetting = max(0, original_accuracy - current_accuracy)
        
        return ExperimentResult(
            task_id=f"{task_id}_forgetting",
            method=method_name,
            accuracy=current_accuracy,
            forgetting=forgetting,
            inference_time=0.0,
            memory_usage=0.0,
            parameters=0,
            run_id=run_id,
            timestamp=time.time(),
            metadata={
                'original_accuracy': original_accuracy,
                'evaluation_type': 'forgetting'
            }
        )
    
    def _compute_accuracy(self, model: Any, task_id: str, eval_data) -> float:
        """Compute accuracy when model doesn't provide evaluate_task method."""
        try:
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in eval_data:
                    if hasattr(model, 'forward'):
                        outputs = model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch.get('attention_mask'),
                            task_id=task_id
                        )
                        predictions = outputs['logits'].argmax(dim=-1)
                    else:
                        # Fallback for different model interfaces
                        predictions = torch.zeros(batch['labels'].size(0), dtype=torch.long)
                    
                    correct += (predictions == batch['labels']).sum().item()
                    total += batch['labels'].size(0)
            
            return correct / max(total, 1)
        
        except Exception as e:
            logger.warning(f"Error computing accuracy: {e}")
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def compute_statistical_significance(
        self,
        results_by_method: Dict[str, List[ExperimentResult]],
        metric: str = 'accuracy',
        alpha: float = 0.05
    ) -> List[StatisticalTestResult]:
        """
        Compute statistical significance between methods using multiple tests.
        
        Args:
            results_by_method: Results grouped by method
            metric: Metric to test ('accuracy', 'forgetting', etc.)
            alpha: Significance level
            
        Returns:
            List of statistical test results
        """
        methods = list(results_by_method.keys())
        test_results = []
        
        for i, method_a in enumerate(methods):
            for method_b in methods[i+1:]:
                # Extract metric values for each method
                values_a = self._extract_metric_values(results_by_method[method_a], metric)
                values_b = self._extract_metric_values(results_by_method[method_b], metric)
                
                if len(values_a) == 0 or len(values_b) == 0:
                    continue
                
                # Perform statistical tests
                test_result = self._perform_statistical_test(
                    values_a, values_b, method_a, method_b, metric, alpha
                )
                test_results.append(test_result)
        
        return test_results
    
    def _extract_metric_values(self, results: List[ExperimentResult], metric: str) -> List[float]:
        """Extract metric values from results."""
        values = []
        for result in results:
            if hasattr(result, metric):
                values.append(getattr(result, metric))
        return values
    
    def _perform_statistical_test(
        self,
        values_a: List[float],
        values_b: List[float],
        method_a: str,
        method_b: str,
        metric: str,
        alpha: float
    ) -> StatisticalTestResult:
        """Perform statistical significance test between two methods."""
        
        try:
            # Use Welch's t-test for unequal variances
            t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
            
            # Calculate effect size (Cohen's d)
            mean_a, mean_b = np.mean(values_a), np.mean(values_b)
            std_a, std_b = np.std(values_a, ddof=1), np.std(values_b, ddof=1)
            pooled_std = np.sqrt(((len(values_a) - 1) * std_a**2 + (len(values_b) - 1) * std_b**2) / 
                                (len(values_a) + len(values_b) - 2))
            effect_size = (mean_a - mean_b) / max(pooled_std, 1e-8)
            
            # Confidence interval for difference of means
            se_diff = pooled_std * np.sqrt(1/len(values_a) + 1/len(values_b))
            df = len(values_a) + len(values_b) - 2
            t_critical = stats.t.ppf(1 - alpha/2, df)
            
            diff = mean_a - mean_b
            ci_lower = diff - t_critical * se_diff
            ci_upper = diff + t_critical * se_diff
            
            return StatisticalTestResult(
                method_a=method_a,
                method_b=method_b,
                metric=metric,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=(ci_lower, ci_upper),
                is_significant=p_value < alpha,
                test_statistic=t_stat,
                test_name="Welch's t-test"
            )
        
        except Exception as e:
            logger.error(f"Statistical test failed: {e}")
            return StatisticalTestResult(
                method_a=method_a,
                method_b=method_b,
                metric=metric,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                is_significant=False,
                test_statistic=0.0,
                test_name="Failed"
            )
    
    def generate_research_report(
        self,
        results_by_method: Dict[str, List[ExperimentResult]],
        significance_tests: List[StatisticalTestResult],
        title: str = "Continual Learning Research Results"
    ) -> str:
        """Generate publication-ready research report."""
        
        report_sections = []
        
        # Title and summary
        report_sections.append(f"# {title}\n")
        report_sections.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Experimental setup
        report_sections.append("## Experimental Setup")
        report_sections.append(f"- Methods evaluated: {len(results_by_method)}")
        report_sections.append(f"- Total experimental runs: {len(self.results)}")
        report_sections.append(f"- Random seed: {self.seed}")
        report_sections.append("")
        
        # Results summary
        report_sections.append("## Results Summary")
        
        # Create summary table
        summary_data = self._create_results_summary(results_by_method)
        for method, stats in summary_data.items():
            report_sections.append(f"### {method}")
            report_sections.append(f"- Average Accuracy: {stats['accuracy_mean']:.4f} ± {stats['accuracy_std']:.4f}")
            report_sections.append(f"- Average Forgetting: {stats['forgetting_mean']:.4f} ± {stats['forgetting_std']:.4f}")
            report_sections.append(f"- Average Inference Time: {stats['inference_time_mean']:.4f} ± {stats['inference_time_std']:.4f} seconds")
            report_sections.append("")
        
        # Statistical significance
        report_sections.append("## Statistical Significance Tests")
        for test in significance_tests:
            report_sections.append(f"- {test}")
        report_sections.append("")
        
        # Recommendations
        report_sections.append("## Recommendations")
        best_method = self._identify_best_method(summary_data)
        report_sections.append(f"- Best performing method: **{best_method}**")
        
        significant_improvements = [t for t in significance_tests if t.is_significant and t.effect_size > 0.2]
        if significant_improvements:
            report_sections.append("- Statistically significant improvements found:")
            for test in significant_improvements[:3]:  # Top 3
                report_sections.append(f"  - {test.method_a} > {test.method_b} (p={test.p_value:.4f}, d={test.effect_size:.4f})")
        
        return "\n".join(report_sections)
    
    def _create_results_summary(self, results_by_method: Dict[str, List[ExperimentResult]]) -> Dict[str, Dict[str, float]]:
        """Create statistical summary of results."""
        summary = {}
        
        for method, results in results_by_method.items():
            if not results:
                continue
            
            accuracies = [r.accuracy for r in results]
            forgetting = [r.forgetting for r in results]
            inference_times = [r.inference_time for r in results]
            
            summary[method] = {
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'forgetting_mean': np.mean(forgetting),
                'forgetting_std': np.std(forgetting),
                'inference_time_mean': np.mean(inference_times),
                'inference_time_std': np.std(inference_times),
                'num_results': len(results)
            }
        
        return summary
    
    def _identify_best_method(self, summary_data: Dict[str, Dict[str, float]]) -> str:
        """Identify the best performing method based on balanced scoring."""
        if not summary_data:
            return "None"
        
        best_method = None
        best_score = -float('inf')
        
        for method, stats in summary_data.items():
            # Balanced score: high accuracy, low forgetting, reasonable speed
            accuracy_score = stats['accuracy_mean']
            forgetting_penalty = stats['forgetting_mean']
            speed_bonus = 1.0 / max(stats['inference_time_mean'], 0.001)  # Prefer faster methods
            
            # Combine scores with weights
            combined_score = (0.6 * accuracy_score - 
                            0.3 * forgetting_penalty + 
                            0.1 * min(speed_bonus, 1.0))  # Cap speed bonus
            
            if combined_score > best_score:
                best_score = combined_score
                best_method = method
        
        return best_method or "Unknown"
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save experimental results to JSON file."""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"research_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert results to serializable format
        results_data = {
            'experiment_metadata': self.experiment_metadata,
            'results': [result.to_dict() for result in self.results],
            'timestamp': time.time(),
            'seed': self.seed
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return str(filepath)
    
    def create_visualization(
        self,
        results_by_method: Dict[str, List[ExperimentResult]],
        metrics: List[str] = None,
        save_plots: bool = True
    ) -> List[str]:
        """Create visualization plots for research results."""
        if metrics is None:
            metrics = ['accuracy', 'forgetting']
        
        plot_files = []
        
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available, skipping visualization")
            return []

        try:
            # Set up plot style
            plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
            
            for metric in metrics:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Prepare data for plotting
                method_names = []
                metric_values = []
                
                for method, results in results_by_method.items():
                    values = self._extract_metric_values(results, metric)
                    if values:
                        method_names.extend([method] * len(values))
                        metric_values.extend(values)
                
                if method_names and metric_values:
                    # Create box plot
                    unique_methods = list(set(method_names))
                    plot_data = []
                    for method in unique_methods:
                        method_values = [v for m, v in zip(method_names, metric_values) if m == method]
                        plot_data.append(method_values)
                    
                    ax.boxplot(plot_data, labels=unique_methods)
                    ax.set_title(f'{metric.capitalize()} Comparison Across Methods')
                    ax.set_ylabel(metric.capitalize())
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    if save_plots:
                        plot_filename = f"{metric}_comparison.png"
                        plot_path = self.output_dir / plot_filename
                        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plot_files.append(str(plot_path))
                        logger.info(f"Plot saved: {plot_path}")
                
                plt.close()
        
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        return plot_files


def run_comprehensive_research_validation(
    continual_transformer_class,
    config_class,
    baseline_methods: Dict[str, Any] = None,
    num_runs: int = 3
) -> Dict[str, Any]:
    """
    Run comprehensive research validation for continual transformer.
    
    This function provides a complete research validation pipeline including:
    - Multiple baseline comparisons
    - Statistical significance testing
    - Publication-ready reporting
    - Reproducible experimental protocols
    
    Args:
        continual_transformer_class: ContinualTransformer class
        config_class: ContinualConfig class
        baseline_methods: Dictionary of baseline methods to compare against
        num_runs: Number of experimental runs for statistical validation
        
    Returns:
        Dictionary containing all experimental results and analysis
    """
    
    validator = AdvancedResearchValidator(seed=42)
    
    # Create test models
    models = {}
    
    # Main continual transformer
    config = config_class()
    models['ContinualTransformer'] = continual_transformer_class(config)
    
    # Add baseline methods if provided
    if baseline_methods:
        models.update(baseline_methods)
    
    # Create synthetic tasks for testing
    tasks = create_synthetic_continual_tasks(num_tasks=3, samples_per_task=100)
    
    # Run controlled experiment
    logger.info("Starting comprehensive research validation...")
    results_by_method = validator.run_controlled_experiment(
        models=models,
        tasks=tasks,
        num_runs=num_runs
    )
    
    # Compute statistical significance
    significance_tests = validator.compute_statistical_significance(results_by_method)
    
    # Generate research report
    report = validator.generate_research_report(
        results_by_method, 
        significance_tests,
        "Continual Transformer Research Validation"
    )
    
    # Save results
    results_file = validator.save_results()
    
    # Create visualizations
    plot_files = validator.create_visualization(results_by_method)
    
    logger.info("Research validation completed successfully!")
    
    return {
        'results_by_method': results_by_method,
        'significance_tests': significance_tests,
        'report': report,
        'results_file': results_file,
        'plot_files': plot_files,
        'validator': validator
    }


def create_synthetic_continual_tasks(num_tasks: int = 3, samples_per_task: int = 100) -> List[Dict[str, Any]]:
    """Create synthetic continual learning tasks for research validation."""
    tasks = []
    
    for i in range(num_tasks):
        # Create synthetic data
        task_data = {
            'input_ids': torch.randint(0, 1000, (samples_per_task, 128)),
            'attention_mask': torch.ones(samples_per_task, 128),
            'labels': torch.randint(0, 2, (samples_per_task,))
        }
        
        # Create data loaders (simplified)
        class SimpleDataset:
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data['labels'])
            
            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.data.items()}
        
        dataset = SimpleDataset(task_data)
        
        # Split into train/eval
        train_size = int(0.8 * len(dataset))
        eval_size = len(dataset) - train_size
        
        train_data, eval_data = torch.utils.data.random_split(dataset, [train_size, eval_size])
        
        tasks.append({
            'task_id': f'synthetic_task_{i}',
            'task_type': 'classification',
            'num_labels': 2,
            'num_samples': samples_per_task,
            'train_data': torch.utils.data.DataLoader(train_data, batch_size=32),
            'eval_data': torch.utils.data.DataLoader(eval_data, batch_size=32),
            'train_args': {
                'num_epochs': 2,
                'patience': 3
            }
        })
    
    return tasks


if __name__ == "__main__":
    # Example usage for research validation
    logging.basicConfig(level=logging.INFO)
    
    # This would be run with actual models in practice
    logger.info("Advanced Research Validation Framework initialized")
    logger.info("Ready for comprehensive continual learning experiments")