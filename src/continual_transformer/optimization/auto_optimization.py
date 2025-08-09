"""Automatic optimization and self-tuning for continual learning models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import logging
import numpy as np
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math

logger = logging.getLogger(__name__)

@dataclass
class OptimizationMetrics:
    """Metrics for optimization performance."""
    accuracy: float
    loss: float
    training_time: float
    memory_usage: float
    convergence_speed: float
    stability_score: float
    
    def overall_score(self) -> float:
        """Calculate overall optimization score."""
        # Weighted combination of metrics (higher is better)
        score = (
            self.accuracy * 0.3 +
            (1.0 / max(self.loss, 0.01)) * 0.2 +
            (1.0 / max(self.training_time, 0.1)) * 0.15 +
            (1.0 / max(self.memory_usage, 0.1)) * 0.15 +
            self.convergence_speed * 0.1 +
            self.stability_score * 0.1
        )
        return min(max(score, 0.0), 10.0)  # Clamp to [0, 10]

class HyperparameterOptimizer(ABC):
    """Abstract base class for hyperparameter optimization strategies."""
    
    @abstractmethod
    def suggest_hyperparameters(self, trial_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Suggest next set of hyperparameters to try."""
        pass
    
    @abstractmethod
    def update_with_result(self, hyperparams: Dict[str, Any], metrics: OptimizationMetrics):
        """Update optimizer with trial results."""
        pass

class BayesianOptimizer(HyperparameterOptimizer):
    """Bayesian optimization for hyperparameter tuning."""
    
    def __init__(self, search_space: Dict[str, Tuple[float, float]]):
        self.search_space = search_space
        self.trial_history = []
        self.best_params = None
        self.best_score = -float('inf')
        
        # Simple Gaussian Process approximation
        self.acquisition_weight = 0.1  # Exploration vs exploitation
    
    def suggest_hyperparameters(self, trial_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Suggest hyperparameters using Bayesian optimization."""
        self.trial_history = trial_history
        
        if len(trial_history) < 3:
            # Random exploration for first few trials
            return self._random_sample()
        
        # Use acquisition function to balance exploration/exploitation
        return self._acquisition_function_sample()
    
    def _random_sample(self) -> Dict[str, Any]:
        """Random sampling from search space."""
        params = {}
        for param_name, (low, high) in self.search_space.items():
            if param_name.endswith('_log'):
                # Log-scale sampling
                params[param_name.replace('_log', '')] = 10 ** np.random.uniform(np.log10(low), np.log10(high))
            else:
                params[param_name] = np.random.uniform(low, high)
        return params
    
    def _acquisition_function_sample(self) -> Dict[str, Any]:
        """Sample using acquisition function (simplified Expected Improvement)."""
        # Generate multiple candidates and select best according to acquisition function
        candidates = [self._random_sample() for _ in range(20)]
        best_candidate = None
        best_acquisition = -float('inf')
        
        for candidate in candidates:
            acquisition_value = self._compute_acquisition(candidate)
            if acquisition_value > best_acquisition:
                best_acquisition = acquisition_value
                best_candidate = candidate
        
        return best_candidate or self._random_sample()
    
    def _compute_acquisition(self, params: Dict[str, Any]) -> float:
        """Compute acquisition function value (simplified)."""
        # Estimate mean and variance based on similar past trials
        similar_trials = self._find_similar_trials(params)
        
        if not similar_trials:
            return 1.0  # High uncertainty = high acquisition value
        
        scores = [trial['score'] for trial in similar_trials]
        mean_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 1.0
        
        # Expected improvement approximation
        improvement = max(0, mean_score - self.best_score)
        acquisition = improvement + self.acquisition_weight * std_score
        
        return acquisition
    
    def _find_similar_trials(self, params: Dict[str, Any], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Find trials with similar hyperparameters."""
        similar = []
        
        for trial in self.trial_history:
            similarity = self._compute_similarity(params, trial['params'])
            if similarity > threshold:
                similar.append(trial)
        
        return similar
    
    def _compute_similarity(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> float:
        """Compute similarity between parameter sets."""
        if set(params1.keys()) != set(params2.keys()):
            return 0.0
        
        similarities = []
        for key in params1:
            if key in self.search_space:
                low, high = self.search_space[key]
                range_val = high - low
                diff = abs(params1[key] - params2[key]) / range_val
                similarities.append(1.0 - min(diff, 1.0))
        
        return np.mean(similarities) if similarities else 0.0
    
    def update_with_result(self, hyperparams: Dict[str, Any], metrics: OptimizationMetrics):
        """Update optimizer with trial results."""
        score = metrics.overall_score()
        
        trial_data = {
            'params': hyperparams.copy(),
            'metrics': metrics,
            'score': score
        }
        
        self.trial_history.append(trial_data)
        
        if score > self.best_score:
            self.best_score = score
            self.best_params = hyperparams.copy()
            logger.info(f"New best hyperparameters found with score {score:.4f}")

class AdaptiveLearningRateScheduler(_LRScheduler):
    """Adaptive learning rate scheduler that adjusts based on training progress."""
    
    def __init__(
        self,
        optimizer,
        patience: int = 10,
        factor: float = 0.5,
        min_lr: float = 1e-8,
        threshold: float = 0.01,
        monitor_metric: str = "loss"
    ):
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.threshold = threshold
        self.monitor_metric = monitor_metric
        
        self.best_metric = None
        self.bad_epochs = 0
        self.metric_history = []
        
        super(AdaptiveLearningRateScheduler, self).__init__(optimizer)
    
    def step(self, metrics: Optional[Dict[str, float]] = None):
        """Step the scheduler with current metrics."""
        if metrics is None:
            # Standard step without adaptation
            super(AdaptiveLearningRateScheduler, self).step()
            return
        
        current_metric = metrics.get(self.monitor_metric)
        if current_metric is None:
            logger.warning(f"Metric '{self.monitor_metric}' not found in metrics")
            return
        
        self.metric_history.append(current_metric)
        
        # Check if we should reduce learning rate
        if self._should_reduce_lr(current_metric):
            self._reduce_lr()
            self.bad_epochs = 0
            self.best_metric = current_metric
        elif self.best_metric is None or self._is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        
        # Also check for oscillation patterns
        if self._detect_oscillation():
            self._reduce_lr(factor=0.8)  # Gentler reduction for oscillation
            logger.info("Learning rate reduced due to oscillation detection")
    
    def _should_reduce_lr(self, current_metric: float) -> bool:
        """Check if learning rate should be reduced."""
        if self.best_metric is None:
            return False
        
        # For loss, lower is better; for accuracy, higher is better
        if self.monitor_metric in ['loss', 'val_loss']:
            improvement = self.best_metric - current_metric
        else:
            improvement = current_metric - self.best_metric
        
        return (self.bad_epochs >= self.patience and 
                improvement < self.threshold)
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self.monitor_metric in ['loss', 'val_loss']:
            return current < best - self.threshold
        else:
            return current > best + self.threshold
    
    def _reduce_lr(self, factor: Optional[float] = None):
        """Reduce learning rate."""
        factor = factor or self.factor
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = max(old_lr * factor, self.min_lr)
            param_group['lr'] = new_lr
            
            if new_lr != old_lr:
                logger.info(f"Reduced learning rate for group {i}: {old_lr:.6f} -> {new_lr:.6f}")
    
    def _detect_oscillation(self) -> bool:
        """Detect if metrics are oscillating."""
        if len(self.metric_history) < 6:
            return False
        
        recent_metrics = self.metric_history[-6:]
        
        # Check for alternating pattern
        increases = 0
        decreases = 0
        
        for i in range(1, len(recent_metrics)):
            if recent_metrics[i] > recent_metrics[i-1]:
                increases += 1
            else:
                decreases += 1
        
        # If roughly equal increases and decreases, might be oscillating
        return abs(increases - decreases) <= 1 and min(increases, decreases) >= 2

class AutoOptimizerSelector:
    """Automatically select and configure optimizers based on model and data characteristics."""
    
    def __init__(self):
        self.optimizer_configs = {
            "adamw": {
                "class": optim.AdamW,
                "default_params": {"lr": 2e-5, "weight_decay": 0.01, "betas": (0.9, 0.999)},
                "search_space": {
                    "lr_log": (1e-6, 1e-2),
                    "weight_decay": (0.0, 0.1),
                    "beta1": (0.85, 0.95),
                    "beta2": (0.99, 0.999)
                }
            },
            "adam": {
                "class": optim.Adam,
                "default_params": {"lr": 1e-3, "betas": (0.9, 0.999)},
                "search_space": {
                    "lr_log": (1e-6, 1e-2),
                    "beta1": (0.85, 0.95),
                    "beta2": (0.99, 0.999)
                }
            },
            "sgd": {
                "class": optim.SGD,
                "default_params": {"lr": 1e-2, "momentum": 0.9, "weight_decay": 1e-4},
                "search_space": {
                    "lr_log": (1e-5, 1e-1),
                    "momentum": (0.8, 0.99),
                    "weight_decay": (0.0, 1e-3)
                }
            },
            "rmsprop": {
                "class": optim.RMSprop,
                "default_params": {"lr": 1e-3, "momentum": 0.9, "alpha": 0.99},
                "search_space": {
                    "lr_log": (1e-6, 1e-2),
                    "momentum": (0.8, 0.95),
                    "alpha": (0.9, 0.999)
                }
            }
        }
    
    def select_optimizer(
        self,
        model: nn.Module,
        model_size: str = "unknown",
        task_type: str = "classification",
        data_characteristics: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Select best optimizer configuration based on model and data characteristics."""
        
        # Heuristic-based optimizer selection
        if model_size == "large" or self._count_parameters(model) > 100_000_000:
            # Large models often benefit from AdamW
            optimizer_name = "adamw"
            params = self.optimizer_configs["adamw"]["default_params"].copy()
            params["lr"] = 1e-5  # Lower learning rate for large models
            
        elif task_type == "continual_learning":
            # Continual learning often benefits from lower learning rates
            optimizer_name = "adamw"
            params = self.optimizer_configs["adamw"]["default_params"].copy()
            params["lr"] = 5e-6
            
        elif data_characteristics and data_characteristics.get("batch_size", 32) < 8:
            # Small batches often work better with Adam
            optimizer_name = "adam"
            params = self.optimizer_configs["adam"]["default_params"].copy()
            
        else:
            # Default to AdamW for transformer models
            optimizer_name = "adamw"
            params = self.optimizer_configs["adamw"]["default_params"].copy()
        
        logger.info(f"Selected optimizer: {optimizer_name} with params: {params}")
        return optimizer_name, params
    
    def create_optimizer(
        self,
        optimizer_name: str,
        model_parameters,
        params: Dict[str, Any]
    ) -> optim.Optimizer:
        """Create optimizer instance."""
        if optimizer_name not in self.optimizer_configs:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        optimizer_class = self.optimizer_configs[optimizer_name]["class"]
        
        # Handle special parameter names
        processed_params = {}
        for key, value in params.items():
            if key == "beta1":
                if "betas" not in processed_params:
                    processed_params["betas"] = (value, 0.999)
                else:
                    processed_params["betas"] = (value, processed_params["betas"][1])
            elif key == "beta2":
                if "betas" not in processed_params:
                    processed_params["betas"] = (0.9, value)
                else:
                    processed_params["betas"] = (processed_params["betas"][0], value)
            else:
                processed_params[key] = value
        
        return optimizer_class(model_parameters, **processed_params)
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters in model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AutoTrainingLoop:
    """Automatic training loop with adaptive optimization."""
    
    def __init__(
        self,
        model,
        config: Optional[Dict[str, Any]] = None,
        enable_hyperparameter_optimization: bool = True
    ):
        self.model = model
        self.config = config or {}
        self.enable_hp_optimization = enable_hyperparameter_optimization
        
        # Initialize components
        self.optimizer_selector = AutoOptimizerSelector()
        
        if enable_hyperparameter_optimization:
            # Define search space for hyperparameter optimization
            search_space = {
                "lr_log": (1e-6, 1e-2),
                "weight_decay": (0.0, 0.1),
                "batch_size": (4, 64),
                "warmup_steps": (0, 1000)
            }
            self.hp_optimizer = BayesianOptimizer(search_space)
        
        self.optimization_history = []
    
    def auto_train(
        self,
        train_dataloader,
        eval_dataloader=None,
        num_epochs: int = 10,
        task_id: str = "auto_task",
        max_optimization_trials: int = 5
    ) -> Dict[str, Any]:
        """Automatic training with adaptive optimization."""
        
        if self.enable_hp_optimization and max_optimization_trials > 1:
            return self._hyperparameter_optimization_training(
                train_dataloader, eval_dataloader, num_epochs, 
                task_id, max_optimization_trials
            )
        else:
            return self._single_training_run(
                train_dataloader, eval_dataloader, num_epochs, task_id
            )
    
    def _hyperparameter_optimization_training(
        self,
        train_dataloader,
        eval_dataloader,
        num_epochs: int,
        task_id: str,
        max_trials: int
    ) -> Dict[str, Any]:
        """Training with hyperparameter optimization."""
        
        best_metrics = None
        best_hyperparams = None
        
        for trial in range(max_trials):
            logger.info(f"Starting hyperparameter optimization trial {trial + 1}/{max_trials}")
            
            # Get hyperparameters for this trial
            hyperparams = self.hp_optimizer.suggest_hyperparameters(
                self.optimization_history
            )
            
            # Train with these hyperparameters
            trial_metrics = self._single_training_run(
                train_dataloader, eval_dataloader, num_epochs, 
                task_id, hyperparams
            )
            
            # Update optimizer with results
            opt_metrics = OptimizationMetrics(
                accuracy=trial_metrics.get("final_accuracy", 0.0),
                loss=trial_metrics.get("final_loss", float('inf')),
                training_time=trial_metrics.get("training_time", 0.0),
                memory_usage=trial_metrics.get("peak_memory_mb", 0.0),
                convergence_speed=trial_metrics.get("convergence_speed", 0.0),
                stability_score=trial_metrics.get("stability_score", 0.0)
            )
            
            self.hp_optimizer.update_with_result(hyperparams, opt_metrics)
            
            # Track best results
            if best_metrics is None or opt_metrics.overall_score() > best_metrics.overall_score():
                best_metrics = opt_metrics
                best_hyperparams = hyperparams
                
            # Store trial results
            trial_result = {
                "trial": trial,
                "hyperparams": hyperparams,
                "metrics": trial_metrics,
                "score": opt_metrics.overall_score()
            }
            self.optimization_history.append(trial_result)
        
        logger.info(f"Hyperparameter optimization completed. Best score: {best_metrics.overall_score():.4f}")
        
        return {
            "best_hyperparams": best_hyperparams,
            "best_metrics": best_metrics,
            "optimization_history": self.optimization_history,
            "num_trials": max_trials
        }
    
    def _single_training_run(
        self,
        train_dataloader,
        eval_dataloader,
        num_epochs: int,
        task_id: str,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Single training run with given or default hyperparameters."""
        
        start_time = time.time()
        
        # Use provided hyperparams or select automatically
        if hyperparams is None:
            optimizer_name, optimizer_params = self.optimizer_selector.select_optimizer(
                self.model, task_type="continual_learning"
            )
        else:
            optimizer_name = "adamw"  # Default
            optimizer_params = {
                "lr": hyperparams.get("lr", 2e-5),
                "weight_decay": hyperparams.get("weight_decay", 0.01)
            }
        
        # Create optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = self.optimizer_selector.create_optimizer(
            optimizer_name, trainable_params, optimizer_params
        )
        
        # Create adaptive learning rate scheduler
        scheduler = AdaptiveLearningRateScheduler(optimizer, patience=3, factor=0.7)
        
        # Training metrics tracking
        train_losses = []
        train_accuracies = []
        eval_losses = []
        eval_accuracies = []
        
        initial_memory = self._get_memory_usage()
        peak_memory = initial_memory
        
        # Training loop
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['labels'],
                    task_id=task_id
                )
                
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                predictions = outputs['logits'].argmax(dim=-1)
                epoch_correct += (predictions == batch['labels']).sum().item()
                epoch_total += batch['labels'].size(0)
                
                # Track memory usage
                current_memory = self._get_memory_usage()
                peak_memory = max(peak_memory, current_memory)
            
            # Epoch metrics
            epoch_accuracy = epoch_correct / epoch_total
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            
            train_losses.append(avg_epoch_loss)
            train_accuracies.append(epoch_accuracy)
            
            # Evaluation
            if eval_dataloader is not None:
                eval_metrics = self._evaluate(eval_dataloader, task_id)
                eval_losses.append(eval_metrics['loss'])
                eval_accuracies.append(eval_metrics['accuracy'])
                
                # Update scheduler
                scheduler.step({"loss": eval_metrics['loss']})
            else:
                scheduler.step({"loss": avg_epoch_loss})
            
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Loss: {avg_epoch_loss:.4f}, Acc: {epoch_accuracy:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
        
        total_time = time.time() - start_time
        
        # Calculate additional metrics
        convergence_speed = self._calculate_convergence_speed(train_losses)
        stability_score = self._calculate_stability_score(train_accuracies)
        
        return {
            "final_accuracy": train_accuracies[-1] if train_accuracies else 0.0,
            "final_loss": train_losses[-1] if train_losses else float('inf'),
            "training_time": total_time,
            "peak_memory_mb": peak_memory,
            "convergence_speed": convergence_speed,
            "stability_score": stability_score,
            "train_losses": train_losses,
            "train_accuracies": train_accuracies,
            "eval_losses": eval_losses,
            "eval_accuracies": eval_accuracies,
            "optimizer_config": {
                "name": optimizer_name,
                "params": optimizer_params
            }
        }
    
    def _evaluate(self, eval_dataloader, task_id: str) -> Dict[str, float]:
        """Evaluate model on eval dataset."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['labels'],
                    task_id=task_id
                )
                
                total_loss += outputs['loss'].item()
                predictions = outputs['logits'].argmax(dim=-1)
                total_correct += (predictions == batch['labels']).sum().item()
                total_samples += batch['labels'].size(0)
        
        self.model.train()
        
        return {
            "loss": total_loss / len(eval_dataloader),
            "accuracy": total_correct / total_samples
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            # Placeholder for CPU memory tracking
            return 0.0
    
    def _calculate_convergence_speed(self, losses: List[float]) -> float:
        """Calculate convergence speed based on loss reduction."""
        if len(losses) < 2:
            return 0.0
        
        initial_loss = losses[0]
        final_loss = losses[-1]
        
        if initial_loss <= final_loss:
            return 0.0
        
        # Convergence speed as relative improvement per epoch
        relative_improvement = (initial_loss - final_loss) / initial_loss
        speed = relative_improvement / len(losses)
        
        return min(speed, 1.0)  # Cap at 1.0
    
    def _calculate_stability_score(self, accuracies: List[float]) -> float:
        """Calculate training stability score based on accuracy variance."""
        if len(accuracies) < 2:
            return 1.0
        
        # Stability as inverse of relative standard deviation
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        if mean_acc == 0:
            return 0.0
        
        coefficient_of_variation = std_acc / mean_acc
        stability = 1.0 / (1.0 + coefficient_of_variation)
        
        return min(stability, 1.0)

__all__ = [
    "OptimizationMetrics",
    "HyperparameterOptimizer", 
    "BayesianOptimizer",
    "AdaptiveLearningRateScheduler",
    "AutoOptimizerSelector",
    "AutoTrainingLoop"
]