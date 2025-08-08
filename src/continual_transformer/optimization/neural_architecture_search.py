"""Neural Architecture Search for optimal adapter configurations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import random
import logging
from dataclasses import dataclass
import copy
from collections import defaultdict
import time
import json

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureCandidate:
    """Represents a candidate architecture configuration."""
    config: Dict[str, Any]
    performance: float
    efficiency: float
    complexity: int
    training_time: float
    memory_usage: float
    
    @property
    def score(self) -> float:
        """Compute overall architecture score."""
        # Weighted combination of performance and efficiency
        return 0.6 * self.performance + 0.3 * self.efficiency - 0.1 * (self.complexity / 1000.0)


class AdapterSearchSpace:
    """Defines the search space for adapter architectures."""
    
    def __init__(self):
        self.search_dimensions = {
            'adapter_type': ['activation', 'multi_layer', 'attention', 'lora', 'adaptive'],
            'adapter_size': [16, 32, 64, 128, 256],
            'num_layers': [1, 2, 3, 4],
            'activation_function': ['gelu', 'relu', 'tanh', 'swish'],
            'dropout_prob': [0.0, 0.1, 0.2, 0.3],
            'normalization': ['layer_norm', 'batch_norm', 'group_norm', 'none'],
            'residual_connection': [True, False],
            'attention_heads': [4, 8, 12, 16],  # For attention adapters
            'rank': [8, 16, 32, 64],  # For LoRA adapters
            'experts': [2, 4, 8],  # For adaptive adapters
        }
        
        # Constraints to ensure valid configurations
        self.constraints = {
            'attention': {
                'adapter_size': lambda x: x % self.search_dimensions['attention_heads'][0] == 0
            },
            'lora': {
                'rank': lambda x: x <= 128
            }
        }
    
    def sample_architecture(self) -> Dict[str, Any]:
        """Sample a random architecture configuration."""
        
        config = {}
        
        # Sample adapter type first as it affects other choices
        adapter_type = random.choice(self.search_dimensions['adapter_type'])
        config['adapter_type'] = adapter_type
        
        # Sample other dimensions
        for dim, choices in self.search_dimensions.items():
            if dim != 'adapter_type':
                config[dim] = random.choice(choices)
        
        # Apply type-specific constraints
        if adapter_type in self.constraints:
            for param, constraint in self.constraints[adapter_type].items():
                if param in config:
                    # Find valid value that satisfies constraint
                    valid_choices = [c for c in self.search_dimensions[param] if constraint(c)]
                    if valid_choices:
                        config[param] = random.choice(valid_choices)
        
        # Remove irrelevant parameters for specific adapter types
        if adapter_type != 'attention':
            config.pop('attention_heads', None)
        if adapter_type != 'lora':
            config.pop('rank', None)
        if adapter_type != 'adaptive':
            config.pop('experts', None)
        
        return config
    
    def mutate_architecture(self, config: Dict[str, Any], mutation_rate: float = 0.3) -> Dict[str, Any]:
        """Mutate an existing architecture configuration."""
        
        mutated_config = copy.deepcopy(config)
        
        for param, value in mutated_config.items():
            if random.random() < mutation_rate and param in self.search_dimensions:
                # Mutate this parameter
                choices = self.search_dimensions[param]
                new_value = random.choice(choices)
                mutated_config[param] = new_value
        
        return mutated_config
    
    def crossover_architectures(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Create offspring by crossing over two parent architectures."""
        
        child = {}
        
        for param in set(parent1.keys()) | set(parent2.keys()):
            # Randomly choose from which parent to inherit
            if param in parent1 and param in parent2:
                child[param] = random.choice([parent1[param], parent2[param]])
            elif param in parent1:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        
        return child


class NASOptimizer:
    """Neural Architecture Search optimizer for adapter configurations."""
    
    def __init__(self, model, config, search_strategy='evolutionary'):
        self.model = model
        self.config = config
        self.search_strategy = search_strategy
        
        # Search components
        self.search_space = AdapterSearchSpace()
        self.population = []
        self.population_size = 20
        self.num_generations = 10
        
        # Performance tracking
        self.search_history = []
        self.best_architectures = []
        
        # Evaluation cache
        self.evaluation_cache = {}
        
    def search_optimal_architecture(self, task_id: str, train_data, val_data) -> Dict[str, Any]:
        """Search for optimal adapter architecture for a specific task."""
        
        logger.info(f"Starting NAS for task {task_id} using {self.search_strategy} strategy")
        
        if self.search_strategy == 'evolutionary':
            return self._evolutionary_search(task_id, train_data, val_data)
        elif self.search_strategy == 'random':
            return self._random_search(task_id, train_data, val_data)
        elif self.search_strategy == 'bayesian':
            return self._bayesian_optimization(task_id, train_data, val_data)
        else:
            raise ValueError(f"Unknown search strategy: {self.search_strategy}")
    
    def _evolutionary_search(self, task_id: str, train_data, val_data) -> Dict[str, Any]:
        """Evolutionary algorithm for architecture search."""
        
        # Initialize population
        self.population = []
        for _ in range(self.population_size):
            config = self.search_space.sample_architecture()
            candidate = self._evaluate_architecture(config, task_id, train_data, val_data)
            if candidate is not None:
                self.population.append(candidate)
        
        best_candidate = None
        
        # Evolution loop
        for generation in range(self.num_generations):
            logger.info(f"Generation {generation + 1}/{self.num_generations}")
            
            # Selection
            self.population.sort(key=lambda x: x.score, reverse=True)
            survivors = self.population[:self.population_size // 2]
            
            # Track best candidate
            if not best_candidate or survivors[0].score > best_candidate.score:
                best_candidate = survivors[0]
            
            # Reproduction
            new_population = survivors.copy()
            
            while len(new_population) < self.population_size:
                # Tournament selection for parents
                parent1 = self._tournament_selection(survivors)
                parent2 = self._tournament_selection(survivors)
                
                # Crossover
                if random.random() < 0.8:  # Crossover probability
                    child_config = self.search_space.crossover_architectures(
                        parent1.config, parent2.config
                    )
                else:
                    child_config = random.choice([parent1.config, parent2.config])
                
                # Mutation
                if random.random() < 0.2:  # Mutation probability
                    child_config = self.search_space.mutate_architecture(child_config)
                
                # Evaluate child
                child_candidate = self._evaluate_architecture(child_config, task_id, train_data, val_data)
                if child_candidate is not None:
                    new_population.append(child_candidate)
            
            self.population = new_population
            
            # Log generation statistics
            scores = [c.score for c in self.population]
            logger.info(
                f"Generation {generation + 1}: "
                f"Best: {max(scores):.4f}, "
                f"Mean: {np.mean(scores):.4f}, "
                f"Std: {np.std(scores):.4f}"
            )
        
        # Record search results
        self.search_history.append({
            'task_id': task_id,
            'strategy': 'evolutionary',
            'best_architecture': best_candidate.config if best_candidate else None,
            'best_score': best_candidate.score if best_candidate else 0.0,
            'generations': self.num_generations
        })
        
        return best_candidate.config if best_candidate else self.search_space.sample_architecture()
    
    def _random_search(self, task_id: str, train_data, val_data) -> Dict[str, Any]:
        """Random search for architecture optimization."""
        
        num_trials = 50
        best_candidate = None
        
        for trial in range(num_trials):
            config = self.search_space.sample_architecture()
            candidate = self._evaluate_architecture(config, task_id, train_data, val_data)
            
            if candidate is not None:
                if best_candidate is None or candidate.score > best_candidate.score:
                    best_candidate = candidate
            
            if (trial + 1) % 10 == 0:
                logger.info(f"Random search trial {trial + 1}/{num_trials}")
        
        # Record search results
        self.search_history.append({
            'task_id': task_id,
            'strategy': 'random',
            'best_architecture': best_candidate.config if best_candidate else None,
            'best_score': best_candidate.score if best_candidate else 0.0,
            'trials': num_trials
        })
        
        return best_candidate.config if best_candidate else self.search_space.sample_architecture()
    
    def _bayesian_optimization(self, task_id: str, train_data, val_data) -> Dict[str, Any]:
        """Bayesian optimization for architecture search."""
        
        # Simplified Bayesian optimization
        # In practice, would use libraries like GPyOpt or Optuna
        
        num_initial_samples = 10
        num_iterations = 20
        
        # Initial random sampling
        candidates = []
        for _ in range(num_initial_samples):
            config = self.search_space.sample_architecture()
            candidate = self._evaluate_architecture(config, task_id, train_data, val_data)
            if candidate is not None:
                candidates.append(candidate)
        
        best_candidate = max(candidates, key=lambda x: x.score) if candidates else None
        
        # Iterative improvement
        for iteration in range(num_iterations):
            # Acquisition function: Upper Confidence Bound
            next_config = self._acquisition_function(candidates)
            next_candidate = self._evaluate_architecture(next_config, task_id, train_data, val_data)
            
            if next_candidate is not None:
                candidates.append(next_candidate)
                if next_candidate.score > best_candidate.score:
                    best_candidate = next_candidate
            
            if (iteration + 1) % 5 == 0:
                logger.info(f"Bayesian optimization iteration {iteration + 1}/{num_iterations}")
        
        # Record search results
        self.search_history.append({
            'task_id': task_id,
            'strategy': 'bayesian',
            'best_architecture': best_candidate.config if best_candidate else None,
            'best_score': best_candidate.score if best_candidate else 0.0,
            'iterations': num_iterations
        })
        
        return best_candidate.config if best_candidate else self.search_space.sample_architecture()
    
    def _tournament_selection(self, population: List[ArchitectureCandidate]) -> ArchitectureCandidate:
        """Tournament selection for evolutionary algorithm."""
        
        tournament_size = 3
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.score)
    
    def _acquisition_function(self, candidates: List[ArchitectureCandidate]) -> Dict[str, Any]:
        """Acquisition function for Bayesian optimization."""
        
        # Simplified acquisition: exploration vs exploitation
        if random.random() < 0.3:  # Exploration
            return self.search_space.sample_architecture()
        else:  # Exploitation - mutate best configuration
            best_candidate = max(candidates, key=lambda x: x.score)
            return self.search_space.mutate_architecture(best_candidate.config, mutation_rate=0.2)
    
    def _evaluate_architecture(
        self, 
        config: Dict[str, Any], 
        task_id: str, 
        train_data, 
        val_data
    ) -> Optional[ArchitectureCandidate]:
        """Evaluate a candidate architecture."""
        
        # Check cache first
        config_key = json.dumps(config, sort_keys=True)
        if config_key in self.evaluation_cache:
            return self.evaluation_cache[config_key]
        
        try:
            # Create adapter with given configuration
            adapter = self._create_adapter_from_config(config)
            
            # Measure training time and memory
            start_time = time.time()
            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Train adapter briefly to get performance estimate
            performance_metrics = self._quick_train_and_evaluate(
                adapter, task_id, train_data, val_data
            )
            
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Calculate metrics
            training_time = end_time - start_time
            memory_usage = (end_memory - start_memory) / 1024 / 1024  # MB
            complexity = self._calculate_complexity(config)
            
            # Create candidate
            candidate = ArchitectureCandidate(
                config=config,
                performance=performance_metrics['accuracy'],
                efficiency=1.0 / max(training_time, 0.1),  # Inverse of training time
                complexity=complexity,
                training_time=training_time,
                memory_usage=memory_usage
            )
            
            # Cache result
            self.evaluation_cache[config_key] = candidate
            
            return candidate
            
        except Exception as e:
            logger.warning(f"Failed to evaluate architecture {config}: {e}")
            return None
    
    def _create_adapter_from_config(self, config: Dict[str, Any]) -> nn.Module:
        """Create an adapter instance from configuration."""
        
        adapter_type = config['adapter_type']
        
        # Import adapter classes
        from ..adapters.activation import create_adapter
        
        # Map configuration to adapter parameters
        adapter_kwargs = {
            'hidden_size': self.config.hidden_size,
            'adapter_size': config.get('adapter_size', 64),
            'dropout_prob': config.get('dropout_prob', 0.1),
            'activation_function': config.get('activation_function', 'gelu')
        }
        
        # Add type-specific parameters
        if adapter_type == 'multi_layer':
            adapter_kwargs['num_adapter_layers'] = config.get('num_layers', 2)
        elif adapter_type == 'attention':
            adapter_kwargs['num_attention_heads'] = config.get('attention_heads', 8)
        elif adapter_type == 'lora':
            adapter_kwargs['rank'] = config.get('rank', 16)
        elif adapter_type == 'adaptive':
            adapter_kwargs['num_expert_layers'] = config.get('experts', 4)
        
        return create_adapter(adapter_type, **adapter_kwargs)
    
    def _quick_train_and_evaluate(
        self, 
        adapter: nn.Module, 
        task_id: str, 
        train_data, 
        val_data
    ) -> Dict[str, float]:
        """Quickly train and evaluate an adapter to estimate performance."""
        
        # Replace current adapter temporarily
        original_adapter = self.model.adapters.get(task_id)
        self.model.adapters[task_id] = adapter.to(self.config.device)
        
        # Quick training (few steps)
        adapter.train()
        optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
        
        num_train_steps = 10
        step_count = 0
        
        for batch in train_data:
            if step_count >= num_train_steps:
                break
            
            optimizer.zero_grad()
            
            try:
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.config.device) if hasattr(v, 'to') else v 
                             for k, v in batch.items()}
                    outputs = self.model(**inputs)
                else:
                    inputs = batch[0].to(self.config.device)
                    labels = batch[1].to(self.config.device)
                    outputs = self.model(inputs, labels=labels, task_id=task_id)
                
                if 'loss' in outputs:
                    loss = outputs['loss']
                    loss.backward()
                    optimizer.step()
                
                step_count += 1
                
            except Exception as e:
                logger.warning(f"Training step failed: {e}")
                break
        
        # Quick evaluation
        adapter.eval()
        correct = 0
        total = 0
        eval_steps = 0
        max_eval_steps = 5
        
        with torch.no_grad():
            for batch in val_data:
                if eval_steps >= max_eval_steps:
                    break
                
                try:
                    if isinstance(batch, dict):
                        inputs = {k: v.to(self.config.device) if hasattr(v, 'to') else v 
                                 for k, v in batch.items() if k != 'labels'}
                        labels = batch.get('labels')
                        outputs = self.model(**inputs)
                    else:
                        inputs = batch[0].to(self.config.device)
                        labels = batch[1].to(self.config.device)
                        outputs = self.model(inputs, task_id=task_id)
                    
                    if 'logits' in outputs and labels is not None:
                        predictions = outputs['logits'].argmax(dim=-1)
                        correct += (predictions == labels.to(self.config.device)).sum().item()
                        total += labels.size(0)
                    
                    eval_steps += 1
                    
                except Exception as e:
                    logger.warning(f"Evaluation step failed: {e}")
                    break
        
        # Restore original adapter
        if original_adapter is not None:
            self.model.adapters[task_id] = original_adapter
        
        accuracy = correct / max(total, 1)
        return {'accuracy': accuracy}
    
    def _calculate_complexity(self, config: Dict[str, Any]) -> int:
        """Calculate architecture complexity score."""
        
        complexity = 0
        
        # Base complexity from adapter size
        complexity += config.get('adapter_size', 64)
        
        # Additional complexity from layers
        complexity += config.get('num_layers', 1) * 100
        
        # Attention complexity
        if config.get('adapter_type') == 'attention':
            complexity += config.get('attention_heads', 8) * 50
        
        # Expert complexity for adaptive adapters
        if config.get('adapter_type') == 'adaptive':
            complexity += config.get('experts', 4) * 200
        
        return complexity


class AdapterArchitectureSearch:
    """Specialized architecture search for adapter components."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.performance_predictor = None
        
    def build_performance_predictor(self, architecture_data: List[Dict]) -> nn.Module:
        """Build a neural network to predict architecture performance."""
        
        # Extract features from architecture configurations
        feature_dim = self._get_feature_dimension()
        
        predictor = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Train predictor on existing data
        if architecture_data:
            self._train_performance_predictor(predictor, architecture_data)
        
        self.performance_predictor = predictor
        return predictor
    
    def _get_feature_dimension(self) -> int:
        """Get the dimension of architecture feature vector."""
        
        # Features: adapter_type(5) + adapter_size(1) + num_layers(1) + 
        # activation(4) + dropout(1) + normalization(4) + residual(1) +
        # attention_heads(1) + rank(1) + experts(1)
        return 20
    
    def _encode_architecture(self, config: Dict[str, Any]) -> torch.Tensor:
        """Encode architecture configuration as feature vector."""
        
        features = []
        
        # One-hot encode adapter type
        adapter_types = ['activation', 'multi_layer', 'attention', 'lora', 'adaptive']
        adapter_type_vec = [0.0] * len(adapter_types)
        if config.get('adapter_type') in adapter_types:
            idx = adapter_types.index(config['adapter_type'])
            adapter_type_vec[idx] = 1.0
        features.extend(adapter_type_vec)
        
        # Numerical features (normalized)
        features.append(config.get('adapter_size', 64) / 256.0)
        features.append(config.get('num_layers', 1) / 4.0)
        
        # One-hot encode activation function
        activations = ['gelu', 'relu', 'tanh', 'swish']
        activation_vec = [0.0] * len(activations)
        if config.get('activation_function') in activations:
            idx = activations.index(config['activation_function'])
            activation_vec[idx] = 1.0
        features.extend(activation_vec)
        
        # Dropout probability
        features.append(config.get('dropout_prob', 0.1))
        
        # One-hot encode normalization
        normalizations = ['layer_norm', 'batch_norm', 'group_norm', 'none']
        norm_vec = [0.0] * len(normalizations)
        if config.get('normalization') in normalizations:
            idx = normalizations.index(config['normalization'])
            norm_vec[idx] = 1.0
        features.extend(norm_vec)
        
        # Residual connection
        features.append(1.0 if config.get('residual_connection', True) else 0.0)
        
        # Optional features (set to 0 if not applicable)
        features.append(config.get('attention_heads', 0) / 16.0)
        features.append(config.get('rank', 0) / 64.0)
        features.append(config.get('experts', 0) / 8.0)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _train_performance_predictor(self, predictor: nn.Module, data: List[Dict]):
        """Train the performance predictor on architecture data."""
        
        if not data:
            return
        
        optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Prepare training data
        features = []
        targets = []
        
        for entry in data:
            if 'config' in entry and 'performance' in entry:
                feature_vec = self._encode_architecture(entry['config'])
                features.append(feature_vec)
                targets.append(entry['performance'])
        
        if not features:
            return
        
        features = torch.stack(features)
        targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        
        # Training loop
        predictor.train()
        for epoch in range(100):
            optimizer.zero_grad()
            predictions = predictor(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.debug(f"Predictor training epoch {epoch}, loss: {loss.item():.4f}")
        
        logger.info("Performance predictor training completed")
    
    def predict_architecture_performance(self, config: Dict[str, Any]) -> float:
        """Predict architecture performance using trained predictor."""
        
        if self.performance_predictor is None:
            return 0.5  # Default neutral prediction
        
        feature_vec = self._encode_architecture(config)
        
        self.performance_predictor.eval()
        with torch.no_grad():
            prediction = self.performance_predictor(feature_vec.unsqueeze(0))
        
        return prediction.item()


class TaskSpecificNAS:
    """Task-specific Neural Architecture Search for adapters."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.task_architectures = {}
        
    def search_for_task(self, task_id: str, task_data, budget: int = 50) -> Dict[str, Any]:
        """Search for optimal architecture specific to a task."""
        
        # Analyze task characteristics
        task_features = self._analyze_task_characteristics(task_data)
        
        # Generate candidate architectures based on task features
        candidates = self._generate_task_specific_candidates(task_features, budget)
        
        # Evaluate candidates
        best_architecture = None
        best_score = -1.0
        
        for candidate in candidates:
            try:
                score = self._evaluate_candidate_for_task(candidate, task_id, task_data)
                if score > best_score:
                    best_score = score
                    best_architecture = candidate
            except Exception as e:
                logger.warning(f"Failed to evaluate candidate: {e}")
        
        # Store result
        self.task_architectures[task_id] = {
            'architecture': best_architecture,
            'score': best_score,
            'task_features': task_features
        }
        
        logger.info(f"Found optimal architecture for task {task_id} with score {best_score:.4f}")
        
        return best_architecture or self._get_default_architecture()
    
    def _analyze_task_characteristics(self, task_data) -> Dict[str, float]:
        """Analyze characteristics of the task data."""
        
        characteristics = {
            'complexity': 0.5,  # Placeholder
            'data_size': 0.5,
            'class_balance': 0.5,
            'sequence_length': 0.5,
            'noise_level': 0.5
        }
        
        # Analyze actual task data
        total_samples = 0
        sequence_lengths = []
        
        for batch in task_data:
            if isinstance(batch, dict):
                if 'input_ids' in batch:
                    seq_lens = batch['input_ids'].shape[1] if batch['input_ids'].ndim > 1 else 1
                    sequence_lengths.append(seq_lens)
                total_samples += batch.get('input_ids', torch.tensor([0])).shape[0]
            else:
                total_samples += batch[0].shape[0]
                if batch[0].ndim > 1:
                    sequence_lengths.append(batch[0].shape[1])
        
        # Update characteristics based on analysis
        if sequence_lengths:
            avg_seq_len = np.mean(sequence_lengths)
            characteristics['sequence_length'] = min(avg_seq_len / 512.0, 1.0)
        
        characteristics['data_size'] = min(total_samples / 10000.0, 1.0)
        
        return characteristics
    
    def _generate_task_specific_candidates(
        self, 
        task_features: Dict[str, float], 
        budget: int
    ) -> List[Dict[str, Any]]:
        """Generate architecture candidates tailored to task characteristics."""
        
        candidates = []
        search_space = AdapterSearchSpace()
        
        # Generate candidates based on task characteristics
        for _ in range(budget):
            candidate = search_space.sample_architecture()
            
            # Adjust candidate based on task features
            candidate = self._adjust_for_task_features(candidate, task_features)
            candidates.append(candidate)
        
        return candidates
    
    def _adjust_for_task_features(
        self, 
        candidate: Dict[str, Any], 
        task_features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Adjust architecture candidate based on task features."""
        
        # High complexity tasks benefit from larger adapters
        if task_features['complexity'] > 0.7:
            candidate['adapter_size'] = max(candidate.get('adapter_size', 64), 128)
        
        # Long sequences benefit from attention mechanisms
        if task_features['sequence_length'] > 0.7:
            if random.random() < 0.3:
                candidate['adapter_type'] = 'attention'
        
        # Large datasets can handle more complex architectures
        if task_features['data_size'] > 0.8:
            candidate['num_layers'] = max(candidate.get('num_layers', 1), 2)
        
        # High noise requires more regularization
        if task_features['noise_level'] > 0.6:
            candidate['dropout_prob'] = max(candidate.get('dropout_prob', 0.1), 0.2)
        
        return candidate
    
    def _evaluate_candidate_for_task(
        self, 
        candidate: Dict[str, Any], 
        task_id: str, 
        task_data
    ) -> float:
        """Evaluate architecture candidate for specific task."""
        
        # This would involve creating and training the adapter
        # For now, return a simulated score based on architecture properties
        
        score = 0.5  # Base score
        
        # Score based on architecture properties
        adapter_size = candidate.get('adapter_size', 64)
        score += (adapter_size / 256.0) * 0.2  # Larger adapters get higher score
        
        num_layers = candidate.get('num_layers', 1)
        score += min(num_layers / 4.0, 0.2)  # More layers up to a limit
        
        # Penalize very complex architectures
        if candidate.get('adapter_type') == 'adaptive':
            experts = candidate.get('experts', 4)
            if experts > 6:
                score -= 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _get_default_architecture(self) -> Dict[str, Any]:
        """Get default architecture configuration."""
        
        return {
            'adapter_type': 'activation',
            'adapter_size': 64,
            'num_layers': 2,
            'activation_function': 'gelu',
            'dropout_prob': 0.1,
            'normalization': 'layer_norm',
            'residual_connection': True
        }
    
    def get_architecture_for_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the optimal architecture for a specific task."""
        
        if task_id in self.task_architectures:
            return self.task_architectures[task_id]['architecture']
        
        return None
    
    def transfer_architecture(self, source_task_id: str, target_task_id: str) -> Optional[Dict[str, Any]]:
        """Transfer architecture from source task to target task."""
        
        source_arch = self.get_architecture_for_task(source_task_id)
        if source_arch is None:
            return None
        
        # Create modified architecture for target task
        target_arch = copy.deepcopy(source_arch)
        
        # Apply small modifications for the new task
        search_space = AdapterSearchSpace()
        target_arch = search_space.mutate_architecture(target_arch, mutation_rate=0.1)
        
        self.task_architectures[target_task_id] = {
            'architecture': target_arch,
            'score': 0.8,  # Assume good transfer performance
            'transferred_from': source_task_id
        }
        
        logger.info(f"Transferred architecture from {source_task_id} to {target_task_id}")
        
        return target_arch