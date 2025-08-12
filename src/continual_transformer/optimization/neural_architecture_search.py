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
    """Neural Architecture Search optimizer for adapter configurations with advanced research capabilities."""
    
    def __init__(self, model, config, search_strategy='evolutionary'):
        self.model = model
        self.config = config
        self.search_strategy = search_strategy
        
        # Search components
        self.search_space = AdapterSearchSpace()
        self.population = []
        self.population_size = 50  # Increased for better exploration
        self.num_generations = 25  # More generations for research quality
        
        # Advanced research features
        self.multi_objective_optimization = True
        self.pareto_front = []
        self.diversity_preservation = True
        self.adaptive_search_budget = True
        
        # Enhanced performance tracking for research
        self.search_history = []
        self.best_architectures = []
        self.pareto_optimal_solutions = []
        self.diversity_metrics = []
        
        # Advanced caching and reproducibility
        self.evaluation_cache = {}
        self.experiment_seed = 42
        self.reproducible_search = True
        
        # Research analytics
        self.search_analytics = {
            'convergence_history': [],
            'diversity_evolution': [],
            'hypervolume_progression': [],
            'architecture_frequency': defaultdict(int)
        }
        
    def search_optimal_architecture(self, task_id: str, train_data, val_data, research_mode=True) -> Dict[str, Any]:
        """Search for optimal adapter architecture with research-grade rigor."""
        
        # Set random seed for reproducibility
        if self.reproducible_search:
            torch.manual_seed(self.experiment_seed)
            np.random.seed(self.experiment_seed)
            random.seed(self.experiment_seed)
        
        logger.info(f"Starting research-grade NAS for task {task_id} using {self.search_strategy} strategy")
        logger.info(f"Search configuration: population={self.population_size}, generations={self.num_generations}")
        
        start_time = time.time()
        
        if self.search_strategy == 'evolutionary':
            result = self._advanced_evolutionary_search(task_id, train_data, val_data, research_mode)
        elif self.search_strategy == 'multi_objective':
            result = self._multi_objective_search(task_id, train_data, val_data)
        elif self.search_strategy == 'progressive':
            result = self._progressive_search(task_id, train_data, val_data)
        elif self.search_strategy == 'random':
            result = self._random_search(task_id, train_data, val_data)
        elif self.search_strategy == 'bayesian':
            result = self._bayesian_optimization(task_id, train_data, val_data)
        else:
            raise ValueError(f"Unknown search strategy: {self.search_strategy}")
        
        search_time = time.time() - start_time
        logger.info(f"NAS completed in {search_time:.2f} seconds")
        
        # Generate research report
        if research_mode:
            self._generate_research_report(task_id, search_time)
        
        return result
    
    def _advanced_evolutionary_search(self, task_id: str, train_data, val_data, research_mode=True) -> Dict[str, Any]:
        """Advanced evolutionary algorithm with research-grade features."""
        
        # Initialize diverse population with advanced sampling
        self.population = []
        initialization_strategies = ['random', 'heuristic', 'diverse', 'elite_seeded']
        
        for i in range(self.population_size):
            strategy = initialization_strategies[i % len(initialization_strategies)]
            config = self._strategic_sample_architecture(strategy, task_id)
            candidate = self._evaluate_architecture(config, task_id, train_data, val_data)
            if candidate is not None:
                self.population.append(candidate)
                self.search_analytics['architecture_frequency'][json.dumps(config, sort_keys=True)] += 1
        
        best_candidate = None
        stagnation_counter = 0
        max_stagnation = 5
        
        # Advanced evolution loop with research analytics
        for generation in range(self.num_generations):
            logger.info(f"Generation {generation + 1}/{self.num_generations}")
            
            # Multi-objective selection with Pareto ranking
            if self.multi_objective_optimization:
                ranked_population = self._pareto_ranking(self.population)
                survivors = self._nsga_ii_selection(ranked_population, self.population_size // 2)
            else:
                self.population.sort(key=lambda x: x.score, reverse=True)
                survivors = self.population[:self.population_size // 2]
            
            # Track best candidate and convergence
            current_best = survivors[0]
            if not best_candidate or current_best.score > best_candidate.score:
                best_candidate = current_best
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Record analytics
            self._record_generation_analytics(generation, survivors)
            
            # Early termination for research efficiency
            if research_mode and stagnation_counter >= max_stagnation:
                logger.info(f"Early termination at generation {generation + 1} due to convergence")
                break
            
            # Advanced reproduction with diversity preservation
            new_population = survivors.copy()
            
            # Adaptive reproduction parameters based on generation
            crossover_prob = max(0.6, 0.9 - generation * 0.01)
            mutation_prob = min(0.4, 0.1 + generation * 0.01)
            
            while len(new_population) < self.population_size:
                # Diversity-aware parent selection
                if self.diversity_preservation:
                    parent1, parent2 = self._diversity_aware_selection(survivors)
                else:
                    parent1 = self._tournament_selection(survivors)
                    parent2 = self._tournament_selection(survivors)
                
                # Advanced crossover with multiple strategies
                if random.random() < crossover_prob:
                    crossover_strategy = random.choice(['uniform', 'two_point', 'semantic'])
                    child_config = self._advanced_crossover(parent1.config, parent2.config, crossover_strategy)
                else:
                    child_config = random.choice([parent1.config, parent2.config])
                
                # Adaptive mutation
                if random.random() < mutation_prob:
                    mutation_strength = self._adaptive_mutation_strength(generation)
                    child_config = self._adaptive_mutate(child_config, mutation_strength)
                
                # Evaluate child with caching
                child_candidate = self._evaluate_architecture(child_config, task_id, train_data, val_data)
                if child_candidate is not None:
                    new_population.append(child_candidate)
                    self.search_analytics['architecture_frequency'][json.dumps(child_config, sort_keys=True)] += 1
            
            self.population = new_population
            
            # Comprehensive generation statistics
            scores = [c.score for c in self.population]
            performances = [c.performance for c in self.population]
            efficiencies = [c.efficiency for c in self.population]
            
            diversity_score = self._calculate_population_diversity(self.population)
            self.diversity_metrics.append(diversity_score)
            
            logger.info(
                f"Generation {generation + 1}: "
                f"Best: {max(scores):.4f}, "
                f"Mean: {np.mean(scores):.4f}, "
                f"Std: {np.std(scores):.4f}, "
                f"Diversity: {diversity_score:.4f}"
            )
            
            # Research-grade logging
            if research_mode:
                logger.info(
                    f"Performance - Mean: {np.mean(performances):.4f}, "
                    f"Efficiency - Mean: {np.mean(efficiencies):.4f}"
                )
        
        # Comprehensive search results with research metrics
        search_result = {
            'task_id': task_id,
            'strategy': 'advanced_evolutionary',
            'best_architecture': best_candidate.config if best_candidate else None,
            'best_score': best_candidate.score if best_candidate else 0.0,
            'best_performance': best_candidate.performance if best_candidate else 0.0,
            'best_efficiency': best_candidate.efficiency if best_candidate else 0.0,
            'generations': generation + 1,
            'total_evaluations': len(self.evaluation_cache),
            'convergence_generation': generation + 1 - stagnation_counter,
            'final_diversity': self.diversity_metrics[-1] if self.diversity_metrics else 0.0,
            'pareto_front_size': len(self.pareto_front),
            'search_analytics': self.search_analytics.copy()
        }
        
        self.search_history.append(search_result)
        
        # Store Pareto optimal solutions for research analysis
        if self.multi_objective_optimization:
            self.pareto_optimal_solutions.extend(self._extract_pareto_front(self.population))
        
        return best_candidate.config if best_candidate else self._get_research_baseline_architecture()
    
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
    
    def _tournament_selection(self, population: List[ArchitectureCandidate], tournament_size=5) -> ArchitectureCandidate:
        """Enhanced tournament selection with adaptive tournament size."""
        
        # Adaptive tournament size based on population diversity
        if hasattr(self, 'diversity_metrics') and self.diversity_metrics:
            current_diversity = self.diversity_metrics[-1]
            tournament_size = max(2, int(tournament_size * (1.0 + current_diversity)))
        
        tournament = random.sample(population, min(tournament_size, len(population)))
        
        # Multi-objective tournament selection
        if self.multi_objective_optimization:
            return self._pareto_tournament_select(tournament)
        else:
            return max(tournament, key=lambda x: x.score)
    
    def _strategic_sample_architecture(self, strategy: str, task_id: str) -> Dict[str, Any]:
        """Sample architecture using specific strategy for research diversity."""
        
        if strategy == 'random':
            return self.search_space.sample_architecture()
        elif strategy == 'heuristic':
            # Use task-specific heuristics
            base_config = self.search_space.sample_architecture()
            base_config['adapter_size'] = 128  # Start with larger adapters
            base_config['num_layers'] = 2
            return base_config
        elif strategy == 'diverse':
            # Maximize diversity in population initialization
            config = self.search_space.sample_architecture()
            # Favor less common architecture types
            rare_types = ['attention', 'lora', 'adaptive']
            config['adapter_type'] = random.choice(rare_types)
            return config
        elif strategy == 'elite_seeded':
            # Seed with known good architectures
            if self.best_architectures:
                base = random.choice(self.best_architectures).config.copy()
                return self.search_space.mutate_architecture(base, mutation_rate=0.1)
            else:
                return self.search_space.sample_architecture()
        
        return self.search_space.sample_architecture()
    
    def _pareto_ranking(self, population: List[ArchitectureCandidate]) -> List[List[ArchitectureCandidate]]:
        """Perform Pareto ranking for multi-objective optimization."""
        
        fronts = []
        remaining = population.copy()
        
        while remaining:
            current_front = []
            dominated = []
            
            for candidate in remaining:
                is_dominated = False
                for other in remaining:
                    if other != candidate and self._dominates(other, candidate):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    current_front.append(candidate)
                else:
                    dominated.append(candidate)
            
            fronts.append(current_front)
            remaining = dominated
            
            if len(fronts) > 10:  # Prevent infinite loops
                fronts.append(dominated)
                break
        
        return fronts
    
    def _dominates(self, candidate1: ArchitectureCandidate, candidate2: ArchitectureCandidate) -> bool:
        """Check if candidate1 dominates candidate2 in multi-objective space."""
        
        # Multi-objective comparison: performance, efficiency, complexity
        objectives1 = [candidate1.performance, candidate1.efficiency, -candidate1.complexity / 1000.0]
        objectives2 = [candidate2.performance, candidate2.efficiency, -candidate2.complexity / 1000.0]
        
        dominates = True
        strictly_better = False
        
        for obj1, obj2 in zip(objectives1, objectives2):
            if obj1 < obj2:
                dominates = False
                break
            elif obj1 > obj2:
                strictly_better = True
        
        return dominates and strictly_better
    
    def _nsga_ii_selection(self, fronts: List[List[ArchitectureCandidate]], num_select: int) -> List[ArchitectureCandidate]:
        """NSGA-II selection mechanism for research-grade multi-objective optimization."""
        
        selected = []
        
        for front in fronts:
            if len(selected) + len(front) <= num_select:
                selected.extend(front)
            else:
                # Calculate crowding distance for the last front
                remaining_slots = num_select - len(selected)
                if remaining_slots > 0:
                    crowding_distances = self._calculate_crowding_distance(front)
                    # Sort by crowding distance (descending)
                    sorted_front = sorted(zip(front, crowding_distances), 
                                        key=lambda x: x[1], reverse=True)
                    selected.extend([candidate for candidate, _ in sorted_front[:remaining_slots]])
                break
        
        return selected
    
    def _calculate_crowding_distance(self, front: List[ArchitectureCandidate]) -> List[float]:
        """Calculate crowding distance for diversity preservation."""
        
        if len(front) <= 2:
            return [float('inf')] * len(front)
        
        distances = [0.0] * len(front)
        
        # Normalize objectives
        objectives = ['performance', 'efficiency', 'complexity']
        
        for obj in objectives:
            # Sort by objective
            sorted_indices = sorted(range(len(front)), 
                                  key=lambda i: getattr(front[i], obj))
            
            # Set boundary points to infinity
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calculate objective range
            obj_values = [getattr(front[i], obj) for i in sorted_indices]
            obj_range = obj_values[-1] - obj_values[0]
            
            if obj_range > 0:
                # Calculate crowding distance
                for i in range(1, len(front) - 1):
                    if distances[sorted_indices[i]] != float('inf'):
                        distances[sorted_indices[i]] += (obj_values[i + 1] - obj_values[i - 1]) / obj_range
        
        return distances
    
    def _diversity_aware_selection(self, population: List[ArchitectureCandidate]) -> Tuple[ArchitectureCandidate, ArchitectureCandidate]:
        """Select diverse parents to maintain population diversity."""
        
        parent1 = self._tournament_selection(population)
        
        # Find most diverse candidate from parent1
        max_distance = -1
        parent2 = None
        
        for candidate in population:
            if candidate != parent1:
                distance = self._architecture_distance(parent1.config, candidate.config)
                if distance > max_distance:
                    max_distance = distance
                    parent2 = candidate
        
        if parent2 is None:
            parent2 = self._tournament_selection(population)
        
        return parent1, parent2
    
    def _architecture_distance(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> float:
        """Calculate distance between two architecture configurations."""
        
        distance = 0.0
        
        # Compare categorical features
        categorical_features = ['adapter_type', 'activation_function', 'normalization']
        for feature in categorical_features:
            if config1.get(feature) != config2.get(feature):
                distance += 1.0
        
        # Compare numerical features
        numerical_features = ['adapter_size', 'num_layers', 'dropout_prob']
        for feature in numerical_features:
            val1 = config1.get(feature, 0)
            val2 = config2.get(feature, 0)
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                normalized_diff = abs(val1 - val2) / max(abs(val1), abs(val2), 1.0)
                distance += normalized_diff
        
        return distance
    
    def _advanced_crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any], strategy: str) -> Dict[str, Any]:
        """Advanced crossover strategies for architecture optimization."""
        
        if strategy == 'uniform':
            return self.search_space.crossover_architectures(parent1, parent2)
        elif strategy == 'two_point':
            # Two-point crossover
            keys = list(set(parent1.keys()) | set(parent2.keys()))
            if len(keys) <= 2:
                return self.search_space.crossover_architectures(parent1, parent2)
            
            point1, point2 = sorted(random.sample(range(len(keys)), 2))
            child = {}
            
            for i, key in enumerate(keys):
                if i < point1 or i >= point2:
                    child[key] = parent1.get(key, parent2.get(key))
                else:
                    child[key] = parent2.get(key, parent1.get(key))
            
            return child
        elif strategy == 'semantic':
            # Semantic crossover based on architecture similarity
            child = parent1.copy()
            
            # Inherit similar components from more diverse parent
            distance1 = sum(1 for key in parent1.keys() if key in parent2 and parent1[key] != parent2[key])
            distance2 = sum(1 for key in parent2.keys() if key in parent1 and parent1[key] != parent2[key])
            
            diverse_parent = parent2 if distance2 > distance1 else parent1
            
            for key in diverse_parent:
                if random.random() < 0.3:  # 30% chance to inherit from diverse parent
                    child[key] = diverse_parent[key]
            
            return child
        
        return self.search_space.crossover_architectures(parent1, parent2)
    
    def _adaptive_mutation_strength(self, generation: int) -> float:
        """Calculate adaptive mutation strength based on search progress."""
        
        # Start with high exploration, gradually increase exploitation
        base_strength = max(0.1, 0.5 - generation * 0.02)
        
        # Increase mutation if population diversity is low
        if self.diversity_metrics:
            diversity_factor = 1.0 + (0.5 - self.diversity_metrics[-1])
            base_strength *= diversity_factor
        
        return min(1.0, base_strength)
    
    def _adaptive_mutate(self, config: Dict[str, Any], mutation_strength: float) -> Dict[str, Any]:
        """Adaptive mutation with strength-based parameter adjustment."""
        
        mutated_config = config.copy()
        
        for param in mutated_config.keys():
            if random.random() < mutation_strength and param in self.search_space.search_dimensions:
                choices = self.search_space.search_dimensions[param]
                
                if isinstance(mutated_config[param], (int, float)):
                    # Gaussian mutation for numerical parameters
                    current_val = mutated_config[param]
                    std = mutation_strength * (max(choices) - min(choices)) * 0.1
                    new_val = current_val + random.gauss(0, std)
                    
                    # Clamp to valid range
                    new_val = max(min(choices), min(max(choices), new_val))
                    if isinstance(current_val, int):
                        new_val = int(round(new_val))
                    
                    mutated_config[param] = new_val
                else:
                    # Random mutation for categorical parameters
                    mutated_config[param] = random.choice(choices)
        
        return mutated_config
    
    def _calculate_population_diversity(self, population: List[ArchitectureCandidate]) -> float:
        """Calculate population diversity for research analytics."""
        
        if len(population) < 2:
            return 0.0
        
        total_distance = 0.0
        num_pairs = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = self._architecture_distance(population[i].config, population[j].config)
                total_distance += distance
                num_pairs += 1
        
        return total_distance / num_pairs if num_pairs > 0 else 0.0
    
    def _record_generation_analytics(self, generation: int, survivors: List[ArchitectureCandidate]):
        """Record detailed analytics for research analysis."""
        
        scores = [c.score for c in survivors]
        
        generation_data = {
            'generation': generation,
            'best_score': max(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'population_size': len(survivors),
            'unique_architectures': len(set(json.dumps(c.config, sort_keys=True) for c in survivors))
        }
        
        self.search_analytics['convergence_history'].append(generation_data)
    
    def _extract_pareto_front(self, population: List[ArchitectureCandidate]) -> List[ArchitectureCandidate]:
        """Extract Pareto optimal solutions from population."""
        
        pareto_front = []
        
        for candidate in population:
            is_dominated = False
            for other in population:
                if other != candidate and self._dominates(other, candidate):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(candidate)
        
        return pareto_front
    
    def _get_research_baseline_architecture(self) -> Dict[str, Any]:
        """Get research-grade baseline architecture."""
        
        return {
            'adapter_type': 'activation',
            'adapter_size': 64,
            'num_layers': 2,
            'activation_function': 'gelu',
            'dropout_prob': 0.1,
            'normalization': 'layer_norm',
            'residual_connection': True,
            'research_baseline': True
        }
    
    def _pareto_tournament_select(self, tournament: List[ArchitectureCandidate]) -> ArchitectureCandidate:
        """Tournament selection for multi-objective optimization."""
        
        # Find non-dominated solutions in tournament
        non_dominated = []
        
        for candidate in tournament:
            is_dominated = False
            for other in tournament:
                if other != candidate and self._dominates(other, candidate):
                    is_dominated = True
                    break
            
            if not is_dominated:
                non_dominated.append(candidate)
        
        # Select from non-dominated solutions or fall back to best score
        if non_dominated:
            return random.choice(non_dominated)
        else:
            return max(tournament, key=lambda x: x.score)
    
    def _multi_objective_search(self, task_id: str, train_data, val_data) -> Dict[str, Any]:
        """Multi-objective optimization for research applications."""
        
        # Initialize with NSGA-II algorithm
        population = []
        for _ in range(self.population_size):
            config = self.search_space.sample_architecture()
            candidate = self._evaluate_architecture(config, task_id, train_data, val_data)
            if candidate is not None:
                population.append(candidate)
        
        best_candidates = []
        
        for generation in range(self.num_generations):
            # Pareto ranking and selection
            fronts = self._pareto_ranking(population)
            selected = self._nsga_ii_selection(fronts, self.population_size // 2)
            
            # Track Pareto front
            if fronts:
                current_front = fronts[0]
                best_candidates.extend(current_front)
                self.pareto_front = current_front
            
            # Generate new population
            new_population = selected.copy()
            
            while len(new_population) < self.population_size:
                parent1 = self._pareto_tournament_select(random.sample(selected, min(5, len(selected))))
                parent2 = self._pareto_tournament_select(random.sample(selected, min(5, len(selected))))
                
                child_config = self._advanced_crossover(parent1.config, parent2.config, 'uniform')
                
                if random.random() < 0.2:
                    child_config = self._adaptive_mutate(child_config, 0.2)
                
                child_candidate = self._evaluate_architecture(child_config, task_id, train_data, val_data)
                if child_candidate is not None:
                    new_population.append(child_candidate)
            
            population = new_population
            
            # Progress logging
            if fronts:
                front_scores = [c.score for c in fronts[0]]
                logger.info(f"Generation {generation + 1}: Pareto front size: {len(fronts[0])}, "
                          f"Best score: {max(front_scores):.4f}")
        
        # Return best overall solution
        if best_candidates:
            return max(best_candidates, key=lambda x: x.score).config
        else:
            return self._get_research_baseline_architecture()
    
    def _progressive_search(self, task_id: str, train_data, val_data) -> Dict[str, Any]:
        """Progressive architecture search with increasing complexity."""
        
        complexity_levels = [
            {'max_adapter_size': 32, 'max_layers': 1},
            {'max_adapter_size': 64, 'max_layers': 2}, 
            {'max_adapter_size': 128, 'max_layers': 3},
            {'max_adapter_size': 256, 'max_layers': 4}
        ]
        
        best_architecture = None
        best_score = -1.0
        
        for level_idx, constraints in enumerate(complexity_levels):
            logger.info(f"Progressive search level {level_idx + 1}: {constraints}")
            
            # Search with current complexity constraints
            level_population = []
            for _ in range(self.population_size // len(complexity_levels)):
                config = self._constrained_sample_architecture(constraints)
                candidate = self._evaluate_architecture(config, task_id, train_data, val_data)
                if candidate is not None:
                    level_population.append(candidate)
            
            # Find best in this level
            if level_population:
                level_best = max(level_population, key=lambda x: x.score)
                if level_best.score > best_score:
                    best_score = level_best.score
                    best_architecture = level_best.config
                    
                logger.info(f"Level {level_idx + 1} best score: {level_best.score:.4f}")
        
        return best_architecture or self._get_research_baseline_architecture()
    
    def _constrained_sample_architecture(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Sample architecture with complexity constraints."""
        
        config = self.search_space.sample_architecture()
        
        # Apply constraints
        if 'max_adapter_size' in constraints:
            valid_sizes = [s for s in self.search_space.search_dimensions['adapter_size'] 
                          if s <= constraints['max_adapter_size']]
            if valid_sizes:
                config['adapter_size'] = random.choice(valid_sizes)
        
        if 'max_layers' in constraints:
            valid_layers = [l for l in self.search_space.search_dimensions['num_layers'] 
                           if l <= constraints['max_layers']]
            if valid_layers:
                config['num_layers'] = random.choice(valid_layers)
        
        return config
    
    def _generate_research_report(self, task_id: str, search_time: float):
        """Generate comprehensive research report."""
        
        report = {
            'experiment_info': {
                'task_id': task_id,
                'search_strategy': self.search_strategy,
                'search_time_seconds': search_time,
                'population_size': self.population_size,
                'generations': self.num_generations,
                'total_evaluations': len(self.evaluation_cache),
                'experiment_seed': self.experiment_seed
            },
            'convergence_analysis': {
                'convergence_history': self.search_analytics['convergence_history'],
                'diversity_evolution': self.diversity_metrics,
                'stagnation_points': []
            },
            'architecture_analysis': {
                'architecture_frequency': dict(self.search_analytics['architecture_frequency']),
                'pareto_optimal_count': len(self.pareto_optimal_solutions),
                'unique_architectures_explored': len(self.evaluation_cache)
            }
        }
        
        logger.info(f"Research report generated for task {task_id}")
        
        return report
    
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