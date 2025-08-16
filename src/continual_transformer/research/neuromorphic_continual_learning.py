"""
Neuromorphic Continual Learning

Bio-inspired continual learning using spiking neural networks and synaptic plasticity.
Implements spike-timing dependent plasticity (STDP) for efficient continual adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import math
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class NeuromorphicConfig:
    """Configuration for neuromorphic continual learning."""
    spike_threshold: float = 1.0
    membrane_time_constant: float = 20.0  # ms
    refractory_period: float = 2.0  # ms
    stdp_learning_rate: float = 0.01
    stdp_time_window: float = 20.0  # ms
    synaptic_decay: float = 0.95
    homeostatic_scaling: bool = True
    lateral_inhibition: float = 0.1
    max_spike_rate: float = 100.0  # Hz
    adaptation_threshold: float = 0.8
    plasticity_decay: float = 0.99
    enable_metaplasticity: bool = True


class SpikingNeuron:
    """Individual spiking neuron with leaky integrate-and-fire dynamics."""
    
    def __init__(self, config: NeuromorphicConfig):
        self.config = config
        self.membrane_potential = 0.0
        self.spike_times = deque(maxlen=1000)
        self.refractory_time = 0.0
        self.adaptation_current = 0.0
        self.last_spike_time = -float('inf')
        
        # Synaptic weights and plasticity
        self.input_weights = {}
        self.synaptic_traces = {}
        self.eligibility_traces = {}
        
        # Homeostatic mechanisms
        self.target_rate = 10.0  # Hz
        self.average_rate = 0.0
        self.homeostatic_factor = 1.0
        
    def update(self, input_current: float, dt: float = 1.0) -> bool:
        """Update neuron state and return True if spike occurs."""
        
        current_time = time.time() * 1000  # Convert to ms
        
        # Check refractory period
        if current_time - self.last_spike_time < self.config.refractory_period:
            return False
        
        # Membrane dynamics (leaky integrate-and-fire)
        membrane_decay = math.exp(-dt / self.config.membrane_time_constant)
        self.membrane_potential = (
            self.membrane_potential * membrane_decay + 
            input_current * self.homeostatic_factor
        )
        
        # Adaptation current (spike-frequency adaptation)
        self.adaptation_current *= math.exp(-dt / (self.config.membrane_time_constant * 5))
        self.membrane_potential -= self.adaptation_current
        
        # Check for spike
        if self.membrane_potential >= self.config.spike_threshold:
            self.spike()
            return True
        
        return False
    
    def spike(self):
        """Generate spike and update internal state."""
        current_time = time.time() * 1000
        
        # Reset membrane potential
        self.membrane_potential = 0.0
        self.last_spike_time = current_time
        
        # Add to spike history
        self.spike_times.append(current_time)
        
        # Update adaptation current
        self.adaptation_current += 0.1
        
        # Update average firing rate
        self._update_firing_rate()
        
        # Homeostatic scaling
        if self.config.homeostatic_scaling:
            self._homeostatic_scaling()
    
    def _update_firing_rate(self):
        """Update average firing rate for homeostatic mechanisms."""
        current_time = time.time() * 1000
        recent_spikes = [t for t in self.spike_times if current_time - t < 1000]  # Last 1 second
        self.average_rate = len(recent_spikes)  # Spikes per second (Hz)
    
    def _homeostatic_scaling(self):
        """Apply homeostatic scaling to maintain target firing rate."""
        if self.average_rate > self.target_rate * 1.2:
            self.homeostatic_factor *= 0.99  # Decrease excitability
        elif self.average_rate < self.target_rate * 0.8:
            self.homeostatic_factor *= 1.01  # Increase excitability
        
        # Bound homeostatic factor
        self.homeostatic_factor = np.clip(self.homeostatic_factor, 0.1, 2.0)


class STDPSynapse:
    """Synapse with spike-timing dependent plasticity."""
    
    def __init__(self, config: NeuromorphicConfig, initial_weight: float = 0.5):
        self.config = config
        self.weight = initial_weight
        self.pre_trace = 0.0  # Presynaptic trace
        self.post_trace = 0.0  # Postsynaptic trace
        self.eligibility = 0.0  # Eligibility trace for three-factor learning
        
        # Metaplasticity
        self.plasticity_threshold = 1.0
        self.recent_changes = deque(maxlen=100)
        
    def update_traces(self, dt: float = 1.0):
        """Update synaptic traces."""
        decay_factor = math.exp(-dt / self.config.stdp_time_window)
        self.pre_trace *= decay_factor
        self.post_trace *= decay_factor
        self.eligibility *= decay_factor
    
    def pre_spike(self):
        """Handle presynaptic spike."""
        self.pre_trace += 1.0
        
        # Depression: post-before-pre
        if self.post_trace > 0:
            weight_change = -self.config.stdp_learning_rate * self.post_trace
            self._apply_weight_change(weight_change)
    
    def post_spike(self):
        """Handle postsynaptic spike."""
        self.post_trace += 1.0
        
        # Potentiation: pre-before-post
        if self.pre_trace > 0:
            weight_change = self.config.stdp_learning_rate * self.pre_trace
            self._apply_weight_change(weight_change)
    
    def _apply_weight_change(self, change: float):
        """Apply weight change with metaplasticity."""
        
        if self.config.enable_metaplasticity:
            # Metaplasticity: adjust learning based on recent history
            recent_activity = sum(abs(c) for c in self.recent_changes)
            if recent_activity > self.plasticity_threshold:
                change *= 0.5  # Reduce plasticity if very active recently
        
        # Apply change
        old_weight = self.weight
        self.weight += change
        
        # Bound weights
        self.weight = np.clip(self.weight, 0.0, 2.0)
        
        # Record change
        actual_change = self.weight - old_weight
        self.recent_changes.append(actual_change)
    
    def get_transmission(self, spike: bool) -> float:
        """Get synaptic transmission for spike."""
        if spike:
            return self.weight
        return 0.0


class SpikingNeuralNetwork:
    """Spiking neural network with STDP learning."""
    
    def __init__(self, config: NeuromorphicConfig, input_size: int, hidden_size: int, output_size: int):
        self.config = config
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Create neurons
        self.input_neurons = [SpikingNeuron(config) for _ in range(input_size)]
        self.hidden_neurons = [SpikingNeuron(config) for _ in range(hidden_size)]
        self.output_neurons = [SpikingNeuron(config) for _ in range(output_size)]
        
        # Create synapses
        self.input_to_hidden_synapses = {}
        self.hidden_to_output_synapses = {}
        self.lateral_synapses = {}  # For lateral inhibition
        
        self._initialize_synapses()
        
        # Network state
        self.spike_history = defaultdict(list)
        self.voltage_history = defaultdict(list)
        
    def _initialize_synapses(self):
        """Initialize synaptic connections."""
        
        # Input to hidden connections
        for i in range(self.input_size):
            for h in range(self.hidden_size):
                weight = np.random.normal(0.5, 0.1)
                self.input_to_hidden_synapses[(i, h)] = STDPSynapse(self.config, weight)
        
        # Hidden to output connections
        for h in range(self.hidden_size):
            for o in range(self.output_size):
                weight = np.random.normal(0.5, 0.1)
                self.hidden_to_output_synapses[(h, o)] = STDPSynapse(self.config, weight)
        
        # Lateral inhibition in hidden layer
        for h1 in range(self.hidden_size):
            for h2 in range(self.hidden_size):
                if h1 != h2:
                    weight = -self.config.lateral_inhibition
                    self.lateral_synapses[(h1, h2)] = STDPSynapse(self.config, weight)
    
    def forward(self, input_spikes: List[bool], learning: bool = True) -> List[bool]:
        """Forward pass through spiking network."""
        
        # Process input layer
        input_outputs = []
        for i, (neuron, spike) in enumerate(zip(self.input_neurons, input_spikes)):
            if spike:
                neuron.spike()
            input_outputs.append(spike)
        
        # Process hidden layer
        hidden_outputs = []
        for h, hidden_neuron in enumerate(self.hidden_neurons):
            
            # Collect inputs from input layer
            total_current = 0.0
            for i, input_spike in enumerate(input_outputs):
                synapse = self.input_to_hidden_synapses[(i, h)]
                current = synapse.get_transmission(input_spike)
                total_current += current
                
                # STDP learning
                if learning:
                    if input_spike:
                        synapse.pre_spike()
            
            # Lateral inhibition
            for h2, other_spike in enumerate(hidden_outputs):
                if h != h2 and (h2, h) in self.lateral_synapses:
                    synapse = self.lateral_synapses[(h2, h)]
                    inhibition = synapse.get_transmission(other_spike)
                    total_current += inhibition
            
            # Update neuron
            spike_occurred = hidden_neuron.update(total_current)
            hidden_outputs.append(spike_occurred)
            
            # STDP learning for postsynaptic spikes
            if learning and spike_occurred:
                for i in range(self.input_size):
                    synapse = self.input_to_hidden_synapses[(i, h)]
                    synapse.post_spike()
        
        # Process output layer
        output_spikes = []
        for o, output_neuron in enumerate(self.output_neurons):
            
            # Collect inputs from hidden layer
            total_current = 0.0
            for h, hidden_spike in enumerate(hidden_outputs):
                synapse = self.hidden_to_output_synapses[(h, o)]
                current = synapse.get_transmission(hidden_spike)
                total_current += current
                
                # STDP learning
                if learning:
                    if hidden_spike:
                        synapse.pre_spike()
            
            # Update neuron
            spike_occurred = output_neuron.update(total_current)
            output_spikes.append(spike_occurred)
            
            # STDP learning for postsynaptic spikes
            if learning and spike_occurred:
                for h in range(self.hidden_size):
                    synapse = self.hidden_to_output_synapses[(h, o)]
                    synapse.post_spike()
        
        # Update all synaptic traces
        if learning:
            self._update_all_traces()
        
        # Record activity
        self._record_activity(input_outputs, hidden_outputs, output_spikes)
        
        return output_spikes
    
    def _update_all_traces(self):
        """Update all synaptic traces."""
        for synapse in self.input_to_hidden_synapses.values():
            synapse.update_traces()
        
        for synapse in self.hidden_to_output_synapses.values():
            synapse.update_traces()
        
        for synapse in self.lateral_synapses.values():
            synapse.update_traces()
    
    def _record_activity(self, input_spikes: List[bool], hidden_spikes: List[bool], output_spikes: List[bool]):
        """Record network activity for analysis."""
        current_time = time.time()
        
        self.spike_history['input'].append((current_time, input_spikes))
        self.spike_history['hidden'].append((current_time, hidden_spikes))
        self.spike_history['output'].append((current_time, output_spikes))
        
        # Keep only recent history
        for layer in self.spike_history:
            if len(self.spike_history[layer]) > 1000:
                self.spike_history[layer] = self.spike_history[layer][-1000:]


class NeuromorphicContinualLearner:
    """Main neuromorphic continual learning system."""
    
    def __init__(self, model, config: NeuromorphicConfig):
        self.model = model
        self.config = config
        
        # Neuromorphic components
        self.spiking_networks = {}
        self.synaptic_consolidation = {}
        self.task_specific_plasticity = {}
        
        # Rate encoding parameters
        self.encoding_window = 100  # ms
        self.max_rate = config.max_spike_rate
        
        # Continual learning state
        self.task_memories = defaultdict(list)
        self.synaptic_importance = {}
        self.plasticity_history = defaultdict(list)
        
        logger.info("Neuromorphic continual learning system initialized")
    
    def encode_to_spikes(self, activation: torch.Tensor, time_steps: int = 100) -> List[List[bool]]:
        """Convert neural activations to spike trains using rate encoding."""
        
        # Normalize activations to [0, 1]
        activation = torch.sigmoid(activation)
        
        # Convert to spike rates
        spike_rates = activation * self.max_rate  # Hz
        
        # Generate spike trains
        spike_trains = []
        dt = self.encoding_window / time_steps  # ms per time step
        
        for t in range(time_steps):
            spikes = []
            for rate in spike_rates:
                # Poisson process for spike generation
                spike_prob = rate * dt / 1000.0  # Convert to probability
                spike = np.random.random() < spike_prob
                spikes.append(spike)
            spike_trains.append(spikes)
        
        return spike_trains
    
    def decode_from_spikes(self, spike_trains: List[List[bool]]) -> torch.Tensor:
        """Decode spike trains back to activations using rate decoding."""
        
        if not spike_trains:
            return torch.zeros(1)
        
        # Count spikes over time window
        spike_counts = np.sum(spike_trains, axis=0)
        
        # Convert to rates (Hz)
        time_window = len(spike_trains) * self.encoding_window / 1000.0  # seconds
        spike_rates = spike_counts / time_window
        
        # Convert to activations (normalize by max rate)
        activations = spike_rates / self.max_rate
        
        return torch.tensor(activations, dtype=torch.float32)
    
    def create_task_network(self, task_id: str, input_size: int, output_size: int) -> SpikingNeuralNetwork:
        """Create task-specific spiking neural network."""
        
        hidden_size = min(128, input_size * 2)  # Adaptive hidden size
        
        network = SpikingNeuralNetwork(
            self.config,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size
        )
        
        self.spiking_networks[task_id] = network
        self.synaptic_consolidation[task_id] = {}
        self.task_specific_plasticity[task_id] = 1.0  # Full plasticity initially
        
        logger.info(f"Created spiking network for task {task_id}: {input_size}→{hidden_size}→{output_size}")
        return network
    
    def neuromorphic_adapt(
        self,
        task_id: str,
        input_activations: torch.Tensor,
        target_output: torch.Tensor,
        num_epochs: int = 100
    ) -> Dict[str, Any]:
        """Adapt using neuromorphic learning principles."""
        
        if task_id not in self.spiking_networks:
            # Create network if it doesn't exist
            input_size = input_activations.size(-1)
            output_size = target_output.size(-1) if target_output.dim() > 0 else 1
            self.create_task_network(task_id, input_size, output_size)
        
        network = self.spiking_networks[task_id]
        
        # Encode inputs and targets to spikes
        input_spikes = self.encode_to_spikes(input_activations.flatten())
        target_spikes = self.encode_to_spikes(target_output.flatten())
        
        # Training metrics
        performance_history = []
        plasticity_changes = []
        
        # Neuromorphic learning loop
        for epoch in range(num_epochs):
            epoch_correct = 0
            epoch_total = 0
            
            # Present spike patterns
            for t, (input_pattern, target_pattern) in enumerate(zip(input_spikes, target_spikes)):
                
                # Forward pass
                output_spikes = network.forward(input_pattern, learning=True)
                
                # Compute performance (simplified)
                correct = sum(1 for o, t in zip(output_spikes, target_pattern) if o == t)
                epoch_correct += correct
                epoch_total += len(output_spikes)
                
                # Neuromodulation for reinforcement
                performance_score = correct / len(output_spikes)
                self._apply_neuromodulation(network, performance_score)
            
            # Epoch metrics
            epoch_accuracy = epoch_correct / max(epoch_total, 1)
            performance_history.append(epoch_accuracy)
            
            # Synaptic consolidation
            if epoch % 10 == 0:
                consolidation_strength = self._consolidate_synapses(task_id, epoch_accuracy)
                plasticity_changes.append(consolidation_strength)
            
            # Early stopping
            if len(performance_history) > 10:
                recent_improvement = performance_history[-1] - performance_history[-10]
                if recent_improvement < 0.01:  # Minimal improvement
                    break
        
        # Store task memory
        self.task_memories[task_id].append({
            'input_pattern': input_activations.clone(),
            'target_pattern': target_output.clone(),
            'final_accuracy': performance_history[-1] if performance_history else 0.0,
            'adaptation_time': time.time()
        })
        
        adaptation_result = {
            'task_id': task_id,
            'epochs_trained': len(performance_history),
            'final_accuracy': performance_history[-1] if performance_history else 0.0,
            'convergence_speed': self._compute_convergence_speed(performance_history),
            'synaptic_changes': len(plasticity_changes),
            'network_size': {
                'input': network.input_size,
                'hidden': network.hidden_size,
                'output': network.output_size
            }
        }
        
        logger.info(f"Neuromorphic adaptation complete for {task_id}: accuracy={adaptation_result['final_accuracy']:.3f}")
        return adaptation_result
    
    def _apply_neuromodulation(self, network: SpikingNeuralNetwork, performance_score: float):
        """Apply neuromodulation based on performance feedback."""
        
        # Dopaminergic modulation of plasticity
        if performance_score > 0.8:
            # Good performance - strengthen recent changes
            modulation_factor = 1.2
        elif performance_score < 0.3:
            # Poor performance - weaken recent changes
            modulation_factor = 0.8
        else:
            # Moderate performance - no change
            modulation_factor = 1.0
        
        # Apply modulation to synapses
        for synapse in network.input_to_hidden_synapses.values():
            if synapse.recent_changes:
                last_change = synapse.recent_changes[-1]
                synapse.weight += 0.1 * modulation_factor * last_change
                synapse.weight = np.clip(synapse.weight, 0.0, 2.0)
        
        for synapse in network.hidden_to_output_synapses.values():
            if synapse.recent_changes:
                last_change = synapse.recent_changes[-1]
                synapse.weight += 0.1 * modulation_factor * last_change
                synapse.weight = np.clip(synapse.weight, 0.0, 2.0)
    
    def _consolidate_synapses(self, task_id: str, performance: float) -> float:
        """Consolidate important synapses to prevent forgetting."""
        
        if task_id not in self.spiking_networks:
            return 0.0
        
        network = self.spiking_networks[task_id]
        consolidation_strength = 0.0
        
        # Consolidate synapses with high importance
        for (i, h), synapse in network.input_to_hidden_synapses.items():
            importance = self._compute_synaptic_importance(synapse, performance)
            
            if importance > self.config.adaptation_threshold:
                # Consolidate this synapse
                consolidation_key = f"{task_id}_ih_{i}_{h}"
                self.synaptic_consolidation[task_id][consolidation_key] = {
                    'weight': synapse.weight,
                    'importance': importance,
                    'consolidation_time': time.time()
                }
                consolidation_strength += importance
        
        for (h, o), synapse in network.hidden_to_output_synapses.items():
            importance = self._compute_synaptic_importance(synapse, performance)
            
            if importance > self.config.adaptation_threshold:
                # Consolidate this synapse
                consolidation_key = f"{task_id}_ho_{h}_{o}"
                self.synaptic_consolidation[task_id][consolidation_key] = {
                    'weight': synapse.weight,
                    'importance': importance,
                    'consolidation_time': time.time()
                }
                consolidation_strength += importance
        
        # Reduce plasticity for this task as consolidation increases
        total_consolidation = len(self.synaptic_consolidation[task_id])
        self.task_specific_plasticity[task_id] = max(0.1, 1.0 - total_consolidation * 0.01)
        
        return consolidation_strength
    
    def _compute_synaptic_importance(self, synapse: STDPSynapse, performance: float) -> float:
        """Compute importance of synapse for consolidation."""
        
        # Factors contributing to synaptic importance:
        # 1. Weight magnitude
        weight_importance = abs(synapse.weight) / 2.0  # Normalized by max weight
        
        # 2. Recent plasticity activity
        recent_activity = sum(abs(change) for change in synapse.recent_changes)
        activity_importance = min(1.0, recent_activity / 10.0)
        
        # 3. Performance correlation
        performance_importance = performance
        
        # 4. Trace activity
        trace_importance = (synapse.pre_trace + synapse.post_trace) / 2.0
        
        # Combined importance score
        importance = (
            0.3 * weight_importance +
            0.2 * activity_importance +
            0.3 * performance_importance +
            0.2 * trace_importance
        )
        
        return min(1.0, importance)
    
    def _compute_convergence_speed(self, performance_history: List[float]) -> float:
        """Compute how quickly the network converged."""
        
        if len(performance_history) < 5:
            return 0.0
        
        # Find when performance reached 80% of final value
        final_performance = performance_history[-1]
        target_performance = 0.8 * final_performance
        
        convergence_epoch = len(performance_history)
        for i, perf in enumerate(performance_history):
            if perf >= target_performance:
                convergence_epoch = i
                break
        
        # Normalize by total epochs (faster convergence = higher score)
        convergence_speed = 1.0 - (convergence_epoch / len(performance_history))
        return max(0.0, convergence_speed)
    
    def synaptic_replay(self, task_id: str, replay_strength: float = 0.5) -> Dict[str, Any]:
        """Replay consolidated synaptic patterns to maintain memory."""
        
        if task_id not in self.synaptic_consolidation:
            return {'replayed': 0, 'strengthened': 0}
        
        network = self.spiking_networks[task_id]
        replayed = 0
        strengthened = 0
        
        # Replay consolidated synapses
        for synapse_key, consolidation_info in self.synaptic_consolidation[task_id].items():
            
            # Parse synapse key
            parts = synapse_key.split('_')
            if len(parts) >= 4:
                layer = parts[1]
                i, j = int(parts[2]), int(parts[3])
                
                # Get synapse
                if layer == 'ih' and (i, j) in network.input_to_hidden_synapses:
                    synapse = network.input_to_hidden_synapses[(i, j)]
                elif layer == 'ho' and (i, j) in network.hidden_to_output_synapses:
                    synapse = network.hidden_to_output_synapses[(i, j)]
                else:
                    continue
                
                # Apply replay strengthening
                target_weight = consolidation_info['weight']
                importance = consolidation_info['importance']
                
                # Move current weight towards consolidated weight
                weight_diff = target_weight - synapse.weight
                synapse.weight += replay_strength * importance * weight_diff
                synapse.weight = np.clip(synapse.weight, 0.0, 2.0)
                
                replayed += 1
                if abs(weight_diff) > 0.1:
                    strengthened += 1
        
        replay_result = {
            'task_id': task_id,
            'replayed': replayed,
            'strengthened': strengthened,
            'replay_strength': replay_strength
        }
        
        logger.info(f"Synaptic replay for {task_id}: {replayed} synapses replayed, {strengthened} strengthened")
        return replay_result
    
    def get_neuromorphic_status(self) -> Dict[str, Any]:
        """Get comprehensive neuromorphic system status."""
        
        status = {
            'config': {
                'spike_threshold': self.config.spike_threshold,
                'stdp_learning_rate': self.config.stdp_learning_rate,
                'membrane_time_constant': self.config.membrane_time_constant
            },
            'networks': {
                'total_tasks': len(self.spiking_networks),
                'task_ids': list(self.spiking_networks.keys())
            },
            'consolidation': {
                'total_consolidated_synapses': sum(
                    len(consolidation) for consolidation in self.synaptic_consolidation.values()
                ),
                'tasks_with_consolidation': len([
                    task for task, consol in self.synaptic_consolidation.items() if consol
                ])
            },
            'plasticity': {
                'task_plasticity_levels': self.task_specific_plasticity.copy()
            }
        }
        
        # Add network statistics
        if self.spiking_networks:
            total_neurons = sum(
                net.input_size + net.hidden_size + net.output_size
                for net in self.spiking_networks.values()
            )
            total_synapses = sum(
                len(net.input_to_hidden_synapses) + len(net.hidden_to_output_synapses)
                for net in self.spiking_networks.values()
            )
            
            status['network_stats'] = {
                'total_neurons': total_neurons,
                'total_synapses': total_synapses,
                'avg_neurons_per_task': total_neurons / len(self.spiking_networks),
                'avg_synapses_per_task': total_synapses / len(self.spiking_networks)
            }
        
        return status
    
    def save_neuromorphic_state(self, filepath: str):
        """Save neuromorphic learning state."""
        
        # Extract synaptic weights from networks
        network_states = {}
        for task_id, network in self.spiking_networks.items():
            network_states[task_id] = {
                'input_to_hidden_weights': {
                    f"{i}_{h}": synapse.weight
                    for (i, h), synapse in network.input_to_hidden_synapses.items()
                },
                'hidden_to_output_weights': {
                    f"{h}_{o}": synapse.weight
                    for (h, o), synapse in network.hidden_to_output_synapses.items()
                },
                'network_size': {
                    'input': network.input_size,
                    'hidden': network.hidden_size,
                    'output': network.output_size
                }
            }
        
        state = {
            'config': self.config.__dict__,
            'network_states': network_states,
            'synaptic_consolidation': self.synaptic_consolidation,
            'task_specific_plasticity': self.task_specific_plasticity,
            'task_memories': {
                task_id: [
                    {
                        'input_pattern': mem['input_pattern'].cpu(),
                        'target_pattern': mem['target_pattern'].cpu(),
                        'final_accuracy': mem['final_accuracy'],
                        'adaptation_time': mem['adaptation_time']
                    }
                    for mem in memories
                ]
                for task_id, memories in self.task_memories.items()
            }
        }
        
        torch.save(state, filepath)
        logger.info(f"Neuromorphic state saved to {filepath}")


def create_neuromorphic_learner(
    model,
    spike_threshold: float = 1.0,
    stdp_learning_rate: float = 0.01,
    membrane_time_constant: float = 20.0,
    **kwargs
) -> NeuromorphicContinualLearner:
    """Factory function to create neuromorphic continual learner."""
    
    config = NeuromorphicConfig(
        spike_threshold=spike_threshold,
        stdp_learning_rate=stdp_learning_rate,
        membrane_time_constant=membrane_time_constant,
        **kwargs
    )
    
    return NeuromorphicContinualLearner(model, config)


# Demonstration function
def demonstrate_neuromorphic_learning():
    """Demonstrate neuromorphic continual learning capabilities."""
    
    logger.info("Demonstrating Neuromorphic Continual Learning")
    
    print("Neuromorphic Continual Learning Framework:")
    print("✓ Spiking neural networks with leaky integrate-and-fire neurons")
    print("✓ Spike-timing dependent plasticity (STDP)")
    print("✓ Homeostatic scaling for stability")
    print("✓ Synaptic consolidation for memory preservation")
    print("✓ Lateral inhibition for competition")
    print("✓ Neuromodulation for reinforcement")
    print("✓ Metaplasticity for adaptive learning rates")


if __name__ == "__main__":
    demonstrate_neuromorphic_learning()