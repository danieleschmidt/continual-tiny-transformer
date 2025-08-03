"""Activation-based adapters for zero-parameter continual learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math


class ActivationAdapter(nn.Module):
    """Zero-parameter activation adapter for task-specific modifications.
    
    This adapter modifies activations rather than adding parameters,
    enabling continual learning without parameter expansion.
    """
    
    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 64,
        num_layers: int = 12,
        activation_function: str = "gelu",
        dropout_prob: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        
        # Activation transformation layers (lightweight)
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        
        # Activation function
        if activation_function == "gelu":
            self.activation = nn.GELU()
        elif activation_function == "relu":
            self.activation = nn.ReLU()
        elif activation_function == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.GELU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Task-specific scaling factors (learned)
        self.scale_factor = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize adapter weights with small values."""
        # Initialize down projection with small random values
        nn.init.normal_(self.down_project.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down_project.bias)
        
        # Initialize up projection to zero (identity at initialization)
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
        
        # Initialize scale factor to small value
        nn.init.constant_(self.scale_factor, 0.1)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply activation adapter to hidden states.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            Adapted hidden states with same shape as input
        """
        # Store original for residual connection
        residual = hidden_states
        
        # Down-project to adapter dimension
        adapter_hidden = self.down_project(hidden_states)
        
        # Apply activation function
        adapter_hidden = self.activation(adapter_hidden)
        
        # Apply dropout
        adapter_hidden = self.dropout(adapter_hidden)
        
        # Up-project back to original dimension
        adapter_output = self.up_project(adapter_hidden)
        
        # Scale the adapter output
        adapter_output = self.scale_factor * adapter_output
        
        # Residual connection
        output = residual + adapter_output
        
        # Layer normalization for stability
        output = self.layer_norm(output)
        
        return output
    
    def get_adapter_parameters(self) -> int:
        """Get number of parameters in this adapter."""
        return sum(p.numel() for p in self.parameters())


class MultiLayerActivationAdapter(nn.Module):
    """Multi-layer activation adapter for deeper adaptations."""
    
    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 64,
        num_adapter_layers: int = 2,
        activation_function: str = "gelu",
        dropout_prob: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        self.num_adapter_layers = num_adapter_layers
        
        # Build adapter layers
        layers = []
        
        # First layer: down-project
        layers.append(nn.Linear(hidden_size, adapter_size))
        layers.append(self._get_activation(activation_function))
        layers.append(nn.Dropout(dropout_prob))
        
        # Middle layers
        for _ in range(num_adapter_layers - 2):
            layers.append(nn.Linear(adapter_size, adapter_size))
            layers.append(self._get_activation(activation_function))
            layers.append(nn.Dropout(dropout_prob))
        
        # Final layer: up-project
        if num_adapter_layers > 1:
            layers.append(nn.Linear(adapter_size, hidden_size))
        
        self.adapter_layers = nn.Sequential(*layers)
        
        # Layer normalization and scaling
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.scale_factor = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation_function: str):
        """Get activation function by name."""
        if activation_function == "gelu":
            return nn.GELU()
        elif activation_function == "relu":
            return nn.ReLU()
        elif activation_function == "tanh":
            return nn.Tanh()
        elif activation_function == "swish":
            return nn.SiLU()
        else:
            return nn.GELU()
    
    def _init_weights(self):
        """Initialize adapter weights."""
        for module in self.adapter_layers:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)
        
        # Make last layer output close to zero initially
        if len(self.adapter_layers) > 0:
            last_linear = None
            for module in reversed(self.adapter_layers):
                if isinstance(module, nn.Linear):
                    last_linear = module
                    break
            
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)
        
        nn.init.constant_(self.scale_factor, 0.1)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply multi-layer adapter."""
        residual = hidden_states
        
        # Apply adapter transformation
        adapter_output = self.adapter_layers(hidden_states)
        
        # Scale and add residual
        output = residual + self.scale_factor * adapter_output
        
        # Layer normalization
        output = self.layer_norm(output)
        
        return output


class AttentionAdapter(nn.Module):
    """Attention-based adapter for selective feature modification."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        adapter_size: int = 64,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.adapter_size = adapter_size
        self.head_size = adapter_size // num_attention_heads
        
        # Multi-head attention for feature selection
        self.query = nn.Linear(hidden_size, adapter_size)
        self.key = nn.Linear(hidden_size, adapter_size)
        self.value = nn.Linear(hidden_size, adapter_size)
        
        # Output projection
        self.output_proj = nn.Linear(adapter_size, hidden_size)
        
        # Normalization and dropout
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Scaling factor
        self.scale_factor = nn.Parameter(torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention adapter weights."""
        for module in [self.query, self.key, self.value]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        
        # Initialize output projection to zero
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        nn.init.constant_(self.scale_factor, 0.1)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply attention-based adaptation."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        residual = hidden_states
        
        # Compute Q, K, V
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_attention_heads, self.head_size).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_attention_heads, self.head_size).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_attention_heads, self.head_size).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_size)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.adapter_size
        )
        
        # Output projection
        adapter_output = self.output_proj(context)
        
        # Scale and residual connection
        output = residual + self.scale_factor * adapter_output
        
        # Layer normalization
        output = self.layer_norm(output)
        
        return output


class HyperAdapter(nn.Module):
    """Hypernetwork-based adapter that generates task-specific parameters."""
    
    def __init__(
        self,
        hidden_size: int,
        adapter_size: int = 64,
        hyper_size: int = 128,
        task_embedding_size: int = 32,
        max_tasks: int = 50
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.adapter_size = adapter_size
        self.hyper_size = hyper_size
        self.task_embedding_size = task_embedding_size
        self.max_tasks = max_tasks
        
        # Task embeddings
        self.task_embeddings = nn.Embedding(max_tasks, task_embedding_size)
        
        # Hypernetwork to generate adapter parameters
        self.hyper_net = nn.Sequential(
            nn.Linear(task_embedding_size, hyper_size),
            nn.ReLU(),
            nn.Linear(hyper_size, adapter_size * hidden_size + adapter_size)  # W + b
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize hypernetwork weights."""
        nn.init.normal_(self.task_embeddings.weight, mean=0.0, std=0.02)
        
        for module in self.hyper_net:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor, task_id: int) -> torch.Tensor:
        """Apply hypernetwork-generated adapter."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        residual = hidden_states
        
        # Get task embedding
        task_emb = self.task_embeddings(torch.tensor(task_id, device=hidden_states.device))
        
        # Generate adapter parameters
        hyper_output = self.hyper_net(task_emb)
        
        # Split into weight and bias
        weight_size = self.adapter_size * self.hidden_size
        adapter_weight = hyper_output[:weight_size].view(self.adapter_size, self.hidden_size)
        adapter_bias = hyper_output[weight_size:]
        
        # Apply generated linear transformation
        # Reshape hidden states for batch matrix multiplication
        hidden_flat = hidden_states.view(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
        
        # Apply adapter transformation
        adapter_output = F.linear(hidden_flat, adapter_weight, adapter_bias)
        
        # Reshape back
        adapter_output = adapter_output.view(batch_size, seq_len, self.adapter_size)
        
        # Project back to hidden size (optional - could be learned)
        if self.adapter_size != self.hidden_size:
            # Simple averaging for dimension mismatch
            if self.adapter_size > self.hidden_size:
                adapter_output = adapter_output[:, :, :self.hidden_size]
            else:
                # Pad with zeros
                padding = torch.zeros(
                    batch_size, seq_len, self.hidden_size - self.adapter_size,
                    device=hidden_states.device
                )
                adapter_output = torch.cat([adapter_output, padding], dim=-1)
        
        # Residual connection and normalization
        output = residual + 0.1 * adapter_output  # Small scaling factor
        output = self.layer_norm(output)
        
        return output


def create_adapter(adapter_type: str, **kwargs) -> nn.Module:
    """Factory function to create different types of adapters."""
    if adapter_type == "activation":
        return ActivationAdapter(**kwargs)
    elif adapter_type == "multi_layer":
        return MultiLayerActivationAdapter(**kwargs)
    elif adapter_type == "attention":
        return AttentionAdapter(**kwargs)
    elif adapter_type == "hyper":
        return HyperAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")