"""Configuration system for continual transformer."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import yaml
from pathlib import Path


@dataclass
class ContinualConfig:
    """Configuration for continual learning transformer."""
    
    # Model architecture
    model_name: str = "distilbert-base-uncased"
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 6
    max_sequence_length: int = 512
    
    # Continual learning parameters
    max_tasks: int = 50
    freeze_base_model: bool = True
    adaptation_method: str = "activation"  # activation, prompt, adapter
    knowledge_distillation_alpha: float = 0.5
    temperature: float = 4.0
    
    # Task routing
    task_routing_method: str = "learned"  # learned, manual, embedding
    router_hidden_size: int = 256
    
    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    
    # Memory management
    replay_buffer_size: int = 1000
    replay_ratio: float = 0.2
    
    # Device and performance
    device: str = "auto"  # auto, cpu, cuda, mps  
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # Monitoring and logging
    log_level: str = "INFO"
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    
    # Paths
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    
    # Task-specific configurations
    task_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Advanced options
    use_knowledge_distillation: bool = True
    elastic_weight_consolidation: bool = True
    ewc_lambda: float = 1000.0
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Validate adaptation method
        valid_methods = ["activation", "prompt", "adapter"]
        if self.adaptation_method not in valid_methods:
            raise ValueError(f"adaptation_method must be one of {valid_methods}")
        
        # Validate task routing method
        valid_routing = ["learned", "manual", "embedding"]
        if self.task_routing_method not in valid_routing:
            raise ValueError(f"task_routing_method must be one of {valid_routing}")
        
        # Ensure paths exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device if needed
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ContinualConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_yaml(self, config_path: str):
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
    
    def get_task_config(self, task_id: str) -> Dict[str, Any]:
        """Get task-specific configuration."""
        return self.task_configs.get(task_id, {})
    
    def set_task_config(self, task_id: str, config: Dict[str, Any]):
        """Set task-specific configuration."""
        self.task_configs[task_id] = config
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for transformers."""
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "max_position_embeddings": self.max_sequence_length,
        }
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        main_params = [
            f"model_name='{self.model_name}'",
            f"max_tasks={self.max_tasks}",
            f"adaptation_method='{self.adaptation_method}'",
            f"device='{self.device}'"
        ]
        return f"ContinualConfig({', '.join(main_params)})"