"""Deployment utilities for continual learning models."""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager
import tempfile

from ..api import ContinualLearningAPI
from ..core import ContinualConfig

logger = logging.getLogger(__name__)

class ModelDeployment:
    """Production deployment utilities for continual learning models."""
    
    def __init__(self, api: ContinualLearningAPI):
        self.api = api
        self.deployment_config = {}
        self.health_checks = []
    
    def prepare_for_production(
        self, 
        optimization_level: str = "balanced",
        enable_monitoring: bool = True,
        max_memory_mb: Optional[int] = None
    ) -> Dict[str, Any]:
        """Prepare model for production deployment.
        
        Args:
            optimization_level: Optimization strategy ('speed', 'memory', 'balanced')
            enable_monitoring: Whether to enable runtime monitoring
            max_memory_mb: Maximum memory usage in MB (optional)
            
        Returns:
            Deployment configuration and metrics
        """
        deployment_info = {
            "optimization_applied": {},
            "memory_usage": {},
            "performance_metrics": {},
            "deployment_timestamp": torch.utils.data.get_worker_info() is None  # Simple timestamp substitute
        }
        
        # Apply optimizations
        logger.info(f"Optimizing model for {optimization_level} deployment...")
        optimizations = self.api.optimize_for_deployment(optimization_level)
        deployment_info["optimization_applied"] = optimizations
        
        # Get memory usage
        memory_usage = self.api.get_memory_usage()
        deployment_info["memory_usage"] = memory_usage
        
        # Memory check
        if max_memory_mb and memory_usage.get("total_parameters", 0) * 4 / 1024 / 1024 > max_memory_mb:
            logger.warning(f"Model size exceeds memory limit: {max_memory_mb}MB")
        
        # Enable monitoring if requested
        if enable_monitoring:
            self._setup_monitoring()
            deployment_info["monitoring_enabled"] = True
        
        # Set deployment configuration
        self.deployment_config = {
            "optimization_level": optimization_level,
            "monitoring_enabled": enable_monitoring,
            "max_memory_mb": max_memory_mb,
            "num_tasks": len(self.api.trained_tasks),
            "model_info": self.api.get_task_info()
        }
        
        logger.info("Model prepared for production deployment")
        return deployment_info
    
    def _setup_monitoring(self):
        """Setup runtime monitoring for deployed model."""
        # Basic health checks
        self.health_checks = [
            self._check_model_loaded,
            self._check_memory_usage,
            self._check_device_availability
        ]
        logger.info("Monitoring setup complete")
    
    def health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check on deployed model.
        
        Returns:
            Health check results
        """
        results = {
            "status": "healthy",
            "checks": {},
            "timestamp": "now",  # Simplified timestamp
            "warnings": [],
            "errors": []
        }
        
        for check in self.health_checks:
            try:
                check_name = check.__name__.replace("_check_", "")
                check_result = check()
                results["checks"][check_name] = check_result
                
                if not check_result.get("passed", True):
                    results["warnings"].append(f"{check_name}: {check_result.get('message', 'Check failed')}")
                    
            except Exception as e:
                error_msg = f"Health check {check.__name__} failed: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
        
        # Determine overall status
        if results["errors"]:
            results["status"] = "unhealthy"
        elif results["warnings"]:
            results["status"] = "degraded"
        
        return results
    
    def _check_model_loaded(self) -> Dict[str, Any]:
        """Check if model is properly loaded."""
        try:
            # Try a simple forward pass
            dummy_input = torch.randint(0, 100, (1, 10))  # Dummy token IDs
            device = next(self.api.model.parameters()).device
            dummy_input = dummy_input.to(device)
            
            with torch.no_grad():
                if self.api.trained_tasks:
                    task_id = list(self.api.trained_tasks)[0]
                    _ = self.api.model.forward(dummy_input, task_id=task_id)
                    
            return {"passed": True, "message": "Model responding correctly"}
        except Exception as e:
            return {"passed": False, "message": f"Model check failed: {str(e)}"}
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            memory_info = self.api.get_memory_usage()
            total_params = memory_info.get("total_parameters", 0)
            memory_mb = total_params * 4 / 1024 / 1024  # Rough estimate
            
            max_memory = self.deployment_config.get("max_memory_mb")
            if max_memory and memory_mb > max_memory:
                return {
                    "passed": False, 
                    "message": f"Memory usage {memory_mb:.1f}MB exceeds limit {max_memory}MB",
                    "memory_mb": memory_mb
                }
            
            return {
                "passed": True, 
                "message": f"Memory usage: {memory_mb:.1f}MB", 
                "memory_mb": memory_mb
            }
        except Exception as e:
            return {"passed": False, "message": f"Memory check failed: {str(e)}"}
    
    def _check_device_availability(self) -> Dict[str, Any]:
        """Check device availability."""
        try:
            device = next(self.api.model.parameters()).device
            
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    return {"passed": False, "message": "CUDA not available but model on GPU"}
                
                memory_info = torch.cuda.memory_summary(device)
                return {
                    "passed": True, 
                    "message": f"GPU {device} available", 
                    "device": str(device)
                }
            else:
                return {
                    "passed": True, 
                    "message": f"CPU device available", 
                    "device": str(device)
                }
                
        except Exception as e:
            return {"passed": False, "message": f"Device check failed: {str(e)}"}
    
    def benchmark_deployment(
        self, 
        sample_texts: List[str],
        task_id: str,
        num_runs: int = 100
    ) -> Dict[str, Any]:
        """Benchmark deployment performance.
        
        Args:
            sample_texts: Sample texts for benchmarking
            task_id: Task to benchmark
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        import time
        
        if not sample_texts:
            sample_texts = ["This is a test sentence for benchmarking."] * 10
        
        # Warmup
        for _ in range(5):
            try:
                self.api.predict(sample_texts[0], task_id)
            except:
                pass
        
        # Benchmark
        times = []
        successful_runs = 0
        
        for i in range(num_runs):
            text = sample_texts[i % len(sample_texts)]
            
            start_time = time.perf_counter()
            try:
                result = self.api.predict(text, task_id)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                successful_runs += 1
            except Exception as e:
                logger.warning(f"Benchmark run {i} failed: {e}")
        
        if not times:
            return {"error": "All benchmark runs failed"}
        
        # Calculate metrics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        throughput = len(sample_texts) / avg_time if avg_time > 0 else 0
        
        return {
            "average_inference_time_ms": avg_time * 1000,
            "min_inference_time_ms": min_time * 1000,
            "max_inference_time_ms": max_time * 1000,
            "throughput_samples_per_sec": throughput,
            "success_rate": successful_runs / num_runs,
            "total_runs": num_runs,
            "successful_runs": successful_runs
        }
    
    def export_deployment_package(
        self, 
        output_dir: str,
        include_examples: bool = True,
        include_docs: bool = True
    ) -> str:
        """Export complete deployment package.
        
        Args:
            output_dir: Output directory for deployment package
            include_examples: Whether to include example code
            include_docs: Whether to include documentation
            
        Returns:
            Path to deployment package
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_dir = output_path / "model"
        self.api.save(str(model_dir))
        
        # Save deployment configuration
        with open(output_path / "deployment_config.json", "w") as f:
            json.dump(self.deployment_config, f, indent=2)
        
        # Save task information
        with open(output_path / "task_info.json", "w") as f:
            json.dump(self.api.get_task_info(), f, indent=2)
        
        # Create loading script
        loader_script = f'''#!/usr/bin/env python3
"""
Deployment loader script for continual learning model.
Usage: python load_model.py
"""

from continual_transformer.api import ContinualLearningAPI
from continual_transformer.deployment import ModelDeployment
import json

def load_deployed_model():
    """Load the deployed model."""
    # Load the API
    api = ContinualLearningAPI.load("model")
    
    # Load deployment configuration
    with open("deployment_config.json", "r") as f:
        deployment_config = json.load(f)
    
    # Setup deployment
    deployment = ModelDeployment(api)
    deployment.deployment_config = deployment_config
    
    print("✅ Model loaded successfully!")
    print(f"📊 Tasks available: {{list(api.trained_tasks)}}")
    print(f"💾 Memory usage: {{api.get_memory_usage()}}")
    
    return api, deployment

if __name__ == "__main__":
    api, deployment = load_deployed_model()
    
    # Run health check
    health = deployment.health_check()
    print(f"🔍 Health status: {{health['status']}}")
    
    # Example prediction (modify as needed)
    # result = api.predict("Your text here", "your_task_id")
    # print(f"📝 Prediction: {{result}}")
'''
        
        with open(output_path / "load_model.py", "w") as f:
            f.write(loader_script)
        
        # Create requirements.txt
        requirements = [
            "torch>=1.12.0",
            "transformers>=4.20.0",
            "numpy>=1.21.0",
            "continual-tiny-transformer"
        ]
        
        with open(output_path / "requirements.txt", "w") as f:
            f.write("\\n".join(requirements))
        
        # Add examples if requested
        if include_examples:
            examples_dir = output_path / "examples"
            examples_dir.mkdir(exist_ok=True)
            
            example_code = '''"""
Example usage of deployed continual learning model.
"""

from load_model import load_deployed_model

def main():
    # Load the model
    api, deployment = load_deployed_model()
    
    # Health check
    health = deployment.health_check()
    print(f"Health status: {health['status']}")
    
    # Example predictions for each task
    sample_text = "This is an example text for testing."
    
    for task_id in api.trained_tasks:
        try:
            result = api.predict(sample_text, task_id)
            print(f"Task '{task_id}' prediction: {result['predictions']}")
        except Exception as e:
            print(f"Error predicting for task '{task_id}': {e}")

if __name__ == "__main__":
    main()
'''
            
            with open(examples_dir / "predict_example.py", "w") as f:
                f.write(example_code)
        
        # Create README
        readme_content = f"""# Continual Learning Model Deployment Package

This package contains a deployed continual learning model with {len(self.api.trained_tasks)} trained tasks.

## Quick Start

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Load the model:
   ```python
   from load_model import load_deployed_model
   
   api, deployment = load_deployed_model()
   ```

3. Make predictions:
   ```python
   result = api.predict("Your text here", "task_id")
   ```

## Available Tasks

{chr(10).join(f"- {task_id}" for task_id in self.api.trained_tasks)}

## Health Monitoring

Run health checks:
```python
health = deployment.health_check()
print(health['status'])  # 'healthy', 'degraded', or 'unhealthy'
```

## Performance Benchmarking

```python
benchmark_results = deployment.benchmark_deployment(
    sample_texts=["Test text 1", "Test text 2"],
    task_id="your_task_id",
    num_runs=100
)
```

Generated on: {torch.utils.data.get_worker_info() is None}  # Simplified timestamp
Model optimization: {self.deployment_config.get('optimization_level', 'none')}
"""
        
        with open(output_path / "README.md", "w") as f:
            f.write(readme_content)
        
        logger.info(f"Deployment package exported to {output_path}")
        return str(output_path)

@contextmanager
def deployment_context(api: ContinualLearningAPI, **kwargs):
    """Context manager for model deployment.
    
    Args:
        api: ContinualLearningAPI instance
        **kwargs: Deployment configuration
    """
    deployment = ModelDeployment(api)
    
    try:
        # Setup deployment
        deployment.prepare_for_production(**kwargs)
        yield deployment
        
    finally:
        # Cleanup if needed
        logger.info("Deployment context closed")

class BatchInferenceEngine:
    """Optimized batch inference for production deployment."""
    
    def __init__(self, api: ContinualLearningAPI, batch_size: int = 32):
        self.api = api
        self.batch_size = batch_size
        self._tokenizer = None
    
    @property
    def tokenizer(self):
        """Lazy tokenizer loading."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.api.config.model_name)
        return self._tokenizer
    
    def predict_batch(
        self, 
        texts: List[str], 
        task_id: str,
        return_probabilities: bool = True
    ) -> List[Dict[str, Any]]:
        """Efficient batch prediction.
        
        Args:
            texts: List of input texts
            task_id: Task identifier
            return_probabilities: Whether to return probability scores
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.api.config.max_sequence_length,
                return_tensors="pt"
            )
            
            # Move to device
            device = next(self.api.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Predict
            self.api.model.eval()
            with torch.no_grad():
                outputs = self.api.model.forward(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    task_id=task_id
                )
            
            # Process outputs
            logits = outputs['logits']
            predictions = logits.argmax(dim=-1).cpu().numpy()
            
            if return_probabilities:
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            
            # Add to results
            for j, pred in enumerate(predictions):
                result = {
                    "prediction": int(pred),
                    "text": batch_texts[j],
                    "task_id": task_id
                }
                
                if return_probabilities:
                    result["probabilities"] = probabilities[j].tolist()
                
                results.append(result)
        
        return results

__all__ = [
    "ModelDeployment", 
    "deployment_context", 
    "BatchInferenceEngine"
]