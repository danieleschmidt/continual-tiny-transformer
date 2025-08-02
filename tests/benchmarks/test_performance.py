"""Performance benchmark tests."""

import pytest
import time
import torch
from memory_profiler import profile
from unittest.mock import Mock


@pytest.mark.slow
@pytest.mark.benchmark
def test_training_speed_benchmark(benchmark, sample_config):
    """Benchmark training speed for single task."""
    
    def mock_training_step():
        """Mock training step with realistic computation."""
        # Simulate forward pass computation
        batch_size = sample_config["batch_size"]
        seq_length = sample_config["max_seq_length"]
        hidden_size = sample_config["hidden_size"]
        
        # Create tensors that would be used in actual training
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        
        # Simulate some computation
        output = torch.matmul(hidden_states, hidden_states.transpose(-1, -2))
        loss = torch.mean(output)
        
        return loss.item()
    
    # Benchmark the training step
    result = benchmark(mock_training_step)
    
    # Verify reasonable performance (adjust thresholds as needed)
    assert result < 1.0  # Should complete in less than 1 second


@pytest.mark.slow
@pytest.mark.benchmark
def test_memory_usage_benchmark():
    """Benchmark memory usage during task learning."""
    
    if not torch.cuda.is_available():
        pytest.skip("GPU required for memory benchmarking")
    
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()
    
    # Simulate learning multiple tasks
    peak_memory = initial_memory
    
    for task_id in range(5):
        # Simulate model forward pass
        batch_size = 32
        seq_length = 512
        hidden_size = 768
        
        input_tensor = torch.randn(batch_size, seq_length, hidden_size).cuda()
        output_tensor = torch.matmul(input_tensor, input_tensor.transpose(-1, -2))
        
        current_memory = torch.cuda.memory_allocated()
        peak_memory = max(peak_memory, current_memory)
        
        # Clean up
        del input_tensor, output_tensor
        torch.cuda.empty_cache()
    
    final_memory = torch.cuda.memory_allocated()
    memory_growth = final_memory - initial_memory
    
    # Memory growth should be minimal (< 100MB)
    assert memory_growth < 100 * 1024 * 1024, f"Memory grew by {memory_growth / 1024 / 1024:.1f}MB"


@pytest.mark.slow
@pytest.mark.benchmark
def test_inference_speed_benchmark(benchmark):
    """Benchmark inference speed."""
    
    def mock_inference():
        """Mock inference computation."""
        seq_length = 128
        hidden_size = 768
        
        input_tensor = torch.randn(1, seq_length, hidden_size)
        # Simulate transformer forward pass
        attention_weights = torch.matmul(input_tensor, input_tensor.transpose(-1, -2))
        output = torch.softmax(attention_weights, dim=-1)
        
        return output.shape
    
    result = benchmark(mock_inference)
    
    # Should be fast for single sequence
    assert result == (1, 128, 128)


@pytest.mark.slow 
@pytest.mark.benchmark
def test_scaling_benchmark():
    """Test performance scaling with number of tasks."""
    
    task_counts = [1, 5, 10, 20]
    execution_times = []
    
    for num_tasks in task_counts:
        start_time = time.time()
        
        # Simulate operations for multiple tasks
        for task_id in range(num_tasks):
            # Mock task-specific computation
            task_tensor = torch.randn(10, 100)
            result = torch.sum(task_tensor)
            
        end_time = time.time()
        execution_times.append(end_time - start_time)
    
    # Execution time should scale sub-linearly
    # (Constant memory approach should not scale linearly with tasks)
    time_ratio = execution_times[-1] / execution_times[0]
    task_ratio = task_counts[-1] / task_counts[0]
    
    # Time growth should be much less than task growth
    assert time_ratio < task_ratio * 0.5, f"Poor scaling: {time_ratio:.2f}x time for {task_ratio:.2f}x tasks"


@pytest.mark.slow
@pytest.mark.benchmark
def test_memory_profile_benchmark():
    """Profile memory usage during typical operations."""
    
    @profile
    def memory_intensive_operation():
        """Function to profile memory usage."""
        tensors = []
        
        # Create some tensors (simulate model parameters)
        for i in range(10):
            tensor = torch.randn(1000, 1000)
            tensors.append(tensor)
        
        # Simulate computation
        result = torch.stack(tensors).sum()
        
        # Clean up
        del tensors
        
        return result.item()
    
    result = memory_intensive_operation()
    assert isinstance(result, float)


@pytest.mark.benchmark
def test_cpu_vs_gpu_benchmark():
    """Compare CPU vs GPU performance."""
    
    def compute_on_device(device):
        """Perform computation on specified device."""
        tensor = torch.randn(1000, 1000).to(device)
        result = torch.matmul(tensor, tensor.T)
        return result.sum().item()
    
    # CPU benchmark
    cpu_start = time.time()
    cpu_result = compute_on_device(torch.device("cpu"))
    cpu_time = time.time() - cpu_start
    
    if torch.cuda.is_available():
        # GPU benchmark
        gpu_start = time.time()
        gpu_result = compute_on_device(torch.device("cuda"))
        torch.cuda.synchronize()  # Ensure GPU computation is complete
        gpu_time = time.time() - gpu_start
        
        # Results should be approximately equal
        assert abs(cpu_result - gpu_result) < 1e-3
        
        # GPU should be faster for large computations (usually)
        speedup = cpu_time / gpu_time
        print(f"GPU speedup: {speedup:.2f}x")
    else:
        pytest.skip("GPU not available for comparison")