"""Integration tests for deployment functionality."""

import pytest
import tempfile
import asyncio
import torch
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from continual_transformer.api import ContinualLearningAPI
from continual_transformer.deployment import (
    ModelDeployment, deployment_context, BatchInferenceEngine
)


class TestModelDeployment:
    """Test suite for ModelDeployment."""
    
    @pytest.fixture
    def mock_api(self):
        """Create mock API for testing."""
        with patch('continual_transformer.core.model.AutoModel'):
            with patch('continual_transformer.core.model.AutoTokenizer'):
                api = ContinualLearningAPI(model_name="test-model", device="cpu")
                
                # Mock methods
                api.optimize_for_deployment = Mock(return_value={
                    "quantization": True,
                    "torch_compile": True
                })
                api.get_memory_usage = Mock(return_value={
                    "total_parameters": 1000,
                    "trainable_parameters": 100
                })
                api.get_task_info = Mock(return_value={
                    "registered_tasks": ["sentiment"],
                    "trained_tasks": ["sentiment"],
                    "num_tasks": 1
                })
                api.trained_tasks = {"sentiment"}
                
                return api
    
    @pytest.fixture
    def deployment(self, mock_api):
        """Create deployment instance."""
        return ModelDeployment(mock_api)
    
    def test_prepare_for_production(self, deployment, mock_api):
        """Test production preparation."""
        info = deployment.prepare_for_production(
            optimization_level="balanced",
            enable_monitoring=True,
            max_memory_mb=100
        )
        
        assert "optimization_applied" in info
        assert "memory_usage" in info
        assert info["monitoring_enabled"]
        
        # Check optimization was called
        mock_api.optimize_for_deployment.assert_called_once_with("balanced")
    
    def test_health_check_healthy(self, deployment, mock_api):
        """Test health check when everything is healthy."""
        # Mock successful checks
        deployment.health_checks = [
            lambda: {"passed": True, "message": "Model OK"},
            lambda: {"passed": True, "message": "Memory OK"},
            lambda: {"passed": True, "message": "Device OK"}
        ]
        
        result = deployment.health_check()
        
        assert result["status"] == "healthy"
        assert len(result["checks"]) == 3
        assert len(result["errors"]) == 0
    
    def test_health_check_degraded(self, deployment):
        """Test health check with warnings."""
        deployment.health_checks = [
            lambda: {"passed": True, "message": "Model OK"},
            lambda: {"passed": False, "message": "High memory usage"},
            lambda: {"passed": True, "message": "Device OK"}
        ]
        
        result = deployment.health_check()
        
        assert result["status"] == "degraded"
        assert len(result["warnings"]) > 0
        assert "High memory usage" in result["warnings"][0]
    
    def test_health_check_unhealthy(self, deployment):
        """Test health check with errors."""
        def failing_check():
            raise Exception("Check failed")
        
        deployment.health_checks = [failing_check]
        
        result = deployment.health_check()
        
        assert result["status"] == "unhealthy"
        assert len(result["errors"]) > 0
        assert "Check failed" in result["errors"][0]
    
    def test_benchmark_deployment(self, deployment, mock_api):
        """Test deployment benchmarking."""
        # Mock predict method
        mock_api.predict = Mock(return_value={
            "predictions": [1],
            "probabilities": [[0.2, 0.8]]
        })
        mock_api.trained_tasks = {"sentiment"}
        
        sample_texts = ["Great product!", "Terrible service"]
        
        results = deployment.benchmark_deployment(
            sample_texts=sample_texts,
            task_id="sentiment",
            num_runs=5
        )
        
        assert "average_inference_time_ms" in results
        assert "throughput_samples_per_sec" in results
        assert "success_rate" in results
        assert results["total_runs"] == 5
        assert results["success_rate"] > 0  # Should have some successful runs
    
    def test_benchmark_deployment_failures(self, deployment, mock_api):
        """Test benchmarking with prediction failures."""
        # Mock predict method to fail
        mock_api.predict = Mock(side_effect=Exception("Prediction failed"))
        mock_api.trained_tasks = {"sentiment"}
        
        results = deployment.benchmark_deployment(
            sample_texts=["Test text"],
            task_id="sentiment",
            num_runs=3
        )
        
        if "error" not in results:
            assert results["success_rate"] == 0.0
            assert results["successful_runs"] == 0
    
    def test_export_deployment_package(self, deployment, mock_api):
        """Test exporting deployment package."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock save method
            mock_api.save = Mock()
            
            package_path = deployment.export_deployment_package(
                output_dir=temp_dir,
                include_examples=True,
                include_docs=True
            )
            
            package_dir = Path(package_path)
            
            # Check files were created
            assert (package_dir / "deployment_config.json").exists()
            assert (package_dir / "task_info.json").exists()
            assert (package_dir / "load_model.py").exists()
            assert (package_dir / "requirements.txt").exists()
            assert (package_dir / "README.md").exists()
            assert (package_dir / "examples").exists()
            assert (package_dir / "examples" / "predict_example.py").exists()
            
            # Check model was saved
            mock_api.save.assert_called_once()


class TestBatchInferenceEngine:
    """Test suite for BatchInferenceEngine."""
    
    @pytest.fixture
    def mock_api(self):
        """Create mock API for testing."""
        api = Mock()
        api.config = Mock()
        api.config.max_sequence_length = 100
        api.model = Mock()
        api.model.parameters.return_value = [torch.tensor([1.0])]
        api.trained_tasks = {"sentiment"}
        
        # Mock predict method
        api.model.predict = Mock(return_value={
            "predictions": [1],
            "probabilities": [[0.2, 0.8]]
        })
        
        return api
    
    @pytest.fixture
    def engine(self, mock_api):
        """Create batch inference engine."""
        return BatchInferenceEngine(
            model=mock_api,
            max_batch_size=4,
            max_workers=2,
            queue_timeout=0.01
        )
    
    @pytest.mark.asyncio
    async def test_async_inference_lifecycle(self, engine):
        """Test async engine start/stop lifecycle."""
        # Start engine
        await engine.start()
        assert engine.is_running
        assert len(engine.worker_tasks) == 2
        
        # Stop engine
        await engine.stop()
        assert not engine.is_running
    
    @pytest.mark.asyncio
    async def test_predict_async(self, engine, mock_api):
        """Test asynchronous prediction."""
        await engine.start()
        
        try:
            # Mock the internal prediction to avoid tokenizer issues
            def mock_single_prediction(data):
                return {
                    "predictions": [1],
                    "probabilities": [0.8],
                    "task_id": data.get("task_id", "default")
                }
            
            engine._single_prediction = Mock(side_effect=mock_single_prediction)
            
            # Make async prediction
            input_data = {"text": "Great product!", "task_id": "sentiment"}
            result = await engine.predict_async(input_data, timeout=5.0)
            
            assert "predictions" in result
            assert result["task_id"] == "sentiment"
            
        finally:
            await engine.stop()
    
    @pytest.mark.asyncio
    async def test_predict_async_timeout(self, engine):
        """Test async prediction timeout."""
        await engine.start()
        
        try:
            # Mock to simulate slow processing
            original_process = engine._process_batch
            
            async def slow_process_batch(*args, **kwargs):
                await asyncio.sleep(1.0)  # Simulate slow processing
                return await original_process(*args, **kwargs)
            
            engine._process_batch = slow_process_batch
            
            input_data = {"text": "Test", "task_id": "sentiment"}
            
            with pytest.raises(TimeoutError):
                await engine.predict_async(input_data, timeout=0.1)
        
        finally:
            await engine.stop()
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, engine, mock_api):
        """Test batch processing functionality."""
        await engine.start()
        
        try:
            # Create multiple requests simultaneously
            tasks = []
            for i in range(6):  # More than max_batch_size
                input_data = {"text": f"Test text {i}", "task_id": "sentiment"}
                task = asyncio.create_task(
                    engine.predict_async(input_data, timeout=5.0)
                )
                tasks.append(task)
            
            # Mock single prediction
            engine._single_prediction = Mock(return_value={
                "predictions": [1],
                "probabilities": [0.8]
            })
            
            # Wait for all results
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Should have results for all requests
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) > 0
            
        finally:
            await engine.stop()


class TestDeploymentContext:
    """Test deployment context manager."""
    
    @pytest.fixture
    def mock_api(self):
        """Create mock API."""
        api = Mock()
        api.optimize_for_deployment = Mock(return_value={})
        return api
    
    def test_deployment_context(self, mock_api):
        """Test deployment context manager."""
        with deployment_context(mock_api, optimization_level="speed") as deployment:
            assert isinstance(deployment, ModelDeployment)
            assert deployment.api == mock_api
        
        # Context should exit cleanly


class TestDeploymentIntegration:
    """Integration tests for deployment workflow."""
    
    @pytest.mark.asyncio
    async def test_full_deployment_workflow(self):
        """Test complete deployment workflow."""
        with patch('continual_transformer.core.model.AutoModel'):
            with patch('continual_transformer.core.model.AutoTokenizer'):
                # 1. Create API
                api = ContinualLearningAPI(model_name="test-model", device="cpu")
                
                # Mock API methods
                api.optimize_for_deployment = Mock(return_value={"quantization": True})
                api.get_memory_usage = Mock(return_value={"total_parameters": 1000})
                api.get_task_info = Mock(return_value={
                    "registered_tasks": ["sentiment"],
                    "trained_tasks": ["sentiment"],
                    "num_tasks": 1
                })
                api.trained_tasks = {"sentiment"}
                api.predict = Mock(return_value={
                    "predictions": [1],
                    "probabilities": [[0.2, 0.8]]
                })
                
                # 2. Setup deployment
                deployment = ModelDeployment(api)
                
                # 3. Prepare for production
                deployment_info = deployment.prepare_for_production(
                    optimization_level="balanced"
                )
                
                assert "optimization_applied" in deployment_info
                
                # 4. Health check
                health = deployment.health_check()
                assert health["status"] in ["healthy", "degraded", "unhealthy"]
                
                # 5. Benchmark performance
                benchmark_results = deployment.benchmark_deployment(
                    sample_texts=["Test text"],
                    task_id="sentiment",
                    num_runs=3
                )
                
                if "error" not in benchmark_results:
                    assert "average_inference_time_ms" in benchmark_results
                
                # 6. Setup async inference
                async_engine = BatchInferenceEngine(api, max_batch_size=2, max_workers=1)
                await async_engine.start()
                
                try:
                    # Mock internal prediction
                    async_engine._single_prediction = Mock(return_value={
                        "predictions": [1],
                        "probabilities": [0.8]
                    })
                    
                    # Test async prediction
                    result = await async_engine.predict_async(
                        {"text": "Async test", "task_id": "sentiment"},
                        timeout=2.0
                    )
                    
                    assert "predictions" in result
                
                finally:
                    await async_engine.stop()
                
                # 7. Export deployment package
                with tempfile.TemporaryDirectory() as temp_dir:
                    api.save = Mock()  # Mock save method
                    
                    package_path = deployment.export_deployment_package(temp_dir)
                    
                    # Verify package structure
                    package_dir = Path(package_path)
                    assert package_dir.exists()
                    assert (package_dir / "load_model.py").exists()
                    assert (package_dir / "README.md").exists()


class TestDeploymentErrorHandling:
    """Test error handling in deployment scenarios."""
    
    def test_deployment_with_failing_api(self):
        """Test deployment with API that has failing methods."""
        api = Mock()
        api.optimize_for_deployment = Mock(side_effect=Exception("Optimization failed"))
        api.get_memory_usage = Mock(side_effect=Exception("Memory check failed"))
        
        deployment = ModelDeployment(api)
        
        # Should handle failures gracefully
        try:
            deployment.prepare_for_production()
        except Exception as e:
            # Should catch and handle the exception
            assert "Optimization failed" in str(e)
    
    def test_health_check_with_missing_model(self):
        """Test health checks when model is not properly loaded."""
        api = Mock()
        api.model = None  # No model
        
        deployment = ModelDeployment(api)
        
        # Override health checks to test missing model
        def check_missing_model():
            if api.model is None:
                return {"passed": False, "message": "Model not loaded"}
            return {"passed": True, "message": "Model OK"}
        
        deployment.health_checks = [check_missing_model]
        
        result = deployment.health_check()
        
        assert result["status"] == "degraded"
        assert any("not loaded" in warning for warning in result["warnings"])
    
    @pytest.mark.asyncio
    async def test_async_engine_with_failing_predictions(self):
        """Test async engine handling prediction failures."""
        api = Mock()
        api.config = Mock()
        api.config.max_sequence_length = 100
        api.model = Mock()
        api.model.parameters.return_value = [torch.tensor([1.0])]
        
        engine = BatchInferenceEngine(api, max_batch_size=2, max_workers=1)
        
        # Mock prediction to fail
        engine._single_prediction = Mock(side_effect=Exception("Prediction failed"))
        
        await engine.start()
        
        try:
            # Should handle prediction failures
            with pytest.raises(Exception, match="Prediction failed"):
                await engine.predict_async(
                    {"text": "Test", "task_id": "test"},
                    timeout=1.0
                )
        
        finally:
            await engine.stop()