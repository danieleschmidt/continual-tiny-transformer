#!/usr/bin/env python3
"""
Production deployment example for continual learning models.
Demonstrates complete deployment workflow with monitoring and optimization.
"""

import torch
from pathlib import Path
from continual_transformer.api import ContinualLearningAPI
from continual_transformer.deployment import ModelDeployment, deployment_context, BatchInferenceEngine

def create_sample_data():
    """Create sample data for demonstration."""
    sentiment_data = [
        {"text": "I love this product!", "label": 1},
        {"text": "This is terrible.", "label": 0},
        {"text": "Amazing quality and service.", "label": 1},
        {"text": "Worst purchase ever.", "label": 0},
        {"text": "Highly recommended!", "label": 1},
    ]
    
    topic_data = [
        {"text": "The new AI model shows promising results.", "label": 0},  # tech
        {"text": "The economy is showing signs of recovery.", "label": 1},  # business
        {"text": "Scientists discover new exoplanet.", "label": 2},  # science
        {"text": "Stock market reaches new highs.", "label": 1},  # business
        {"text": "Machine learning revolutionizes healthcare.", "label": 0},  # tech
    ]
    
    return sentiment_data, topic_data

def train_sample_model():
    """Train a sample model for deployment demonstration."""
    print("🚀 Training sample continual learning model...")
    
    # Initialize API
    api = ContinualLearningAPI(
        model_name="distilbert-base-uncased",
        max_tasks=10,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create sample data
    sentiment_data, topic_data = create_sample_data()
    
    # Add tasks
    api.add_task("sentiment", num_labels=2)
    api.add_task("topic", num_labels=3)
    
    # Train tasks
    print("📝 Training sentiment analysis task...")
    sentiment_metrics = api.train_task(
        task_id="sentiment",
        train_data=sentiment_data,
        epochs=3,
        batch_size=4,
        learning_rate=2e-5
    )
    print(f"✅ Sentiment task trained - Accuracy: {sentiment_metrics['train_accuracy']:.4f}")
    
    print("📝 Training topic classification task...")
    topic_metrics = api.train_task(
        task_id="topic", 
        train_data=topic_data,
        epochs=3,
        batch_size=4,
        learning_rate=2e-5
    )
    print(f"✅ Topic task trained - Accuracy: {topic_metrics['train_accuracy']:.4f}")
    
    return api

def demonstrate_production_deployment():
    """Demonstrate complete production deployment workflow."""
    print("🏭 Production Deployment Demonstration")
    print("=" * 50)
    
    # Train or load model
    api = train_sample_model()
    
    print("\\n📊 Model Information:")
    task_info = api.get_task_info()
    print(f"   Registered tasks: {task_info['registered_tasks']}")
    print(f"   Trained tasks: {task_info['trained_tasks']}")
    print(f"   Memory usage: {task_info['memory_usage']['trainable_parameters']} trainable parameters")
    
    # Deployment preparation
    print("\\n🔧 Preparing for deployment...")
    
    with deployment_context(
        api,
        optimization_level="balanced",
        enable_monitoring=True,
        max_memory_mb=500
    ) as deployment:
        
        # Health check
        print("\\n🔍 Running health checks...")
        health = deployment.health_check()
        print(f"   Status: {health['status']}")
        
        for check_name, result in health['checks'].items():
            status = "✅" if result['passed'] else "❌"
            print(f"   {status} {check_name}: {result['message']}")
        
        if health['warnings']:
            print(f"   ⚠️  Warnings: {health['warnings']}")
        if health['errors']:
            print(f"   ❌ Errors: {health['errors']}")
        
        # Performance benchmarking
        print("\\n⚡ Running performance benchmarks...")
        
        sample_texts = [
            "This product is absolutely fantastic!",
            "The AI revolution is transforming industries.",
            "Terrible customer service experience.",
            "New breakthrough in quantum computing announced.",
            "Great value for money!"
        ]
        
        for task_id in api.trained_tasks:
            print(f"\\n   Benchmarking task: {task_id}")
            
            benchmark_results = deployment.benchmark_deployment(
                sample_texts=sample_texts,
                task_id=task_id,
                num_runs=10  # Reduced for demo
            )
            
            if "error" not in benchmark_results:
                print(f"     Average inference time: {benchmark_results['average_inference_time_ms']:.2f}ms")
                print(f"     Throughput: {benchmark_results['throughput_samples_per_sec']:.1f} samples/sec")
                print(f"     Success rate: {benchmark_results['success_rate']*100:.1f}%")
        
        # Batch inference demonstration
        print("\\n📦 Batch inference demonstration...")
        
        batch_engine = BatchInferenceEngine(api, batch_size=8)
        
        test_texts = [
            "I'm really happy with this purchase!",
            "The economic forecast looks promising.",
            "This is a disappointing product.",
            "Scientists make breakthrough in renewable energy.",
            "Excellent customer support team.",
        ]
        
        for task_id in api.trained_tasks:
            print(f"\\n   Batch predictions for {task_id}:")
            
            batch_results = batch_engine.predict_batch(
                texts=test_texts,
                task_id=task_id,
                return_probabilities=True
            )
            
            for result in batch_results[:3]:  # Show first 3 results
                pred_class = result['prediction']
                max_prob = max(result['probabilities'])
                print(f"     Text: '{result['text'][:40]}...'")
                print(f"     Prediction: Class {pred_class} (confidence: {max_prob:.3f})")
        
        # Export deployment package
        print("\\n📦 Exporting deployment package...")
        
        deployment_dir = Path("./deployment_package")
        package_path = deployment.export_deployment_package(
            output_dir=str(deployment_dir),
            include_examples=True,
            include_docs=True
        )
        
        print(f"   Package exported to: {package_path}")
        print("   Package contents:")
        for item in Path(package_path).rglob("*"):
            if item.is_file():
                print(f"     - {item.relative_to(package_path)}")
        
        # Demonstration of loading deployed model
        print("\\n🔄 Demonstrating model loading from deployment package...")
        
        try:
            # Save current model
            api.save("./temp_model_save")
            
            # Load from deployment package
            loaded_api = ContinualLearningAPI.load("./temp_model_save")
            
            # Quick test
            test_result = loaded_api.predict(
                "This is a test for the loaded model.", 
                list(loaded_api.trained_tasks)[0]
            )
            
            print(f"   ✅ Model loaded and tested successfully!")
            print(f"   Test prediction: {test_result['predictions']}")
            
        except Exception as e:
            print(f"   ❌ Model loading failed: {e}")
    
    print("\\n🎉 Production deployment demonstration complete!")
    print("\\nNext steps for actual deployment:")
    print("1. Set up production environment with proper GPU/CPU resources")
    print("2. Configure monitoring and logging systems") 
    print("3. Implement API endpoints using frameworks like FastAPI or Flask")
    print("4. Set up automated health checks and alerts")
    print("5. Configure load balancing for high availability")
    print("6. Implement proper authentication and security measures")

def demonstrate_inference_optimization():
    """Demonstrate inference optimization techniques."""
    print("\\n⚡ Inference Optimization Demonstration")
    print("=" * 50)
    
    # Create a simple model for optimization testing
    api = ContinualLearningAPI(device="cpu")  # Use CPU for demo
    api.add_task("test_task", num_labels=2)
    
    # Mock some training (simplified)
    print("🔧 Setting up model for optimization...")
    
    # Test different optimization levels
    optimization_levels = ["speed", "memory", "balanced"]
    
    for level in optimization_levels:
        print(f"\\n🎯 Testing {level} optimization...")
        
        try:
            optimizations = api.optimize_for_deployment(level)
            print(f"   Applied optimizations: {list(optimizations.keys())}")
            
            # Simple performance test
            test_text = "This is a test sentence for optimization benchmarking."
            
            import time
            start_time = time.perf_counter()
            
            # Skip actual prediction for demo if no trained model
            if api.trained_tasks:
                result = api.predict(test_text, list(api.trained_tasks)[0])
                end_time = time.perf_counter()
                inference_time = (end_time - start_time) * 1000
                print(f"   Inference time: {inference_time:.2f}ms")
            else:
                print(f"   ⚠️  No trained tasks available for timing test")
                
        except Exception as e:
            print(f"   ❌ Optimization failed: {e}")

if __name__ == "__main__":
    print("🚀 Continual Learning Production Deployment Demo")
    print("=" * 60)
    
    try:
        # Main deployment demonstration
        demonstrate_production_deployment()
        
        # Optimization demonstration
        demonstrate_inference_optimization()
        
    except KeyboardInterrupt:
        print("\\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\\n🧹 Cleaning up demo files...")
        import shutil
        for path in ["./deployment_package", "./temp_model_save"]:
            try:
                if Path(path).exists():
                    shutil.rmtree(path)
                    print(f"   Cleaned up {path}")
            except:
                pass
        
        print("✨ Demo complete!")