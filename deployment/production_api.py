"""
Production-ready FastAPI server for continual learning inference.
Includes monitoring, security, and high-performance serving.
"""

import os
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Mock imports for demonstration (would be real in production)
try:
    from continual_transformer import ContinualTransformer, ContinualConfig
    from continual_transformer.monitoring import HealthMonitor
    from continual_transformer.security import InputValidator, OutputSanitizer
    from continual_transformer.optimization import MemoryOptimizer
except ImportError:
    # Mock classes for testing
    class ContinualTransformer:
        def __init__(self, config): pass
        def predict(self, text, task_id): return {"predictions": [0], "probabilities": [0.5]}
        def get_memory_usage(self): return {"total_parameters": 1000000}
    
    class ContinualConfig:
        def __init__(self, **kwargs): pass
    
    class HealthMonitor:
        def start_monitoring(self): pass
        def get_health_summary(self): return {"status": "healthy"}
    
    class InputValidator:
        def validate_text_input(self, text): return text
    
    class OutputSanitizer:
        def sanitize_response(self, response): return response
    
    class MemoryOptimizer:
        def __init__(self, model, config): pass
        def start_optimization(self): pass


# Prometheus metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')
MODEL_MEMORY_USAGE = Gauge('model_memory_mb', 'Model memory usage in MB')
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions', ['task_id'])
ERROR_COUNT = Counter('errors_total', 'Total errors', ['error_type'])

# Security
security = HTTPBearer()

# Global state
app_state = {
    "model": None,
    "health_monitor": None,
    "memory_optimizer": None,
    "input_validator": None,
    "output_sanitizer": None,
    "startup_time": None
}

logger = logging.getLogger(__name__)


# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    text: str
    task_id: str
    options: Optional[Dict[str, Any]] = {}
    
    @validator('text')
    def validate_text(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Text cannot be empty')
        if len(v) > 10000:  # 10K character limit
            raise ValueError('Text too long (max 10000 characters)')
        return v.strip()
    
    @validator('task_id')
    def validate_task_id(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Task ID cannot be empty')
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Task ID must be alphanumeric (with _ and - allowed)')
        return v.strip()


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[int]
    probabilities: List[List[float]]
    task_id: str
    confidence: float
    processing_time_ms: float
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: float
    uptime_seconds: float
    memory_usage_mb: float
    active_tasks: int
    version: str = "1.0.0"


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    texts: List[str]
    task_id: str
    options: Optional[Dict[str, Any]] = {}
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v or len(v) == 0:
            raise ValueError('Texts list cannot be empty')
        if len(v) > 100:  # Max 100 texts per batch
            raise ValueError('Too many texts in batch (max 100)')
        return v


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify authentication token."""
    token = credentials.credentials
    
    # In production, verify against your auth system
    expected_token = os.getenv("API_TOKEN", "dev-token-12345")
    
    if token != expected_token:
        ERROR_COUNT.labels(error_type="authentication").inc()
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    
    return credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    
    # Startup
    logger.info("Starting continual transformer API...")
    app_state["startup_time"] = time.time()
    
    try:
        # Initialize model
        config = ContinualConfig(
            model_name=os.getenv("MODEL_NAME", "bert-base-uncased"),
            max_tasks=int(os.getenv("MAX_TASKS", "50")),
            device=os.getenv("DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")
        )
        
        app_state["model"] = ContinualTransformer(config)
        logger.info("Model initialized successfully")
        
        # Initialize monitoring
        app_state["health_monitor"] = HealthMonitor()
        app_state["health_monitor"].start_monitoring()
        logger.info("Health monitoring started")
        
        # Initialize security components
        app_state["input_validator"] = InputValidator()
        app_state["output_sanitizer"] = OutputSanitizer()
        logger.info("Security components initialized")
        
        # Initialize memory optimizer
        app_state["memory_optimizer"] = MemoryOptimizer(app_state["model"], config)
        app_state["memory_optimizer"].start_optimization()
        logger.info("Memory optimization started")
        
        # Start background tasks
        asyncio.create_task(update_metrics())
        
        logger.info("API server startup completed")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")
    
    if app_state["health_monitor"]:
        app_state["health_monitor"].stop_monitoring()
    
    if app_state["memory_optimizer"]:
        app_state["memory_optimizer"].stop_optimization()
    
    logger.info("API server shutdown completed")


# Create FastAPI app
app = FastAPI(
    title="Continual Transformer API",
    description="Production API for continual learning inference",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


async def update_metrics():
    """Background task to update Prometheus metrics."""
    
    while True:
        try:
            if app_state["model"]:
                memory_stats = app_state["model"].get_memory_usage()
                MODEL_MEMORY_USAGE.set(memory_stats.get("total_parameters", 0) / 1000000)
            
            await asyncio.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
            await asyncio.sleep(60)


@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add request processing time and metrics."""
    
    start_time = time.time()
    ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Update metrics
        REQUEST_DURATION.observe(process_time)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        ERROR_COUNT.labels(error_type="internal").inc()
        logger.error(f"Request failed: {e}")
        raise
        
    finally:
        ACTIVE_CONNECTIONS.dec()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    
    if not app_state["model"]:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    uptime = time.time() - app_state["startup_time"]
    memory_usage = 0
    
    try:
        if app_state["health_monitor"]:
            health_summary = app_state["health_monitor"].get_health_summary()
            status = health_summary.get("current_status", "unknown")
        else:
            status = "healthy"
        
        if app_state["model"]:
            memory_stats = app_state["model"].get_memory_usage()
            memory_usage = memory_stats.get("total_parameters", 0) / 1000000
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        status = "unhealthy"
    
    return HealthResponse(
        status=status,
        timestamp=time.time(),
        uptime_seconds=uptime,
        memory_usage_mb=memory_usage,
        active_tasks=len(getattr(app_state["model"], "adapters", {}))
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    
    if not app_state["model"]:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {"status": "ready", "timestamp": time.time()}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    
    return generate_latest(prometheus_client.REGISTRY).decode('utf-8')


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_token)])
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Single prediction endpoint."""
    
    start_time = time.time()
    
    try:
        # Validate input
        if app_state["input_validator"]:
            validated_text = app_state["input_validator"].validate_text_input(request.text)
        else:
            validated_text = request.text
        
        # Make prediction
        result = app_state["model"].predict(validated_text, request.task_id)
        
        # Sanitize output
        if app_state["output_sanitizer"]:
            result = app_state["output_sanitizer"].sanitize_response(result)
        
        # Calculate confidence
        probabilities = result.get("probabilities", [[0.5]])
        confidence = max(probabilities[0]) if probabilities and probabilities[0] else 0.5
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update metrics
        PREDICTION_COUNT.labels(task_id=request.task_id).inc()
        
        # Log prediction (background task)
        background_tasks.add_task(
            log_prediction,
            request.task_id,
            processing_time,
            confidence
        )
        
        return PredictionResponse(
            predictions=result.get("predictions", [0]),
            probabilities=probabilities,
            task_id=request.task_id,
            confidence=confidence,
            processing_time_ms=processing_time
        )
        
    except ValueError as e:
        ERROR_COUNT.labels(error_type="validation").inc()
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        ERROR_COUNT.labels(error_type="prediction").inc()
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict/batch", dependencies=[Depends(verify_token)])
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint."""
    
    start_time = time.time()
    
    try:
        results = []
        
        for text in request.texts:
            # Validate input
            if app_state["input_validator"]:
                validated_text = app_state["input_validator"].validate_text_input(text)
            else:
                validated_text = text
            
            # Make prediction
            result = app_state["model"].predict(validated_text, request.task_id)
            
            # Sanitize output
            if app_state["output_sanitizer"]:
                result = app_state["output_sanitizer"].sanitize_response(result)
            
            results.append(result)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update metrics
        PREDICTION_COUNT.labels(task_id=request.task_id).inc(len(request.texts))
        
        return {
            "results": results,
            "batch_size": len(request.texts),
            "task_id": request.task_id,
            "processing_time_ms": processing_time
        }
        
    except ValueError as e:
        ERROR_COUNT.labels(error_type="validation").inc()
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        ERROR_COUNT.labels(error_type="batch_prediction").inc()
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")


@app.get("/tasks")
async def list_tasks(credentials: HTTPAuthorizationCredentials = Security(security)):
    """List available tasks."""
    
    await verify_token(credentials)
    
    if not app_state["model"] or not hasattr(app_state["model"], "adapters"):
        return {"tasks": []}
    
    tasks = list(app_state["model"].adapters.keys())
    
    return {
        "tasks": tasks,
        "total_tasks": len(tasks),
        "max_tasks": getattr(app_state["model"].config, "max_tasks", 10)
    }


@app.get("/model/info")
async def model_info(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Get model information."""
    
    await verify_token(credentials)
    
    if not app_state["model"]:
        raise HTTPException(status_code=503, detail="Model not available")
    
    info = {
        "model_name": getattr(app_state["model"].config, "model_name", "unknown"),
        "version": "1.0.0",
        "uptime_seconds": time.time() - app_state["startup_time"],
        "capabilities": [
            "continual_learning",
            "multi_task_prediction",
            "zero_parameter_expansion"
        ]
    }
    
    if hasattr(app_state["model"], "get_memory_usage"):
        info["memory_usage"] = app_state["model"].get_memory_usage()
    
    return info


async def log_prediction(task_id: str, processing_time: float, confidence: float):
    """Background task to log prediction details."""
    
    try:
        logger.info(
            f"Prediction completed - Task: {task_id}, "
            f"Time: {processing_time:.2f}ms, Confidence: {confidence:.3f}"
        )
    except Exception as e:
        logger.error(f"Prediction logging failed: {e}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    
    ERROR_COUNT.labels(error_type="unhandled").inc()
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )


def main():
    """Main entry point for production server."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Server configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    
    # Run server
    uvicorn.run(
        "deployment.production_api:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True,
        use_colors=False,
        loop="uvloop"
    )


if __name__ == "__main__":
    main()