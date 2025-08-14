"""Reliability and resilience components for SDLC automation."""

import asyncio
import json
import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import hashlib
import pickle
import uuid

from .core import WorkflowTask, WorkflowResult, WorkflowStatus

logger = logging.getLogger(__name__)


class FailureMode(Enum):
    """Types of failures that can occur in SDLC."""
    TASK_TIMEOUT = "task_timeout"
    COMMAND_ERROR = "command_error"
    DEPENDENCY_FAILURE = "dependency_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_ERROR = "network_error"
    DISK_FULL = "disk_full"
    PERMISSION_ERROR = "permission_error"
    UNKNOWN_ERROR = "unknown_error"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure modes."""
    RETRY_IMMEDIATE = "retry_immediate"
    RETRY_EXPONENTIAL = "retry_exponential"
    SKIP_TASK = "skip_task"
    ABORT_WORKFLOW = "abort_workflow"
    FALLBACK_COMMAND = "fallback_command"
    ISOLATE_AND_CONTINUE = "isolate_and_continue"


@dataclass
class FailurePattern:
    """Pattern matching for failure detection."""
    failure_mode: FailureMode
    error_patterns: List[str]
    exit_codes: List[int]
    recovery_strategy: RecoveryStrategy
    max_retries: int = 3
    backoff_multiplier: float = 2.0
    fallback_command: Optional[str] = None
    
    def matches(self, error_message: str, exit_code: int) -> bool:
        """Check if failure matches this pattern."""
        # Check exit codes
        if self.exit_codes and exit_code in self.exit_codes:
            return True
        
        # Check error patterns
        if self.error_patterns and error_message:
            error_lower = error_message.lower()
            return any(pattern.lower() in error_lower for pattern in self.error_patterns)
        
        return False


@dataclass 
class CheckpointData:
    """Data stored at workflow checkpoints."""
    checkpoint_id: str
    workflow_id: str
    task_id: str
    timestamp: float
    task_state: Dict[str, Any]
    system_state: Dict[str, Any]
    recovery_metadata: Dict[str, Any]


class CircuitBreaker:
    """Circuit breaker pattern for task execution."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half_open"
                    self.logger.info("Circuit breaker moving to half-open state")
                else:
                    raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            
            with self.lock:
                if self.state == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                    self.logger.info("Circuit breaker reset to CLOSED state")
            
            return result
            
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    self.logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
                
            raise e
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self.lock:
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "last_failure_time": self.last_failure_time
            }


class HealthChecker:
    """System health monitoring and validation."""
    
    def __init__(self):
        self.health_checks = {}
        self.last_check_results = {}
        self.logger = logging.getLogger(f"{__name__}.HealthChecker")
    
    def register_check(self, name: str, check_func: Callable[[], bool], interval: int = 60):
        """Register a health check function."""
        self.health_checks[name] = {
            'func': check_func,
            'interval': interval,
            'last_run': 0,
            'last_result': True
        }
        self.logger.info(f"Registered health check: {name}")
    
    def run_checks(self) -> Dict[str, bool]:
        """Run all health checks."""
        results = {}
        current_time = time.time()
        
        for name, check_config in self.health_checks.items():
            if current_time - check_config['last_run'] >= check_config['interval']:
                try:
                    result = check_config['func']()
                    check_config['last_result'] = result
                    check_config['last_run'] = current_time
                    results[name] = result
                    
                    if not result:
                        self.logger.warning(f"Health check failed: {name}")
                        
                except Exception as e:
                    self.logger.error(f"Health check error for {name}: {e}")
                    check_config['last_result'] = False
                    results[name] = False
            else:
                results[name] = check_config['last_result']
        
        self.last_check_results = results
        return results
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        results = self.run_checks()
        return all(results.values())
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get detailed health report."""
        self.run_checks()
        
        return {
            'overall_healthy': self.is_healthy(),
            'individual_checks': self.last_check_results.copy(),
            'registered_checks': list(self.health_checks.keys()),
            'last_check_time': time.time()
        }


class ReliabilityManager:
    """Main reliability and resilience management system."""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.failure_patterns = self._init_failure_patterns()
        self.circuit_breakers = {}
        self.checkpoints = {}
        self.health_checker = HealthChecker()
        
        # Recovery state
        self.recovery_history = []
        self.recovery_stats = {
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'by_failure_mode': {}
        }
        
        self.logger = logging.getLogger(f"{__name__}.ReliabilityManager")
        
        # Initialize system health checks
        self._init_health_checks()
    
    def _init_failure_patterns(self) -> List[FailurePattern]:
        """Initialize common failure patterns and recovery strategies."""
        return [
            # Timeout failures
            FailurePattern(
                failure_mode=FailureMode.TASK_TIMEOUT,
                error_patterns=["timeout", "time out", "timed out"],
                exit_codes=[124, 143],  # timeout and sigterm
                recovery_strategy=RecoveryStrategy.RETRY_EXPONENTIAL,
                max_retries=2
            ),
            
            # Permission errors
            FailurePattern(
                failure_mode=FailureMode.PERMISSION_ERROR,
                error_patterns=["permission denied", "access denied", "not permitted"],
                exit_codes=[126, 1],
                recovery_strategy=RecoveryStrategy.FALLBACK_COMMAND,
                max_retries=1,
                fallback_command="sudo make install-dev"  # Example fallback
            ),
            
            # Disk space errors
            FailurePattern(
                failure_mode=FailureMode.DISK_FULL,
                error_patterns=["no space left", "disk full", "out of space"],
                exit_codes=[28],  # ENOSPC
                recovery_strategy=RecoveryStrategy.ISOLATE_AND_CONTINUE,
                max_retries=0
            ),
            
            # Network errors
            FailurePattern(
                failure_mode=FailureMode.NETWORK_ERROR,
                error_patterns=["network", "connection", "timeout", "unreachable"],
                exit_codes=[],
                recovery_strategy=RecoveryStrategy.RETRY_EXPONENTIAL,
                max_retries=3,
                backoff_multiplier=1.5
            ),
            
            # Dependency failures
            FailurePattern(
                failure_mode=FailureMode.DEPENDENCY_FAILURE,
                error_patterns=["module not found", "import error", "package not found"],
                exit_codes=[1, 2],
                recovery_strategy=RecoveryStrategy.FALLBACK_COMMAND,
                fallback_command="make install-dev"
            ),
            
            # Generic command errors
            FailurePattern(
                failure_mode=FailureMode.COMMAND_ERROR,
                error_patterns=[],
                exit_codes=[1, 2, 127],  # Generic errors and command not found
                recovery_strategy=RecoveryStrategy.RETRY_IMMEDIATE,
                max_retries=1
            )
        ]
    
    def _init_health_checks(self):
        """Initialize system health checks."""
        # Disk space check
        def check_disk_space() -> bool:
            try:
                import shutil
                total, used, free = shutil.disk_usage(str(self.project_path))
                free_percent = (free / total) * 100
                return free_percent > 10  # At least 10% free space
            except Exception:
                return False
        
        # Python environment check  
        def check_python_env() -> bool:
            try:
                import sys
                return sys.executable is not None
            except Exception:
                return False
        
        # Git repository check
        def check_git_repo() -> bool:
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "status"], 
                    cwd=str(self.project_path),
                    capture_output=True,
                    timeout=10
                )
                return result.returncode == 0
            except Exception:
                return False
        
        # Network connectivity check
        def check_network() -> bool:
            try:
                import socket
                socket.create_connection(("8.8.8.8", 53), timeout=5).close()
                return True
            except Exception:
                return False
        
        self.health_checker.register_check("disk_space", check_disk_space, 300)  # 5 min
        self.health_checker.register_check("python_env", check_python_env, 600)  # 10 min
        self.health_checker.register_check("git_repo", check_git_repo, 300)      # 5 min
        self.health_checker.register_check("network", check_network, 180)        # 3 min
    
    def get_circuit_breaker(self, task_id: str) -> CircuitBreaker:
        """Get or create circuit breaker for task."""
        if task_id not in self.circuit_breakers:
            self.circuit_breakers[task_id] = CircuitBreaker()
        return self.circuit_breakers[task_id]
    
    def create_checkpoint(self, workflow_id: str, task_id: str, task_state: Dict[str, Any]) -> str:
        """Create recovery checkpoint."""
        checkpoint_id = f"cp_{uuid.uuid4().hex[:8]}"
        
        checkpoint = CheckpointData(
            checkpoint_id=checkpoint_id,
            workflow_id=workflow_id,
            task_id=task_id,
            timestamp=time.time(),
            task_state=task_state,
            system_state=self._capture_system_state(),
            recovery_metadata={}
        )
        
        self.checkpoints[checkpoint_id] = checkpoint
        self.logger.info(f"Created checkpoint {checkpoint_id} for task {task_id}")
        
        return checkpoint_id
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for recovery."""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'timestamp': time.time()
            }
        except Exception:
            return {'timestamp': time.time()}
    
    def analyze_failure(self, result: WorkflowResult) -> Optional[FailurePattern]:
        """Analyze failure and match to known patterns."""
        if result.status == WorkflowStatus.COMPLETED:
            return None
        
        error_message = result.error or ""
        exit_code = result.exit_code
        
        for pattern in self.failure_patterns:
            if pattern.matches(error_message, exit_code):
                self.logger.info(f"Matched failure pattern: {pattern.failure_mode}")
                return pattern
        
        # Default to unknown error pattern
        return FailurePattern(
            failure_mode=FailureMode.UNKNOWN_ERROR,
            error_patterns=[],
            exit_codes=[],
            recovery_strategy=RecoveryStrategy.RETRY_IMMEDIATE,
            max_retries=1
        )
    
    def attempt_recovery(
        self, 
        task: WorkflowTask, 
        failure_result: WorkflowResult,
        attempt_number: int = 1
    ) -> tuple[bool, Optional[WorkflowTask]]:
        """Attempt to recover from task failure."""
        failure_pattern = self.analyze_failure(failure_result)
        
        if not failure_pattern or attempt_number > failure_pattern.max_retries:
            self.logger.error(f"Recovery exhausted for task {task.id}")
            return False, None
        
        recovery_start = time.time()
        strategy = failure_pattern.recovery_strategy
        
        try:
            if strategy == RecoveryStrategy.RETRY_IMMEDIATE:
                self.logger.info(f"Immediate retry for task {task.id}")
                return True, task
            
            elif strategy == RecoveryStrategy.RETRY_EXPONENTIAL:
                backoff_time = (failure_pattern.backoff_multiplier ** (attempt_number - 1))
                self.logger.info(f"Exponential backoff retry for task {task.id} (wait {backoff_time}s)")
                time.sleep(backoff_time)
                return True, task
            
            elif strategy == RecoveryStrategy.FALLBACK_COMMAND:
                if failure_pattern.fallback_command:
                    self.logger.info(f"Fallback command for task {task.id}: {failure_pattern.fallback_command}")
                    
                    fallback_task = WorkflowTask(
                        id=f"{task.id}_fallback",
                        name=f"{task.name} (Fallback)",
                        command=failure_pattern.fallback_command,
                        description=f"Fallback for failed task: {task.name}",
                        priority=task.priority,
                        timeout=task.timeout,
                        environment=task.environment,
                        working_dir=task.working_dir
                    )
                    return True, fallback_task
                else:
                    return False, None
            
            elif strategy == RecoveryStrategy.SKIP_TASK:
                self.logger.warning(f"Skipping failed task {task.id}")
                return False, None
            
            elif strategy == RecoveryStrategy.ISOLATE_AND_CONTINUE:
                self.logger.warning(f"Isolating failed task {task.id} and continuing")
                # Mark task as isolated but allow workflow to continue
                return False, None
            
            elif strategy == RecoveryStrategy.ABORT_WORKFLOW:
                self.logger.error(f"Aborting workflow due to critical failure in task {task.id}")
                raise Exception(f"Workflow aborted due to critical failure: {failure_pattern.failure_mode}")
            
            else:
                return False, None
                
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            return False, None
        
        finally:
            recovery_duration = time.time() - recovery_start
            
            # Update recovery statistics
            self.recovery_stats['total_recoveries'] += 1
            mode_key = failure_pattern.failure_mode.value
            if mode_key not in self.recovery_stats['by_failure_mode']:
                self.recovery_stats['by_failure_mode'][mode_key] = 0
            self.recovery_stats['by_failure_mode'][mode_key] += 1
            
            # Record recovery attempt
            self.recovery_history.append({
                'task_id': task.id,
                'failure_mode': failure_pattern.failure_mode.value,
                'recovery_strategy': strategy.value,
                'attempt_number': attempt_number,
                'recovery_duration': recovery_duration,
                'timestamp': time.time()
            })
    
    def get_reliability_report(self) -> Dict[str, Any]:
        """Get comprehensive reliability report."""
        health_report = self.health_checker.get_health_report()
        
        circuit_breaker_status = {}
        for task_id, cb in self.circuit_breakers.items():
            circuit_breaker_status[task_id] = cb.get_state()
        
        return {
            'system_health': health_report,
            'circuit_breakers': circuit_breaker_status,
            'recovery_stats': self.recovery_stats,
            'active_checkpoints': len(self.checkpoints),
            'failure_patterns': len(self.failure_patterns),
            'recent_recoveries': self.recovery_history[-10:],  # Last 10 recoveries
            'reliability_score': self._calculate_reliability_score()
        }
    
    def _calculate_reliability_score(self) -> float:
        """Calculate overall reliability score (0-100)."""
        score = 100.0
        
        # Health check penalty
        health_report = self.health_checker.get_health_report()
        healthy_checks = sum(1 for h in health_report['individual_checks'].values() if h)
        total_checks = len(health_report['individual_checks'])
        if total_checks > 0:
            health_score = (healthy_checks / total_checks) * 30
            score = score - 30 + health_score
        
        # Circuit breaker penalty
        open_circuit_breakers = sum(
            1 for cb in self.circuit_breakers.values() 
            if cb.get_state()['state'] == 'open'
        )
        if open_circuit_breakers > 0:
            score -= min(20, open_circuit_breakers * 10)
        
        # Recovery success rate
        if self.recovery_stats['total_recoveries'] > 0:
            recovery_rate = (
                self.recovery_stats['successful_recoveries'] / 
                self.recovery_stats['total_recoveries']
            ) * 30
            score = score - 30 + recovery_rate
        
        return max(0.0, min(100.0, score))
    
    def perform_system_recovery(self) -> bool:
        """Perform comprehensive system recovery."""
        self.logger.info("Starting system recovery")
        
        try:
            # 1. Check system health
            if not self.health_checker.is_healthy():
                self.logger.warning("System health checks failing during recovery")
            
            # 2. Reset circuit breakers in open state
            reset_count = 0
            for task_id, cb in self.circuit_breakers.items():
                if cb.get_state()['state'] == 'open':
                    with cb.lock:
                        cb.state = 'closed'
                        cb.failure_count = 0
                    reset_count += 1
                    self.logger.info(f"Reset circuit breaker for task {task_id}")
            
            # 3. Clean old checkpoints
            current_time = time.time()
            old_checkpoints = [
                cp_id for cp_id, cp in self.checkpoints.items()
                if current_time - cp.timestamp > 3600  # 1 hour old
            ]
            for cp_id in old_checkpoints:
                del self.checkpoints[cp_id]
            
            self.logger.info(
                f"System recovery completed: reset {reset_count} circuit breakers, "
                f"cleaned {len(old_checkpoints)} old checkpoints"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"System recovery failed: {e}")
            return False
    
    def export_reliability_data(self, output_path: str):
        """Export reliability data for analysis."""
        data = {
            'reliability_report': self.get_reliability_report(),
            'recovery_history': self.recovery_history,
            'failure_patterns': [
                {
                    'failure_mode': fp.failure_mode.value,
                    'recovery_strategy': fp.recovery_strategy.value,
                    'max_retries': fp.max_retries,
                    'error_patterns': fp.error_patterns,
                    'exit_codes': fp.exit_codes
                }
                for fp in self.failure_patterns
            ],
            'export_timestamp': time.time()
        }
        
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Reliability data exported to {output_path}")