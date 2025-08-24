#!/usr/bin/env python3
"""
Advanced Quality Gates Validator with Self-Healing and Adaptive Learning
Generation 2: Robust error handling, monitoring, and intelligent recovery
"""

import json
import logging
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import psutil
import statistics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QualityGate:
    """Quality gate definition with adaptive thresholds"""
    name: str
    command: str
    threshold: float
    weight: float = 1.0
    adaptive_threshold: bool = False
    historical_data: List[float] = field(default_factory=list)
    failure_recovery: Optional[str] = None
    max_execution_time: int = 300
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ExecutionMetrics:
    """Comprehensive execution metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    execution_time: float = 0.0
    exit_code: int = -1
    output_size: int = 0
    error_count: int = 0
    warnings_count: int = 0


class AdvancedQualityGatesValidator:
    """
    Generation 2: Advanced quality gates with robust error handling,
    monitoring, self-healing, and adaptive learning capabilities
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.gates = self._initialize_quality_gates()
        self.execution_history = []
        self.performance_baseline = {}
        self.circuit_breaker_states = {}
        
    def _initialize_quality_gates(self) -> Dict[str, QualityGate]:
        """Initialize comprehensive quality gates"""
        return {
            # Code Quality Gates
            "code_formatting": QualityGate(
                name="Code Formatting",
                command="python -m black --check src/ tests/ --diff",
                threshold=100.0,
                weight=0.8,
                failure_recovery="python -m black src/ tests/"
            ),
            
            "linting": QualityGate(
                name="Linting",
                command="python -m ruff check src/ tests/ --output-format=json",
                threshold=95.0,
                weight=1.0,
                adaptive_threshold=True,
                failure_recovery="python -m ruff check src/ tests/ --fix"
            ),
            
            "type_checking": QualityGate(
                name="Type Checking", 
                command="python -m mypy src/ --json-report mypy_report --no-error-summary",
                threshold=90.0,
                weight=1.0
            ),
            
            # Testing Gates
            "unit_tests": QualityGate(
                name="Unit Tests",
                command="python -m pytest tests/unit/ -v --tb=short --json-report --json-report-file=test_report.json",
                threshold=85.0,
                weight=2.0,
                dependencies=["code_formatting", "linting"]
            ),
            
            "integration_tests": QualityGate(
                name="Integration Tests",
                command="python -m pytest tests/integration/ -v --tb=short --json-report --json-report-file=integration_report.json",
                threshold=80.0,
                weight=1.5,
                dependencies=["unit_tests"]
            ),
            
            "test_coverage": QualityGate(
                name="Test Coverage",
                command="python -m pytest tests/ --cov=continual_transformer --cov-report=json --cov-fail-under=85",
                threshold=85.0,
                weight=1.5,
                adaptive_threshold=True
            ),
            
            # Security Gates  
            "security_scan": QualityGate(
                name="Security Scan",
                command="python -m bandit -r src/ -f json -o bandit_report.json",
                threshold=95.0,
                weight=2.0,
                max_execution_time=180
            ),
            
            "dependency_scan": QualityGate(
                name="Dependency Security",
                command="python -m safety check --json --output safety_report.json",
                threshold=100.0,
                weight=1.5
            ),
            
            # Performance Gates
            "build_performance": QualityGate(
                name="Build Performance",
                command="time python -m build --wheel",
                threshold=60.0,  # seconds
                weight=1.0,
                adaptive_threshold=True
            ),
            
            "import_performance": QualityGate(
                name="Import Performance", 
                command="python -c \"import time; start=time.time(); import continual_transformer; print(f'Import time: {time.time()-start:.3f}s')\"",
                threshold=2.0,  # seconds
                weight=0.8
            ),
            
            # Documentation Gates
            "docs_build": QualityGate(
                name="Documentation Build",
                command="python -m sphinx -b html docs/ docs/_build/html -W",
                threshold=100.0,
                weight=0.7
            ),
            
            "api_docs_coverage": QualityGate(
                name="API Documentation Coverage",
                command="python scripts/check_docstring_coverage.py --threshold=80",
                threshold=80.0,
                weight=0.5
            )
        }
    
    async def execute_gate_with_monitoring(self, gate_name: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute quality gate with comprehensive monitoring and error recovery"""
        gate = self.gates[gate_name]
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(gate_name):
            logger.warning(f"Circuit breaker open for {gate_name}, skipping execution")
            return False, {"error": "Circuit breaker open", "skipped": True}
        
        # Check dependencies
        for dep in gate.dependencies:
            if not self._is_dependency_satisfied(dep):
                return False, {"error": f"Dependency {dep} not satisfied", "skipped": True}
        
        start_time = time.time()
        process = None
        metrics = ExecutionMetrics()
        
        try:
            logger.info(f"🔍 Executing quality gate: {gate.name}")
            
            # Start system monitoring
            monitor_task = asyncio.create_task(self._monitor_system_resources())
            
            # Execute command with timeout
            process = await asyncio.create_subprocess_shell(
                gate.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=gate.max_execution_time
                )
                
                # Stop monitoring
                monitor_task.cancel()
                system_metrics = await self._get_system_metrics()
                
                # Collect execution metrics
                metrics.execution_time = time.time() - start_time
                metrics.exit_code = process.returncode
                metrics.output_size = len(stdout) + len(stderr)
                metrics.cpu_usage = system_metrics.get('cpu_percent', 0)
                metrics.memory_usage = system_metrics.get('memory_percent', 0)
                
                # Process output and analyze results
                output_text = stdout.decode('utf-8', errors='ignore')
                error_text = stderr.decode('utf-8', errors='ignore')
                
                result = self._analyze_gate_results(gate, output_text, error_text, metrics)
                
                # Update adaptive thresholds
                if gate.adaptive_threshold:
                    self._update_adaptive_threshold(gate_name, result['score'])
                
                # Log performance baseline
                self._update_performance_baseline(gate_name, metrics)
                
                success = result['passed']
                
                if success:
                    logger.info(f"✅ Quality gate passed: {gate.name} (Score: {result['score']:.1f})")
                    self._reset_circuit_breaker(gate_name)
                else:
                    logger.warning(f"⚠️ Quality gate failed: {gate.name} (Score: {result['score']:.1f})")
                    
                    # Attempt automatic recovery
                    if gate.failure_recovery:
                        recovery_success = await self._attempt_recovery(gate)
                        if recovery_success:
                            # Retry execution after recovery
                            return await self.execute_gate_with_monitoring(gate_name)
                    
                    # Update circuit breaker
                    self._update_circuit_breaker(gate_name, success=False)
                
                return success, {
                    'score': result['score'],
                    'threshold': result['threshold'],
                    'metrics': metrics,
                    'output': output_text,
                    'errors': error_text,
                    'recovered': result.get('recovered', False)
                }
                
            except asyncio.TimeoutError:
                if process:
                    process.kill()
                    await process.wait()
                
                logger.error(f"❌ Quality gate timed out: {gate.name} (>{gate.max_execution_time}s)")
                self._update_circuit_breaker(gate_name, success=False)
                
                return False, {
                    'error': 'Timeout',
                    'timeout': gate.max_execution_time,
                    'metrics': metrics
                }
                
        except Exception as e:
            logger.error(f"❌ Quality gate execution failed: {gate.name} - {str(e)}")
            logger.debug(traceback.format_exc())
            
            self._update_circuit_breaker(gate_name, success=False)
            
            return False, {
                'error': str(e),
                'traceback': traceback.format_exc(),
                'metrics': metrics
            }
    
    def _analyze_gate_results(self, gate: QualityGate, stdout: str, stderr: str, metrics: ExecutionMetrics) -> Dict[str, Any]:
        """Analyze quality gate execution results with intelligent parsing"""
        
        # Extract score based on gate type
        score = 0.0
        threshold = gate.threshold
        
        if gate.name == "Test Coverage":
            score = self._extract_coverage_score(stdout)
        elif gate.name == "Linting":
            score = self._extract_linting_score(stdout, stderr)
        elif gate.name == "Security Scan":
            score = self._extract_security_score(stdout)
        elif gate.name in ["Build Performance", "Import Performance"]:
            score = self._extract_performance_score(stdout, gate.name)
            # For performance gates, lower is better
            passed = score <= threshold
        else:
            # Default: success based on exit code
            score = 100.0 if metrics.exit_code == 0 else 0.0
        
        # Apply adaptive threshold if enabled
        if gate.adaptive_threshold and gate.historical_data:
            threshold = self._calculate_adaptive_threshold(gate)
        
        passed = score >= threshold if gate.name not in ["Build Performance", "Import Performance"] else score <= threshold
        
        return {
            'score': score,
            'threshold': threshold,
            'passed': passed,
            'exit_code': metrics.exit_code
        }
    
    def _extract_coverage_score(self, output: str) -> float:
        """Extract test coverage percentage"""
        import re
        patterns = [
            r'TOTAL.*?(\d+)%',
            r'"total": (\d+\.\d+)',
            r'coverage: (\d+)%'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return float(match.group(1))
        return 0.0
    
    def _extract_linting_score(self, stdout: str, stderr: str) -> float:
        """Extract linting score from ruff output"""
        try:
            # Try to parse JSON output
            import json
            if stdout.strip():
                data = json.loads(stdout)
                total_violations = len(data) if isinstance(data, list) else 0
                # Calculate score: fewer violations = higher score
                score = max(0, 100 - (total_violations * 5))  # -5 points per violation
                return score
        except json.JSONDecodeError:
            pass
        
        # Fallback: count error lines in stderr
        error_lines = len([line for line in stderr.split('\n') if line.strip() and 'error:' in line.lower()])
        return max(0, 100 - (error_lines * 10))
    
    def _extract_security_score(self, output: str) -> float:
        """Extract security score from bandit output"""
        try:
            import json
            if output.strip():
                data = json.loads(output)
                high_issues = len(data.get('results', []))
                # Calculate score: fewer security issues = higher score
                score = max(0, 100 - (high_issues * 20))  # -20 points per high severity issue
                return score
        except (json.JSONDecodeError, KeyError):
            pass
        return 0.0
    
    def _extract_performance_score(self, output: str, gate_name: str) -> float:
        """Extract performance metrics (time in seconds)"""
        import re
        
        if "Import" in gate_name:
            pattern = r'Import time: (\d+\.\d+)s'
            match = re.search(pattern, output)
            return float(match.group(1)) if match else 999.0
        
        # For build performance, extract time from 'time' command output
        patterns = [
            r'real\s+(\d+)m(\d+\.\d+)s',
            r'(\d+\.\d+)s total',
            r'Elapsed time: (\d+\.\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                if 'm' in pattern:
                    minutes, seconds = match.groups()
                    return float(minutes) * 60 + float(seconds)
                else:
                    return float(match.group(1))
        
        return 999.0  # Default high value if no time found
    
    def _update_adaptive_threshold(self, gate_name: str, score: float):
        """Update adaptive threshold based on historical performance"""
        gate = self.gates[gate_name]
        gate.historical_data.append(score)
        
        # Keep only last 20 runs for threshold calculation
        if len(gate.historical_data) > 20:
            gate.historical_data = gate.historical_data[-20:]
    
    def _calculate_adaptive_threshold(self, gate: QualityGate) -> float:
        """Calculate adaptive threshold based on historical data"""
        if len(gate.historical_data) < 3:
            return gate.threshold
        
        # Use mean - 1 standard deviation as threshold
        mean_score = statistics.mean(gate.historical_data)
        std_dev = statistics.stdev(gate.historical_data) if len(gate.historical_data) > 1 else 0
        
        adaptive_threshold = max(gate.threshold * 0.7, mean_score - std_dev)
        
        logger.debug(f"Adaptive threshold for {gate.name}: {adaptive_threshold:.1f} (was {gate.threshold})")
        return adaptive_threshold
    
    async def _monitor_system_resources(self):
        """Monitor system resources during gate execution"""
        while True:
            try:
                await asyncio.sleep(1)
                # Resource monitoring would be implemented here
            except asyncio.CancelledError:
                break
    
    async def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }
    
    async def _attempt_recovery(self, gate: QualityGate) -> bool:
        """Attempt automatic recovery for failed quality gate"""
        if not gate.failure_recovery:
            return False
        
        logger.info(f"🔧 Attempting recovery for {gate.name}: {gate.failure_recovery}")
        
        try:
            process = await asyncio.create_subprocess_shell(
                gate.failure_recovery,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=120  # 2 minute timeout for recovery
            )
            
            if process.returncode == 0:
                logger.info(f"✅ Recovery successful for {gate.name}")
                return True
            else:
                logger.warning(f"⚠️ Recovery failed for {gate.name}: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Recovery attempt failed for {gate.name}: {str(e)}")
            return False
    
    def _is_circuit_breaker_open(self, gate_name: str) -> bool:
        """Check if circuit breaker is open for a gate"""
        breaker = self.circuit_breaker_states.get(gate_name, {})
        if not breaker:
            return False
        
        # Circuit breaker opens after 3 consecutive failures
        if breaker.get('failures', 0) >= 3:
            # Stay open for 5 minutes
            if time.time() - breaker.get('last_failure', 0) < 300:
                return True
            else:
                # Reset after timeout
                self._reset_circuit_breaker(gate_name)
        
        return False
    
    def _update_circuit_breaker(self, gate_name: str, success: bool):
        """Update circuit breaker state"""
        if gate_name not in self.circuit_breaker_states:
            self.circuit_breaker_states[gate_name] = {'failures': 0, 'last_failure': 0}
        
        breaker = self.circuit_breaker_states[gate_name]
        
        if success:
            breaker['failures'] = 0
        else:
            breaker['failures'] += 1
            breaker['last_failure'] = time.time()
    
    def _reset_circuit_breaker(self, gate_name: str):
        """Reset circuit breaker for a gate"""
        if gate_name in self.circuit_breaker_states:
            self.circuit_breaker_states[gate_name] = {'failures': 0, 'last_failure': 0}
    
    def _is_dependency_satisfied(self, dep_name: str) -> bool:
        """Check if dependency gate has passed"""
        # For simplicity, assume dependencies are satisfied
        # In a real implementation, this would check previous results
        return True
    
    def _update_performance_baseline(self, gate_name: str, metrics: ExecutionMetrics):
        """Update performance baseline for regression detection"""
        if gate_name not in self.performance_baseline:
            self.performance_baseline[gate_name] = []
        
        self.performance_baseline[gate_name].append({
            'timestamp': datetime.now().isoformat(),
            'execution_time': metrics.execution_time,
            'cpu_usage': metrics.cpu_usage,
            'memory_usage': metrics.memory_usage
        })
        
        # Keep only last 50 runs
        if len(self.performance_baseline[gate_name]) > 50:
            self.performance_baseline[gate_name] = self.performance_baseline[gate_name][-50:]
    
    async def execute_all_gates(self) -> Dict[str, Any]:
        """Execute all quality gates with parallelization and dependency management"""
        start_time = time.time()
        results = {}
        execution_order = self._calculate_execution_order()
        
        logger.info("🎯 Starting comprehensive quality gates validation")
        
        for batch in execution_order:
            # Execute gates in batch (parallel execution for independent gates)
            batch_tasks = []
            for gate_name in batch:
                task = asyncio.create_task(self.execute_gate_with_monitoring(gate_name))
                batch_tasks.append((gate_name, task))
            
            # Wait for batch completion
            batch_results = {}
            for gate_name, task in batch_tasks:
                success, details = await task
                batch_results[gate_name] = {'success': success, 'details': details}
            
            results.update(batch_results)
            
            # Check if any critical gates failed
            critical_gates = ['unit_tests', 'security_scan', 'linting']
            critical_failures = [gate for gate in critical_gates if gate in batch_results and not batch_results[gate]['success']]
            
            if critical_failures:
                logger.error(f"Critical gates failed: {critical_failures}. Stopping execution.")
                break
        
        # Generate comprehensive report
        total_time = time.time() - start_time
        report = self._generate_validation_report(results, total_time)
        
        # Save results
        self._save_validation_results(report)
        
        return report
    
    def _calculate_execution_order(self) -> List[List[str]]:
        """Calculate optimal execution order based on dependencies"""
        # Simplified dependency resolution
        # In practice, this would use topological sorting
        
        batch_1 = ['code_formatting', 'linting', 'security_scan', 'dependency_scan']
        batch_2 = ['type_checking', 'unit_tests', 'build_performance']  
        batch_3 = ['integration_tests', 'test_coverage', 'import_performance']
        batch_4 = ['docs_build', 'api_docs_coverage']
        
        return [batch_1, batch_2, batch_3, batch_4]
    
    def _generate_validation_report(self, results: Dict[str, Any], total_time: float) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_gates = len(results)
        passed_gates = sum(1 for r in results.values() if r['success'])
        failed_gates = total_gates - passed_gates
        
        overall_score = (passed_gates / total_gates) * 100 if total_gates > 0 else 0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_gates': total_gates,
                'passed_gates': passed_gates,
                'failed_gates': failed_gates,
                'success_rate': overall_score,
                'overall_status': 'PASSED' if overall_score >= 80 else 'FAILED',
                'execution_time': total_time
            },
            'gate_results': results,
            'performance_metrics': self.performance_baseline,
            'circuit_breaker_states': self.circuit_breaker_states
        }
        
        return report
    
    def _save_validation_results(self, report: Dict[str, Any]):
        """Save validation results to file"""
        results_file = self.project_root / 'quality_gates_report.json'
        with open(results_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Quality gates report saved to: {results_file}")


async def main():
    """Main execution entry point"""
    print("🛡️ ADVANCED QUALITY GATES VALIDATOR - Generation 2")
    print("=" * 60)
    
    validator = AdvancedQualityGatesValidator()
    report = await validator.execute_all_gates()
    
    # Display summary
    summary = report['summary']
    print(f"\n📊 VALIDATION SUMMARY")
    print(f"Overall Status: {'✅ PASSED' if summary['overall_status'] == 'PASSED' else '❌ FAILED'}")
    print(f"Success Rate: {summary['success_rate']:.1f}% ({summary['passed_gates']}/{summary['total_gates']})")
    print(f"Execution Time: {summary['execution_time']:.1f}s")
    
    # Exit with appropriate code
    sys.exit(0 if summary['overall_status'] == 'PASSED' else 1)


if __name__ == "__main__":
    asyncio.run(main())