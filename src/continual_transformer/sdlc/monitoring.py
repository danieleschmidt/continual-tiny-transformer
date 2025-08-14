"""SDLC monitoring and metrics collection."""

import json
import logging
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import statistics
import psutil
import sqlite3

from .core import WorkflowResult, WorkflowStatus

logger = logging.getLogger(__name__)


@dataclass
class SDLCMetrics:
    """SDLC performance metrics."""
    workflow_count: int = 0
    task_count: int = 0
    success_rate: float = 0.0
    average_duration: float = 0.0
    total_duration: float = 0.0
    failed_count: int = 0
    completed_count: int = 0
    
    # Performance metrics
    throughput_per_hour: float = 0.0
    avg_queue_time: float = 0.0
    resource_utilization: float = 0.0
    
    # Quality metrics
    test_coverage: float = 0.0
    code_quality_score: float = 0.0
    security_issues: int = 0
    
    # Trend data
    last_24h_success_rate: float = 0.0
    last_24h_duration: float = 0.0
    last_7d_trend: str = "stable"


@dataclass
class TaskMetrics:
    """Individual task performance metrics."""
    task_id: str
    task_name: str
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    last_execution: Optional[float] = None
    error_patterns: List[str] = None
    
    def __post_init__(self):
        if self.error_patterns is None:
            self.error_patterns = []
    
    @property
    def success_rate(self) -> float:
        if self.execution_count == 0:
            return 0.0
        return (self.success_count / self.execution_count) * 100.0


class MetricsCollector:
    """Collects and aggregates SDLC metrics."""
    
    def __init__(self, database_path: Optional[str] = None):
        self.database_path = database_path or ":memory:"
        self.conn = sqlite3.connect(self.database_path, check_same_thread=False)
        self.lock = threading.Lock()
        
        # In-memory metrics
        self.workflow_metrics = {}
        self.task_metrics = {}
        self.execution_history = deque(maxlen=1000)
        self.system_metrics = deque(maxlen=100)
        
        # Initialize database
        self._init_database()
        
        self.logger = logging.getLogger(f"{__name__}.MetricsCollector")
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage."""
        cursor = self.conn.cursor()
        
        # Workflow executions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflow_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT NOT NULL,
                workflow_name TEXT,
                status TEXT NOT NULL,
                start_time REAL,
                end_time REAL,
                duration REAL,
                task_count INTEGER,
                success_count INTEGER,
                failure_count INTEGER,
                trigger_type TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)
        
        # Task executions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                task_name TEXT,
                workflow_id TEXT,
                status TEXT NOT NULL,
                start_time REAL,
                end_time REAL,
                duration REAL,
                exit_code INTEGER,
                output TEXT,
                error_message TEXT,
                trigger_type TEXT,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)
        
        # System metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                cpu_percent REAL,
                memory_percent REAL,
                disk_usage_percent REAL,
                active_tasks INTEGER,
                queue_size INTEGER,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)
        
        self.conn.commit()
    
    def record_workflow_execution(
        self, 
        workflow_id: str,
        workflow_name: str,
        results: Dict[str, WorkflowResult],
        trigger_type: str = "manual"
    ):
        """Record workflow execution metrics."""
        with self.lock:
            # Calculate workflow metrics
            start_times = [r.start_time for r in results.values() if r.start_time]
            end_times = [r.end_time for r in results.values() if r.end_time]
            
            workflow_start = min(start_times) if start_times else time.time()
            workflow_end = max(end_times) if end_times else time.time()
            workflow_duration = workflow_end - workflow_start
            
            success_count = sum(1 for r in results.values() if r.status == WorkflowStatus.COMPLETED)
            failure_count = len(results) - success_count
            
            # Record in database
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO workflow_executions 
                (workflow_id, workflow_name, status, start_time, end_time, duration,
                 task_count, success_count, failure_count, trigger_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                workflow_id,
                workflow_name,
                "completed" if failure_count == 0 else "failed",
                workflow_start,
                workflow_end,
                workflow_duration,
                len(results),
                success_count,
                failure_count,
                trigger_type
            ))
            
            # Record individual task results
            for task_result in results.values():
                self.record_task_execution(task_result, workflow_id, trigger_type)
            
            self.conn.commit()
            
            # Update in-memory metrics
            self.execution_history.append({
                'workflow_id': workflow_id,
                'timestamp': workflow_end,
                'duration': workflow_duration,
                'success_count': success_count,
                'failure_count': failure_count,
                'status': 'completed' if failure_count == 0 else 'failed'
            })
            
            self.logger.info(
                f"Recorded workflow execution: {workflow_name} "
                f"({success_count}/{len(results)} tasks successful, {workflow_duration:.2f}s)"
            )
    
    def record_task_execution(
        self, 
        result: WorkflowResult,
        workflow_id: str = None,
        trigger_type: str = "manual"
    ):
        """Record individual task execution metrics."""
        with self.lock:
            # Record in database
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO task_executions 
                (task_id, workflow_id, status, start_time, end_time, duration,
                 exit_code, output, error_message, trigger_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.task_id,
                workflow_id,
                result.status.value,
                result.start_time,
                result.end_time,
                result.duration,
                result.exit_code,
                result.output[:1000] if result.output else None,  # Truncate output
                result.error[:500] if result.error else None,    # Truncate error
                trigger_type
            ))
            self.conn.commit()
            
            # Update task metrics
            if result.task_id not in self.task_metrics:
                self.task_metrics[result.task_id] = TaskMetrics(
                    task_id=result.task_id,
                    task_name=result.task_id  # Would extract from task object in production
                )
            
            task_metric = self.task_metrics[result.task_id]
            task_metric.execution_count += 1
            task_metric.last_execution = result.end_time
            
            if result.status == WorkflowStatus.COMPLETED:
                task_metric.success_count += 1
            else:
                task_metric.failure_count += 1
                if result.error and result.error not in task_metric.error_patterns:
                    task_metric.error_patterns.append(result.error[:100])
            
            # Update duration metrics
            if result.duration > 0:
                if task_metric.execution_count == 1:
                    task_metric.avg_duration = result.duration
                    task_metric.min_duration = result.duration
                    task_metric.max_duration = result.duration
                else:
                    # Running average
                    task_metric.avg_duration = (
                        (task_metric.avg_duration * (task_metric.execution_count - 1) + result.duration) /
                        task_metric.execution_count
                    )
                    task_metric.min_duration = min(task_metric.min_duration, result.duration)
                    task_metric.max_duration = max(task_metric.max_duration, result.duration)
    
    def record_system_metrics(self, active_tasks: int = 0, queue_size: int = 0):
        """Record system performance metrics."""
        with self.lock:
            timestamp = time.time()
            
            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Record in database
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO system_metrics 
                (timestamp, cpu_percent, memory_percent, disk_usage_percent, active_tasks, queue_size)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                cpu_percent,
                memory.percent,
                disk.percent,
                active_tasks,
                queue_size
            ))
            self.conn.commit()
            
            # Update in-memory metrics
            self.system_metrics.append({
                'timestamp': timestamp,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_usage_percent': disk.percent,
                'active_tasks': active_tasks,
                'queue_size': queue_size
            })
    
    def get_sdlc_metrics(self, time_window_hours: int = 24) -> SDLCMetrics:
        """Get comprehensive SDLC metrics."""
        with self.lock:
            current_time = time.time()
            window_start = current_time - (time_window_hours * 3600)
            
            cursor = self.conn.cursor()
            
            # Get workflow metrics
            cursor.execute("""
                SELECT COUNT(*), AVG(duration), SUM(success_count), SUM(failure_count), SUM(task_count)
                FROM workflow_executions 
                WHERE created_at >= ?
            """, (window_start,))
            
            workflow_data = cursor.fetchone()
            workflow_count = workflow_data[0] or 0
            avg_duration = workflow_data[1] or 0.0
            success_count = workflow_data[2] or 0
            failure_count = workflow_data[3] or 0
            total_tasks = workflow_data[4] or 0
            
            # Calculate success rate
            total_executions = success_count + failure_count
            success_rate = (success_count / total_executions * 100) if total_executions > 0 else 0.0
            
            # Get throughput
            throughput = workflow_count / max(time_window_hours, 1)
            
            # Get system metrics
            avg_cpu = 0.0
            if self.system_metrics:
                avg_cpu = statistics.mean(m['cpu_percent'] for m in self.system_metrics)
            
            return SDLCMetrics(
                workflow_count=workflow_count,
                task_count=total_tasks,
                success_rate=success_rate,
                average_duration=avg_duration,
                total_duration=avg_duration * workflow_count,
                failed_count=failure_count,
                completed_count=success_count,
                throughput_per_hour=throughput,
                resource_utilization=avg_cpu,
                last_24h_success_rate=success_rate,
                last_24h_duration=avg_duration
            )
    
    def get_task_metrics(self, task_id: Optional[str] = None) -> Union[TaskMetrics, Dict[str, TaskMetrics]]:
        """Get task-specific metrics."""
        with self.lock:
            if task_id:
                return self.task_metrics.get(task_id)
            return self.task_metrics.copy()
    
    def get_failure_analysis(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get failure analysis and common error patterns."""
        with self.lock:
            current_time = time.time()
            window_start = current_time - (time_window_hours * 3600)
            
            cursor = self.conn.cursor()
            
            # Get failed tasks with errors
            cursor.execute("""
                SELECT task_id, error_message, COUNT(*) as count
                FROM task_executions 
                WHERE status = 'failed' AND created_at >= ? AND error_message IS NOT NULL
                GROUP BY task_id, error_message
                ORDER BY count DESC
                LIMIT 10
            """, (window_start,))
            
            error_patterns = []
            for row in cursor.fetchall():
                error_patterns.append({
                    'task_id': row[0],
                    'error_message': row[1],
                    'occurrence_count': row[2]
                })
            
            # Get failure trends
            cursor.execute("""
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failures
                FROM task_executions 
                WHERE created_at >= ?
                GROUP BY DATE(created_at)
                ORDER BY date DESC
                LIMIT 7
            """, (window_start,))
            
            daily_trends = []
            for row in cursor.fetchall():
                total = row[1]
                failures = row[2]
                failure_rate = (failures / total * 100) if total > 0 else 0.0
                
                daily_trends.append({
                    'date': row[0],
                    'total_tasks': total,
                    'failed_tasks': failures,
                    'failure_rate': failure_rate
                })
            
            return {
                'common_error_patterns': error_patterns,
                'daily_failure_trends': daily_trends,
                'total_error_patterns': len(error_patterns)
            }
    
    def export_metrics(self, output_path: str, format: str = "json"):
        """Export metrics to file."""
        metrics_data = {
            'sdlc_metrics': asdict(self.get_sdlc_metrics()),
            'task_metrics': {k: asdict(v) for k, v in self.get_task_metrics().items()},
            'failure_analysis': self.get_failure_analysis(),
            'export_timestamp': time.time()
        }
        
        output_path = Path(output_path)
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Metrics exported to {output_path}")
    
    def cleanup(self, days_to_keep: int = 30):
        """Clean up old metrics data."""
        with self.lock:
            cutoff_time = time.time() - (days_to_keep * 24 * 3600)
            
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM workflow_executions WHERE created_at < ?", (cutoff_time,))
            cursor.execute("DELETE FROM task_executions WHERE created_at < ?", (cutoff_time,))
            cursor.execute("DELETE FROM system_metrics WHERE created_at < ?", (cutoff_time,))
            
            deleted_count = cursor.rowcount
            self.conn.commit()
            
            self.logger.info(f"Cleaned up {deleted_count} old metric records")
    
    def close(self):
        """Close database connection."""
        self.conn.close()


class SDLCMonitor:
    """High-level SDLC monitoring system."""
    
    def __init__(self, project_path: str, database_path: Optional[str] = None):
        self.project_path = Path(project_path)
        self.metrics_collector = MetricsCollector(database_path)
        self.monitoring_active = False
        self.monitor_thread = None
        self.alert_thresholds = {
            'failure_rate_threshold': 20.0,  # %
            'avg_duration_threshold': 600.0,  # seconds
            'cpu_threshold': 80.0,  # %
            'memory_threshold': 85.0  # %
        }
        self.logger = logging.getLogger(f"{__name__}.SDLCMonitor")
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start background monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self.collect_system_metrics()
                    self.check_alerts()
                    time.sleep(interval_seconds)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
                    time.sleep(interval_seconds)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("SDLC monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("SDLC monitoring stopped")
    
    def collect_system_metrics(self):
        """Collect current system metrics."""
        # This would be called by task runner to report active tasks
        self.metrics_collector.record_system_metrics(
            active_tasks=0,  # Would get from task runner
            queue_size=0     # Would get from task runner
        )
    
    def check_alerts(self):
        """Check for alert conditions."""
        try:
            metrics = self.metrics_collector.get_sdlc_metrics(time_window_hours=1)
            
            alerts = []
            
            # Check failure rate
            if metrics.success_rate < (100 - self.alert_thresholds['failure_rate_threshold']):
                alerts.append(f"High failure rate: {100 - metrics.success_rate:.1f}%")
            
            # Check duration
            if metrics.average_duration > self.alert_thresholds['avg_duration_threshold']:
                alerts.append(f"High average duration: {metrics.average_duration:.1f}s")
            
            # Check system resources
            if metrics.resource_utilization > self.alert_thresholds['cpu_threshold']:
                alerts.append(f"High CPU usage: {metrics.resource_utilization:.1f}%")
            
            if alerts:
                self.logger.warning(f"SDLC Alerts: {'; '.join(alerts)}")
                
        except Exception as e:
            self.logger.error(f"Alert check failed: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        return {
            'metrics': asdict(self.metrics_collector.get_sdlc_metrics()),
            'task_metrics': {k: asdict(v) for k, v in self.metrics_collector.get_task_metrics().items()},
            'failure_analysis': self.metrics_collector.get_failure_analysis(),
            'system_status': {
                'monitoring_active': self.monitoring_active,
                'alert_thresholds': self.alert_thresholds
            }
        }
    
    def generate_report(self, output_path: str):
        """Generate comprehensive SDLC report."""
        report_data = self.get_dashboard_data()
        
        # Add summary statistics
        metrics = report_data['metrics']
        report_data['summary'] = {
            'overall_health': 'good' if metrics['success_rate'] > 80 else 'needs_attention',
            'efficiency_score': min(100, (metrics['success_rate'] * metrics['throughput_per_hour']) / 10),
            'recommendations': self._generate_recommendations(metrics)
        }
        
        # Export report
        self.metrics_collector.export_metrics(output_path)
        self.logger.info(f"SDLC report generated: {output_path}")
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if metrics['success_rate'] < 90:
            recommendations.append("Improve test coverage and error handling")
        
        if metrics['average_duration'] > 300:
            recommendations.append("Optimize slow tasks and consider parallelization")
        
        if metrics['throughput_per_hour'] < 1:
            recommendations.append("Increase automation and reduce manual interventions")
        
        return recommendations
    
    def cleanup(self):
        """Cleanup monitoring resources."""
        self.stop_monitoring()
        self.metrics_collector.close()