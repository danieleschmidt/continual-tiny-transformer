#!/usr/bin/env python3
"""
SDLC Monitoring and Observability Framework
Generation 2: Advanced monitoring, alerting, and performance analytics
"""

import asyncio
import json
import logging
import time
import psutil
import statistics
import subprocess
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: datetime
    value: float
    metric_type: str
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert definition and state"""
    name: str
    condition: Callable[[List[PerformanceMetric]], bool]
    threshold: float
    severity: str = "warning"  # warning, error, critical
    cooldown: int = 300  # 5 minutes cooldown
    last_triggered: Optional[datetime] = None
    is_active: bool = False


class SDLCMonitor:
    """
    Advanced SDLC monitoring system with real-time analytics,
    alerting, and performance trend analysis
    """
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: Dict[str, Alert] = {}
        self.monitoring_active = False
        self.collection_interval = 10  # seconds
        self.performance_baselines = {}
        
        self._initialize_alerts()
        logger.info("SDLC Monitor initialized")
    
    def _initialize_alerts(self):
        """Initialize monitoring alerts"""
        
        # Build performance alerts
        self.alerts["build_time_regression"] = Alert(
            name="Build Time Regression",
            condition=lambda metrics: self._check_regression(metrics, threshold=1.5),
            threshold=1.5,  # 50% increase
            severity="warning"
        )
        
        self.alerts["test_failure_rate"] = Alert(
            name="Test Failure Rate High",
            condition=lambda metrics: self._check_failure_rate(metrics, threshold=10.0),
            threshold=10.0,  # 10% failure rate
            severity="error"
        )
        
        self.alerts["security_vulnerabilities"] = Alert(
            name="Security Vulnerabilities Detected",
            condition=lambda metrics: self._check_security_issues(metrics),
            threshold=0.0,
            severity="critical"
        )
        
        self.alerts["coverage_drop"] = Alert(
            name="Test Coverage Drop",
            condition=lambda metrics: self._check_coverage_drop(metrics, threshold=5.0),
            threshold=5.0,  # 5% drop
            severity="warning"
        )
        
        self.alerts["memory_usage_high"] = Alert(
            name="High Memory Usage",
            condition=lambda metrics: self._check_resource_usage(metrics, "memory", 80.0),
            threshold=80.0,  # 80% memory usage
            severity="warning"
        )
        
        self.alerts["cpu_usage_high"] = Alert(
            name="High CPU Usage",
            condition=lambda metrics: self._check_resource_usage(metrics, "cpu", 90.0),
            threshold=90.0,  # 90% CPU usage
            severity="warning"
        )
    
    def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start monitoring threads
        threading.Thread(target=self._collect_system_metrics, daemon=True).start()
        threading.Thread(target=self._collect_git_metrics, daemon=True).start()
        threading.Thread(target=self._monitor_alerts, daemon=True).start()
        
        logger.info("🔍 SDLC monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        logger.info("📊 SDLC monitoring stopped")
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        while self.monitoring_active:
            try:
                timestamp = datetime.now()
                
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                disk_info = psutil.disk_usage('/')
                
                self.add_metric("cpu_usage", cpu_percent, timestamp, {"type": "system"})
                self.add_metric("memory_usage", memory_info.percent, timestamp, {"type": "system"})
                self.add_metric("disk_usage", disk_info.percent, timestamp, {"type": "system"})
                
                # Process-specific metrics if available
                try:
                    current_process = psutil.Process()
                    self.add_metric("process_cpu", current_process.cpu_percent(), timestamp, {"type": "process"})
                    self.add_metric("process_memory", current_process.memory_info().rss / 1024 / 1024, timestamp, {"type": "process"})  # MB
                except psutil.NoSuchProcess:
                    pass
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_git_metrics(self):
        """Collect Git repository metrics"""
        while self.monitoring_active:
            try:
                timestamp = datetime.now()
                
                # Commit frequency
                commit_count = self._get_recent_commit_count(days=1)
                self.add_metric("commits_per_day", commit_count, timestamp, {"type": "git"})
                
                # Repository size
                repo_size = self._get_repository_size()
                self.add_metric("repository_size", repo_size, timestamp, {"type": "git"})
                
                # Branch count
                branch_count = self._get_branch_count()
                self.add_metric("branch_count", branch_count, timestamp, {"type": "git"})
                
                time.sleep(300)  # Collect git metrics every 5 minutes
                
            except Exception as e:
                logger.error(f"Error collecting git metrics: {e}")
                time.sleep(300)
    
    def _monitor_alerts(self):
        """Monitor alerts and trigger notifications"""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                
                for alert_name, alert in self.alerts.items():
                    # Check cooldown
                    if alert.last_triggered and (current_time - alert.last_triggered).seconds < alert.cooldown:
                        continue
                    
                    # Get relevant metrics
                    relevant_metrics = self._get_relevant_metrics_for_alert(alert_name)
                    
                    # Check condition
                    if alert.condition(relevant_metrics):
                        if not alert.is_active:
                            self._trigger_alert(alert)
                    else:
                        if alert.is_active:
                            self._resolve_alert(alert)
                
                time.sleep(30)  # Check alerts every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring alerts: {e}")
                time.sleep(30)
    
    def add_metric(self, metric_name: str, value: float, timestamp: datetime = None, tags: Dict[str, str] = None):
        """Add a performance metric"""
        if timestamp is None:
            timestamp = datetime.now()
        
        metric = PerformanceMetric(
            timestamp=timestamp,
            value=value,
            metric_type=metric_name,
            tags=tags or {}
        )
        
        self.metrics[metric_name].append(metric)
    
    def get_metrics(self, metric_name: str, hours: int = 24) -> List[PerformanceMetric]:
        """Get metrics for a specific time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        if metric_name not in self.metrics:
            return []
        
        return [m for m in self.metrics[metric_name] if m.timestamp >= cutoff_time]
    
    def calculate_metric_statistics(self, metric_name: str, hours: int = 24) -> Dict[str, float]:
        """Calculate statistics for a metric"""
        metrics = self.get_metrics(metric_name, hours)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": self._calculate_percentile(values, 95),
            "p99": self._calculate_percentile(values, 99)
        }
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            weight = index - lower_index
            return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight
    
    async def collect_build_metrics(self, command: str) -> Dict[str, Any]:
        """Collect metrics during build execution"""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        start_cpu = psutil.cpu_percent()
        
        try:
            # Execute build command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root
            )
            
            # Monitor resources during build
            resource_samples = []
            while process.returncode is None:
                resource_samples.append({
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_mb": psutil.virtual_memory().used / 1024 / 1024
                })
                await asyncio.sleep(1)
                
                # Check if process is done
                try:
                    await asyncio.wait_for(process.wait(), timeout=0.1)
                    break
                except asyncio.TimeoutError:
                    continue
            
            stdout, stderr = await process.communicate()
            
            # Calculate metrics
            end_time = time.time()
            build_duration = end_time - start_time
            memory_delta = psutil.virtual_memory().used - start_memory
            
            avg_cpu = statistics.mean([s["cpu_percent"] for s in resource_samples]) if resource_samples else 0
            peak_memory = max([s["memory_mb"] for s in resource_samples]) if resource_samples else 0
            
            metrics = {
                "build_duration": build_duration,
                "exit_code": process.returncode,
                "memory_delta_mb": memory_delta / 1024 / 1024,
                "avg_cpu_percent": avg_cpu,
                "peak_memory_mb": peak_memory,
                "stdout_size": len(stdout),
                "stderr_size": len(stderr),
                "resource_samples": resource_samples
            }
            
            # Store metrics
            timestamp = datetime.now()
            self.add_metric("build_duration", build_duration, timestamp, {"type": "build"})
            self.add_metric("build_memory_usage", peak_memory, timestamp, {"type": "build"})
            self.add_metric("build_cpu_usage", avg_cpu, timestamp, {"type": "build"})
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting build metrics: {e}")
            return {"error": str(e)}
    
    def analyze_performance_trends(self, metric_name: str, days: int = 7) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        metrics = self.get_metrics(metric_name, hours=days * 24)
        
        if len(metrics) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Group by day
        daily_metrics = defaultdict(list)
        for metric in metrics:
            day_key = metric.timestamp.date()
            daily_metrics[day_key].append(metric.value)
        
        # Calculate daily averages
        daily_averages = {}
        for day, values in daily_metrics.items():
            daily_averages[day] = statistics.mean(values)
        
        # Calculate trend
        sorted_days = sorted(daily_averages.keys())
        values = [daily_averages[day] for day in sorted_days]
        
        trend_direction = "stable"
        if len(values) >= 3:
            # Simple trend analysis
            recent_avg = statistics.mean(values[-3:])
            older_avg = statistics.mean(values[:3])
            
            change_percent = ((recent_avg - older_avg) / older_avg) * 100 if older_avg != 0 else 0
            
            if change_percent > 10:
                trend_direction = "increasing"
            elif change_percent < -10:
                trend_direction = "decreasing"
        
        return {
            "metric_name": metric_name,
            "trend_direction": trend_direction,
            "daily_averages": {str(k): v for k, v in daily_averages.items()},
            "change_percent": change_percent if 'change_percent' in locals() else 0,
            "analysis_period_days": days,
            "data_points": len(metrics)
        }
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "monitoring_status": "active" if self.monitoring_active else "inactive",
            "metrics_summary": {},
            "alerts_status": {},
            "performance_trends": {},
            "system_health": {}
        }
        
        # Metrics summary
        for metric_name in self.metrics.keys():
            stats = self.calculate_metric_statistics(metric_name)
            if stats:
                report["metrics_summary"][metric_name] = stats
        
        # Alerts status
        for alert_name, alert in self.alerts.items():
            report["alerts_status"][alert_name] = {
                "is_active": alert.is_active,
                "severity": alert.severity,
                "last_triggered": alert.last_triggered.isoformat() if alert.last_triggered else None
            }
        
        # Performance trends
        key_metrics = ["build_duration", "cpu_usage", "memory_usage", "test_coverage"]
        for metric_name in key_metrics:
            if metric_name in self.metrics:
                trend_analysis = self.analyze_performance_trends(metric_name)
                if "error" not in trend_analysis:
                    report["performance_trends"][metric_name] = trend_analysis
        
        # System health
        report["system_health"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "active_alerts": sum(1 for alert in self.alerts.values() if alert.is_active),
            "metrics_count": sum(len(deque_obj) for deque_obj in self.metrics.values())
        }
        
        return report
    
    def save_monitoring_report(self, filename: str = None):
        """Save monitoring report to file"""
        if filename is None:
            filename = f"sdlc_monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.generate_monitoring_report()
        report_path = self.project_root / filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Monitoring report saved to: {report_path}")
        return report_path
    
    # Alert condition methods
    def _check_regression(self, metrics: List[PerformanceMetric], threshold: float) -> bool:
        """Check for performance regression"""
        if len(metrics) < 10:
            return False
        
        # Compare recent vs historical average
        recent_values = [m.value for m in metrics[-5:]]
        historical_values = [m.value for m in metrics[:-5]]
        
        if not historical_values:
            return False
        
        recent_avg = statistics.mean(recent_values)
        historical_avg = statistics.mean(historical_values)
        
        return recent_avg > historical_avg * threshold
    
    def _check_failure_rate(self, metrics: List[PerformanceMetric], threshold: float) -> bool:
        """Check test failure rate"""
        if not metrics:
            return False
        
        # Assuming failure rate metric is stored
        recent_failure_rate = metrics[-1].value if metrics else 0
        return recent_failure_rate > threshold
    
    def _check_security_issues(self, metrics: List[PerformanceMetric]) -> bool:
        """Check for security issues"""
        if not metrics:
            return False
        
        # Assuming security issues count is stored
        recent_issues = metrics[-1].value if metrics else 0
        return recent_issues > 0
    
    def _check_coverage_drop(self, metrics: List[PerformanceMetric], threshold: float) -> bool:
        """Check for test coverage drop"""
        if len(metrics) < 2:
            return False
        
        current_coverage = metrics[-1].value
        previous_coverage = metrics[-2].value
        
        drop_percentage = ((previous_coverage - current_coverage) / previous_coverage) * 100 if previous_coverage > 0 else 0
        return drop_percentage > threshold
    
    def _check_resource_usage(self, metrics: List[PerformanceMetric], resource_type: str, threshold: float) -> bool:
        """Check resource usage threshold"""
        if not metrics:
            return False
        
        recent_usage = metrics[-1].value if metrics else 0
        return recent_usage > threshold
    
    def _get_relevant_metrics_for_alert(self, alert_name: str) -> List[PerformanceMetric]:
        """Get relevant metrics for alert evaluation"""
        metric_mapping = {
            "build_time_regression": "build_duration",
            "test_failure_rate": "test_failures",
            "security_vulnerabilities": "security_issues",
            "coverage_drop": "test_coverage",
            "memory_usage_high": "memory_usage",
            "cpu_usage_high": "cpu_usage"
        }
        
        metric_name = metric_mapping.get(alert_name, "")
        if metric_name and metric_name in self.metrics:
            return list(self.metrics[metric_name])
        
        return []
    
    def _trigger_alert(self, alert: Alert):
        """Trigger an alert"""
        alert.is_active = True
        alert.last_triggered = datetime.now()
        
        severity_emoji = {"warning": "⚠️", "error": "❌", "critical": "🚨"}
        emoji = severity_emoji.get(alert.severity, "⚠️")
        
        logger.warning(f"{emoji} ALERT TRIGGERED: {alert.name} (Severity: {alert.severity})")
        
        # In a real system, this would send notifications (email, Slack, etc.)
        self._send_notification(alert)
    
    def _resolve_alert(self, alert: Alert):
        """Resolve an alert"""
        alert.is_active = False
        logger.info(f"✅ ALERT RESOLVED: {alert.name}")
    
    def _send_notification(self, alert: Alert):
        """Send alert notification (placeholder)"""
        # In a real implementation, this would integrate with notification systems
        logger.info(f"📤 Notification sent for alert: {alert.name}")
    
    # Utility methods for Git metrics
    def _get_recent_commit_count(self, days: int = 1) -> int:
        """Get recent commit count"""
        try:
            since_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            result = subprocess.run(
                ['git', 'rev-list', '--count', f'--since={since_date}', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            return int(result.stdout.strip()) if result.returncode == 0 else 0
        except (subprocess.SubprocessError, ValueError):
            return 0
    
    def _get_repository_size(self) -> float:
        """Get repository size in MB"""
        try:
            result = subprocess.run(
                ['du', '-sm', '.git'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.returncode == 0:
                return float(result.stdout.split()[0])
        except (subprocess.SubprocessError, ValueError):
            pass
        return 0.0
    
    def _get_branch_count(self) -> int:
        """Get number of branches"""
        try:
            result = subprocess.run(
                ['git', 'branch', '-a'],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            if result.returncode == 0:
                return len([line for line in result.stdout.split('\n') if line.strip()])
        except subprocess.SubprocessError:
            pass
        return 0


async def main():
    """Main execution for monitoring demo"""
    print("🔍 SDLC MONITORING SYSTEM - Generation 2")
    print("=" * 50)
    
    monitor = SDLCMonitor()
    monitor.start_monitoring()
    
    try:
        # Simulate some build monitoring
        print("📊 Collecting build metrics...")
        build_metrics = await monitor.collect_build_metrics("echo 'Build simulation' && sleep 2")
        print(f"Build Duration: {build_metrics.get('build_duration', 0):.1f}s")
        
        # Wait a bit for metrics collection
        await asyncio.sleep(15)
        
        # Generate report
        print("📋 Generating monitoring report...")
        report_path = monitor.save_monitoring_report()
        print(f"Report saved to: {report_path}")
        
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())