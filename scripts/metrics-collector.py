#!/usr/bin/env python3
"""
Comprehensive metrics collection system for continual-tiny-transformer.
Collects metrics from various sources and updates project metrics.
"""

import json
import os
import sys
import asyncio
import aiohttp
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import argparse


class MetricsCollector:
    """Comprehensive metrics collection system."""
    
    def __init__(self, config_file: str = ".github/project-metrics.json"):
        self.config_file = Path(config_file)
        self.metrics_data = self.load_metrics()
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.repo_owner = os.getenv('GITHUB_REPOSITORY_OWNER', 'your-org')
        self.repo_name = os.getenv('GITHUB_REPOSITORY_NAME', 'continual-tiny-transformer')
    
    def load_metrics(self) -> Dict[str, Any]:
        """Load existing metrics data."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_metrics(self):
        """Save updated metrics data."""
        self.metrics_data['project']['last_updated'] = datetime.utcnow().isoformat() + 'Z'
        with open(self.config_file, 'w') as f:
            json.dump(self.metrics_data, f, indent=2)
    
    async def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect metrics from GitHub API."""
        if not self.github_token:
            print("GITHUB_TOKEN not available, skipping GitHub metrics")
            return {}
        
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        metrics = {}
        
        async with aiohttp.ClientSession(headers=headers) as session:
            # Repository stats
            repo_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
            
            try:
                async with session.get(repo_url) as response:
                    repo_data = await response.json()
                    metrics['repository'] = {
                        'stars': repo_data.get('stargazers_count', 0),
                        'forks': repo_data.get('forks_count', 0),
                        'open_issues': repo_data.get('open_issues_count', 0),
                        'size_kb': repo_data.get('size', 0)
                    }
            except Exception as e:
                print(f"Error collecting repository metrics: {e}")
            
            # Pull requests
            prs_url = f"{repo_url}/pulls?state=all&per_page=100"
            try:
                async with session.get(prs_url) as response:
                    prs_data = await response.json()
                    
                    now = datetime.utcnow()
                    last_week = now - timedelta(days=7)
                    
                    recent_prs = [
                        pr for pr in prs_data 
                        if datetime.fromisoformat(pr['created_at'].replace('Z', '+00:00')) > last_week
                    ]
                    
                    merged_prs = [pr for pr in recent_prs if pr.get('merged_at')]
                    
                    metrics['pull_requests'] = {
                        'total_last_week': len(recent_prs),
                        'merged_last_week': len(merged_prs),
                        'merge_rate': len(merged_prs) / len(recent_prs) if recent_prs else 0
                    }
            except Exception as e:
                print(f"Error collecting PR metrics: {e}")
            
            # Issues
            issues_url = f"{repo_url}/issues?state=all&per_page=100"
            try:
                async with session.get(issues_url) as response:
                    issues_data = await response.json()
                    
                    # Filter out PRs (GitHub includes PRs in issues)
                    actual_issues = [issue for issue in issues_data if not issue.get('pull_request')]
                    
                    open_issues = [issue for issue in actual_issues if issue['state'] == 'open']
                    closed_issues = [issue for issue in actual_issues if issue['state'] == 'closed']
                    
                    # Security issues
                    security_issues = [
                        issue for issue in open_issues 
                        if any(label['name'].lower() in ['security', 'vulnerability'] 
                              for label in issue.get('labels', []))
                    ]
                    
                    metrics['issues'] = {
                        'total_open': len(open_issues),
                        'total_closed': len(closed_issues),
                        'security_open': len(security_issues),
                        'resolution_rate': len(closed_issues) / (len(open_issues) + len(closed_issues)) if actual_issues else 0
                    }
            except Exception as e:
                print(f"Error collecting issues metrics: {e}")
            
            # Workflow runs (CI/CD metrics)
            workflows_url = f"{repo_url}/actions/runs?per_page=50"
            try:
                async with session.get(workflows_url) as response:
                    workflows_data = await response.json()
                    
                    runs = workflows_data.get('workflow_runs', [])
                    
                    last_week_runs = [
                        run for run in runs
                        if datetime.fromisoformat(run['created_at'].replace('Z', '+00:00')) > last_week
                    ]
                    
                    successful_runs = [run for run in last_week_runs if run['conclusion'] == 'success']
                    failed_runs = [run for run in last_week_runs if run['conclusion'] == 'failure']
                    
                    metrics['ci_cd'] = {
                        'total_runs_last_week': len(last_week_runs),
                        'successful_runs': len(successful_runs),
                        'failed_runs': len(failed_runs),
                        'success_rate': len(successful_runs) / len(last_week_runs) if last_week_runs else 0
                    }
            except Exception as e:
                print(f"Error collecting CI/CD metrics: {e}")
        
        return metrics
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics from local analysis."""
        metrics = {}
        
        try:
            # Test coverage
            result = subprocess.run([
                'pytest', '--cov=continual_transformer', '--cov-report=json'
            ], capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0 and os.path.exists('coverage.json'):
                with open('coverage.json', 'r') as f:
                    coverage_data = json.load(f)
                    metrics['test_coverage'] = {
                        'percentage': coverage_data.get('totals', {}).get('percent_covered', 0),
                        'lines_covered': coverage_data.get('totals', {}).get('covered_lines', 0),
                        'lines_total': coverage_data.get('totals', {}).get('num_statements', 0)
                    }
        except Exception as e:
            print(f"Error collecting coverage metrics: {e}")
        
        try:
            # Code complexity with radon
            result = subprocess.run([
                'radon', 'cc', 'src/', '--json'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                total_complexity = 0
                total_functions = 0
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item['type'] == 'function':
                            total_complexity += item['complexity']
                            total_functions += 1
                
                avg_complexity = total_complexity / total_functions if total_functions > 0 else 0
                
                metrics['code_complexity'] = {
                    'average_cyclomatic_complexity': avg_complexity,
                    'total_functions': total_functions,
                    'high_complexity_functions': sum(
                        1 for file_data in complexity_data.values()
                        for item in file_data
                        if item['type'] == 'function' and item['complexity'] > 10
                    )
                }
        except Exception as e:
            print(f"Error collecting complexity metrics: {e}")
        
        try:
            # Linting violations
            result = subprocess.run([
                'ruff', 'check', 'src/', '--format=json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                violations = json.loads(result.stdout)
                
                severity_counts = {'error': 0, 'warning': 0, 'info': 0}
                for violation in violations:
                    severity = violation.get('type', 'info').lower()
                    if severity in severity_counts:
                        severity_counts[severity] += 1
                
                metrics['linting'] = {
                    'total_violations': len(violations),
                    'errors': severity_counts['error'],
                    'warnings': severity_counts['warning'],
                    'info': severity_counts['info']
                }
        except Exception as e:
            print(f"Error collecting linting metrics: {e}")
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics from security scans."""
        metrics = {}
        
        try:
            # Bandit security scan
            result = subprocess.run([
                'bandit', '-r', 'src/', '-f', 'json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                
                results = bandit_data.get('results', [])
                severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
                
                for issue in results:
                    severity = issue.get('issue_severity', 'LOW')
                    if severity in severity_counts:
                        severity_counts[severity] += 1
                
                metrics['security_scan'] = {
                    'total_issues': len(results),
                    'high_severity': severity_counts['HIGH'],
                    'medium_severity': severity_counts['MEDIUM'],
                    'low_severity': severity_counts['LOW']
                }
        except Exception as e:
            print(f"Error collecting security metrics: {e}")
        
        try:
            # Safety dependency scan
            result = subprocess.run([
                'safety', 'check', '--json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                vulnerabilities = safety_data.get('vulnerabilities', [])
                
                metrics['dependency_security'] = {
                    'vulnerable_packages': len(vulnerabilities),
                    'total_vulnerabilities': sum(len(v.get('vulnerabilities', [])) for v in vulnerabilities)
                }
        except Exception as e:
            print(f"Error collecting dependency security metrics: {e}")
        
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        metrics = {}
        
        try:
            # Build time measurement
            start_time = datetime.now()
            result = subprocess.run([
                'python', '-m', 'build'
            ], capture_output=True, text=True)
            build_time = (datetime.now() - start_time).total_seconds()
            
            metrics['build_performance'] = {
                'build_time_seconds': build_time,
                'build_success': result.returncode == 0
            }
        except Exception as e:
            print(f"Error collecting build metrics: {e}")
        
        try:
            # Test execution time
            start_time = datetime.now()
            result = subprocess.run([
                'pytest', 'tests/', '--durations=0', '--quiet'
            ], capture_output=True, text=True)
            test_time = (datetime.now() - start_time).total_seconds()
            
            metrics['test_performance'] = {
                'total_test_time_seconds': test_time,
                'tests_success': result.returncode == 0
            }
        except Exception as e:
            print(f"Error collecting test performance metrics: {e}")
        
        return metrics
    
    async def collect_deployment_metrics(self) -> Dict[str, Any]:
        """Collect deployment and operational metrics."""
        metrics = {}
        
        # These would typically come from monitoring systems
        # For now, we'll use placeholder values that could be populated
        # from Prometheus, Grafana, or deployment logs
        
        metrics['deployment'] = {
            'last_deployment_time': datetime.utcnow().isoformat() + 'Z',
            'deployment_success_rate': 0.95,  # Would come from deployment logs
            'avg_deployment_duration_minutes': 8,
            'rollback_count_last_month': 2
        }
        
        metrics['reliability'] = {
            'uptime_percentage': 99.95,  # Would come from monitoring
            'error_rate': 0.001,
            'response_time_p95_ms': 250,
            'mttr_minutes': 15
        }
        
        return metrics
    
    def update_metrics_in_config(self, collected_metrics: Dict[str, Any]):
        """Update the metrics configuration with collected data."""
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        # Update development metrics
        if 'test_coverage' in collected_metrics:
            self.metrics_data['metrics']['development']['code_quality']['test_coverage'].update({
                'current': collected_metrics['test_coverage']['percentage'],
                'last_measured': timestamp
            })
        
        if 'code_complexity' in collected_metrics:
            self.metrics_data['metrics']['development']['code_quality']['code_complexity'].update({
                'cyclomatic_complexity_avg': collected_metrics['code_complexity']['average_cyclomatic_complexity']
            })
        
        if 'linting' in collected_metrics:
            self.metrics_data['metrics']['development']['code_quality']['linting_violations'].update({
                'total': collected_metrics['linting']['total_violations'],
                'critical': collected_metrics['linting']['errors'],
                'major': collected_metrics['linting']['warnings'],
                'minor': collected_metrics['linting']['info']
            })
        
        # Update security metrics
        if 'security_scan' in collected_metrics:
            self.metrics_data['metrics']['development']['security']['vulnerabilities'].update({
                'critical': collected_metrics['security_scan']['high_severity'],
                'high': collected_metrics['security_scan']['medium_severity'],
                'medium': collected_metrics['security_scan']['low_severity'],
                'total': collected_metrics['security_scan']['total_issues']
            })
        
        # Update performance metrics
        if 'build_performance' in collected_metrics:
            self.metrics_data['metrics']['development']['performance']['build_time'].update({
                'avg_seconds': collected_metrics['build_performance']['build_time_seconds']
            })
        
        # Update CI/CD metrics from GitHub
        if 'ci_cd' in collected_metrics:
            self.metrics_data['metrics']['operational']['deployment']['deployment_success_rate'].update({
                'percentage': collected_metrics['ci_cd']['success_rate'] * 100
            })
        
        # Update automation collection timestamp
        self.metrics_data['metrics']['automation']['metrics_collection']['last_collection'] = timestamp
    
    def generate_metrics_report(self) -> str:
        """Generate a human-readable metrics report."""
        metrics = self.metrics_data['metrics']
        
        report = f"""
# Metrics Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Code Quality
- Test Coverage: {metrics['development']['code_quality']['test_coverage']['current']}% (Target: {metrics['development']['code_quality']['test_coverage']['target']}%)
- Linting Violations: {metrics['development']['code_quality']['linting_violations']['total']} (Target: {metrics['development']['code_quality']['linting_violations']['target']})
- Cyclomatic Complexity: {metrics['development']['code_quality']['code_complexity']['cyclomatic_complexity_avg']:.1f}

## Security
- Critical Vulnerabilities: {metrics['development']['security']['vulnerabilities']['critical']}
- Total Vulnerabilities: {metrics['development']['security']['vulnerabilities']['total']}
- Outdated Dependencies: {metrics['development']['security']['dependency_audit']['outdated']}

## Performance
- Build Time: {metrics['development']['performance']['build_time']['avg_seconds']}s (Target: {metrics['development']['performance']['build_time']['target_seconds']}s)
- Test Execution: {metrics['development']['performance']['test_execution_time']['total_seconds']}s

## Operational
- Deployment Success Rate: {metrics['operational']['deployment']['deployment_success_rate']['percentage']}%
- Uptime: {metrics['operational']['reliability']['uptime']['percentage']}%
- Response Time (P95): {metrics['operational']['reliability']['response_time']['p95_ms']}ms

## Trends
- Development velocity appears to be {'improving' if metrics['team_productivity']['development_velocity']['story_points_per_sprint'] > 40 else 'declining'}
- Security posture is {'good' if metrics['development']['security']['vulnerabilities']['critical'] == 0 else 'needs attention'}
- Performance is {'meeting targets' if metrics['development']['performance']['build_time']['avg_seconds'] <= metrics['development']['performance']['build_time']['target_seconds'] else 'below targets'}
"""
        return report
    
    async def run_collection(self):
        """Run the complete metrics collection process."""
        print("Starting metrics collection...")
        
        collected_metrics = {}
        
        # Collect from various sources
        print("Collecting GitHub metrics...")
        github_metrics = await self.collect_github_metrics()
        collected_metrics.update(github_metrics)
        
        print("Collecting code quality metrics...")
        quality_metrics = self.collect_code_quality_metrics()
        collected_metrics.update(quality_metrics)
        
        print("Collecting security metrics...")
        security_metrics = self.collect_security_metrics()
        collected_metrics.update(security_metrics)
        
        print("Collecting performance metrics...")
        performance_metrics = self.collect_performance_metrics()
        collected_metrics.update(performance_metrics)
        
        print("Collecting deployment metrics...")
        deployment_metrics = await self.collect_deployment_metrics()
        collected_metrics.update(deployment_metrics)
        
        # Update configuration
        print("Updating metrics configuration...")
        self.update_metrics_in_config(collected_metrics)
        
        # Save updated metrics
        self.save_metrics()
        
        # Generate report
        report = self.generate_metrics_report()
        
        # Save report
        with open('metrics-report.md', 'w') as f:
            f.write(report)
        
        print("Metrics collection completed!")
        print(f"Updated metrics saved to {self.config_file}")
        print("Report saved to metrics-report.md")
        
        return collected_metrics


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Collect project metrics')
    parser.add_argument('--config', default='.github/project-metrics.json', 
                       help='Metrics configuration file')
    parser.add_argument('--output', default='metrics-report.md',
                       help='Output report file')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    collector = MetricsCollector(args.config)
    await collector.run_collection()


if __name__ == "__main__":
    asyncio.run(main())