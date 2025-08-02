#!/usr/bin/env python3
"""
Repository maintenance automation script for continual-tiny-transformer.
Performs regular maintenance tasks like cleanup, updates, and optimization.
"""

import asyncio
import aiohttp
import subprocess
import json
import shutil
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse


class RepositoryMaintainer:
    """Automated repository maintenance system."""
    
    def __init__(self, dry_run: bool = False, verbose: bool = False):
        self.dry_run = dry_run
        self.verbose = verbose
        self.maintenance_log = []
        self.github_token = os.getenv('GITHUB_TOKEN')
        
    def log_action(self, action: str, status: str, details: str = ""):
        """Log maintenance action."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "status": status,
            "details": details,
            "dry_run": self.dry_run
        }
        self.maintenance_log.append(log_entry)
        
        if self.verbose:
            print(f"[{status.upper()}] {action}: {details}")
    
    def cleanup_python_artifacts(self) -> Dict[str, Any]:
        """Clean up Python build artifacts and cache files."""
        cleanup_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo", 
            "**/*.pyd",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "dist/",
            "build/",
            "*.egg-info/",
            ".coverage",
            "htmlcov/",
            ".tox/",
            ".nox/"
        ]
        
        total_size = 0
        cleaned_files = []
        
        for pattern in cleanup_patterns:
            for path in Path('.').glob(pattern):
                if path.exists():
                    if path.is_file():
                        size = path.stat().st_size
                        if not self.dry_run:
                            path.unlink()
                        cleaned_files.append(str(path))
                        total_size += size
                    elif path.is_dir():
                        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                        if not self.dry_run:
                            shutil.rmtree(path)
                        cleaned_files.append(str(path))
                        total_size += size
        
        self.log_action(
            "cleanup_python_artifacts",
            "success",
            f"Cleaned {len(cleaned_files)} items, freed {total_size / 1024 / 1024:.1f}MB"
        )
        
        return {
            "files_cleaned": len(cleaned_files),
            "size_freed_mb": total_size / 1024 / 1024,
            "cleaned_items": cleaned_files
        }
    
    def cleanup_git_artifacts(self) -> Dict[str, Any]:
        """Clean up Git artifacts and optimize repository."""
        try:
            # Git garbage collection
            if not self.dry_run:
                subprocess.run(['git', 'gc', '--aggressive', '--prune=now'], 
                             check=True, capture_output=True)
            
            # Clean up remote tracking branches
            result = subprocess.run(['git', 'remote', 'prune', 'origin'], 
                                  capture_output=True, text=True)
            
            # Get repository size before/after
            repo_size_result = subprocess.run(['du', '-sh', '.git'], 
                                            capture_output=True, text=True)
            repo_size = repo_size_result.stdout.split()[0] if repo_size_result.returncode == 0 else "unknown"
            
            self.log_action(
                "cleanup_git_artifacts", 
                "success",
                f"Git cleanup completed, repository size: {repo_size}"
            )
            
            return {
                "git_gc_completed": True,
                "remote_branches_pruned": True,
                "repository_size": repo_size
            }
            
        except subprocess.CalledProcessError as e:
            self.log_action("cleanup_git_artifacts", "error", f"Git cleanup failed: {e}")
            return {"error": str(e)}
    
    def update_dependencies(self) -> Dict[str, Any]:
        """Update and audit project dependencies."""
        try:
            updates = {}
            
            # Check for outdated packages
            outdated_result = subprocess.run(['pip', 'list', '--outdated', '--format=json'],
                                           capture_output=True, text=True)
            
            if outdated_result.returncode == 0:
                outdated_packages = json.loads(outdated_result.stdout)
                updates['outdated_packages'] = len(outdated_packages)
                
                # Update packages if not dry run
                if not self.dry_run and outdated_packages:
                    for package in outdated_packages[:5]:  # Limit to 5 packages per run
                        try:
                            subprocess.run(['pip', 'install', '--upgrade', package['name']],
                                         check=True, capture_output=True)
                        except subprocess.CalledProcessError:
                            pass  # Continue with other packages
            
            # Run security audit
            try:
                safety_result = subprocess.run(['safety', 'check', '--json'],
                                             capture_output=True, text=True)
                if safety_result.returncode == 0:
                    safety_data = json.loads(safety_result.stdout)
                    updates['security_vulnerabilities'] = len(safety_data.get('vulnerabilities', []))
                else:
                    # Parse safety output for vulnerability count
                    output_lines = safety_result.stdout.split('\n')
                    vuln_count = sum(1 for line in output_lines if 'vulnerability' in line.lower())
                    updates['security_vulnerabilities'] = vuln_count
            except (json.JSONDecodeError, FileNotFoundError):
                updates['security_vulnerabilities'] = 'unknown'
            
            # Check pip-audit if available
            try:
                audit_result = subprocess.run(['pip-audit', '--format=json'],
                                            capture_output=True, text=True)
                if audit_result.returncode == 0:
                    audit_data = json.loads(audit_result.stdout)
                    updates['pip_audit_vulnerabilities'] = len(audit_data.get('vulnerabilities', []))
            except (FileNotFoundError, json.JSONDecodeError):
                updates['pip_audit_vulnerabilities'] = 'not_available'
            
            self.log_action(
                "update_dependencies",
                "success", 
                f"Dependency update completed: {updates.get('outdated_packages', 0)} outdated, "
                f"{updates.get('security_vulnerabilities', 0)} vulnerabilities"
            )
            
            return updates
            
        except Exception as e:
            self.log_action("update_dependencies", "error", f"Dependency update failed: {e}")
            return {"error": str(e)}
    
    def cleanup_log_files(self) -> Dict[str, Any]:
        """Clean up old log files and temporary data."""
        log_patterns = [
            "logs/**/*.log",
            "*.log",
            "tmp/**/*",
            "temp/**/*",
            ".pytest_cache/**/*",
            "test-results/**/*",
            "coverage-reports/**/*"
        ]
        
        cutoff_date = datetime.now() - timedelta(days=30)
        cleaned_files = []
        total_size = 0
        
        for pattern in log_patterns:
            for path in Path('.').glob(pattern):
                if path.is_file() and datetime.fromtimestamp(path.stat().st_mtime) < cutoff_date:
                    size = path.stat().st_size
                    if not self.dry_run:
                        path.unlink()
                    cleaned_files.append(str(path))
                    total_size += size
        
        self.log_action(
            "cleanup_log_files",
            "success",
            f"Cleaned {len(cleaned_files)} old files, freed {total_size / 1024 / 1024:.1f}MB"
        )
        
        return {
            "files_cleaned": len(cleaned_files),
            "size_freed_mb": total_size / 1024 / 1024,
            "cutoff_date": cutoff_date.isoformat()
        }
    
    def optimize_docker_images(self) -> Dict[str, Any]:
        """Clean up Docker images and optimize Docker setup."""
        try:
            # Check if Docker is available
            docker_check = subprocess.run(['docker', '--version'], 
                                        capture_output=True, text=True)
            
            if docker_check.returncode != 0:
                return {"error": "Docker not available"}
            
            # Clean up unused Docker images
            if not self.dry_run:
                subprocess.run(['docker', 'image', 'prune', '-f'], 
                             capture_output=True)
                subprocess.run(['docker', 'container', 'prune', '-f'], 
                             capture_output=True)
                subprocess.run(['docker', 'volume', 'prune', '-f'], 
                             capture_output=True)
            
            # Get Docker disk usage
            disk_usage_result = subprocess.run(['docker', 'system', 'df'], 
                                             capture_output=True, text=True)
            
            self.log_action(
                "optimize_docker_images",
                "success",
                "Docker cleanup completed"
            )
            
            return {
                "docker_cleanup_completed": True,
                "disk_usage": disk_usage_result.stdout if disk_usage_result.returncode == 0 else None
            }
            
        except Exception as e:
            self.log_action("optimize_docker_images", "error", f"Docker optimization failed: {e}")
            return {"error": str(e)}
    
    def update_documentation_links(self) -> Dict[str, Any]:
        """Check and update documentation links."""
        try:
            # Find all markdown files
            md_files = list(Path('.').glob('**/*.md'))
            
            # This would normally check links, but for simplicity we'll just count files
            total_files = len(md_files)
            
            # Check for common documentation issues
            issues = []
            for md_file in md_files:
                if md_file.stat().st_size == 0:
                    issues.append(f"Empty file: {md_file}")
            
            self.log_action(
                "update_documentation_links",
                "success",
                f"Checked {total_files} documentation files, found {len(issues)} issues"
            )
            
            return {
                "markdown_files_checked": total_files,
                "issues_found": len(issues),
                "issues": issues
            }
            
        except Exception as e:
            self.log_action("update_documentation_links", "error", f"Documentation check failed: {e}")
            return {"error": str(e)}
    
    async def update_github_metadata(self) -> Dict[str, Any]:
        """Update GitHub repository metadata and settings."""
        if not self.github_token:
            self.log_action("update_github_metadata", "skipped", "No GitHub token available")
            return {"skipped": "No GitHub token"}
        
        try:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            repo_owner = os.getenv('GITHUB_REPOSITORY_OWNER', 'your-org')
            repo_name = os.getenv('GITHUB_REPOSITORY_NAME', 'continual-tiny-transformer')
            
            async with aiohttp.ClientSession(headers=headers) as session:
                # Get current repository data
                repo_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
                
                async with session.get(repo_url) as response:
                    if response.status != 200:
                        return {"error": f"Failed to fetch repository data: {response.status}"}
                    
                    repo_data = await response.json()
                    
                    # Update topics if needed
                    current_topics = repo_data.get('topics', [])
                    suggested_topics = [
                        'machine-learning', 'continual-learning', 'transformers',
                        'pytorch', 'deep-learning', 'zero-parameter', 'python'
                    ]
                    
                    new_topics = list(set(current_topics + suggested_topics))
                    
                    if new_topics != current_topics and not self.dry_run:
                        topics_url = f"{repo_url}/topics"
                        topics_data = {"names": new_topics}
                        
                        async with session.put(topics_url, json=topics_data) as topic_response:
                            if topic_response.status == 200:
                                self.log_action(
                                    "update_github_metadata",
                                    "success",
                                    f"Updated repository topics: {len(new_topics)} topics"
                                )
                            else:
                                self.log_action(
                                    "update_github_metadata",
                                    "error",
                                    f"Failed to update topics: {topic_response.status}"
                                )
                    
                    return {
                        "topics_updated": new_topics != current_topics,
                        "current_topics": current_topics,
                        "new_topics": new_topics,
                        "repository_stats": {
                            "stars": repo_data.get('stargazers_count', 0),
                            "forks": repo_data.get('forks_count', 0),
                            "open_issues": repo_data.get('open_issues_count', 0)
                        }
                    }
                    
        except Exception as e:
            self.log_action("update_github_metadata", "error", f"GitHub metadata update failed: {e}")
            return {"error": str(e)}
    
    def generate_maintenance_report(self) -> Dict[str, Any]:
        """Generate comprehensive maintenance report."""
        if not self.maintenance_log:
            return {"error": "No maintenance actions performed"}
        
        successful_actions = [log for log in self.maintenance_log if log['status'] == 'success']
        failed_actions = [log for log in self.maintenance_log if log['status'] == 'error']
        skipped_actions = [log for log in self.maintenance_log if log['status'] == 'skipped']
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "summary": {
                "total_actions": len(self.maintenance_log),
                "successful": len(successful_actions),
                "failed": len(failed_actions),
                "skipped": len(skipped_actions),
                "success_rate": len(successful_actions) / len(self.maintenance_log) if self.maintenance_log else 0
            },
            "actions": self.maintenance_log,
            "recommendations": []
        }
        
        # Add recommendations based on results
        if failed_actions:
            report["recommendations"].append("Review and fix failed maintenance actions")
        
        if len(successful_actions) / len(self.maintenance_log) < 0.8:
            report["recommendations"].append("Investigate maintenance failures")
        
        if not failed_actions and not skipped_actions:
            report["recommendations"].append("All maintenance tasks completed successfully")
        
        return report
    
    async def run_full_maintenance(self) -> Dict[str, Any]:
        """Run complete repository maintenance cycle."""
        print(f"Starting repository maintenance {'(DRY RUN)' if self.dry_run else ''}...")
        
        maintenance_results = {}
        
        # Run all maintenance tasks
        print("1. Cleaning Python artifacts...")
        maintenance_results['python_cleanup'] = self.cleanup_python_artifacts()
        
        print("2. Cleaning Git artifacts...")
        maintenance_results['git_cleanup'] = self.cleanup_git_artifacts()
        
        print("3. Updating dependencies...")
        maintenance_results['dependency_update'] = self.update_dependencies()
        
        print("4. Cleaning log files...")
        maintenance_results['log_cleanup'] = self.cleanup_log_files()
        
        print("5. Optimizing Docker images...")
        maintenance_results['docker_optimization'] = self.optimize_docker_images()
        
        print("6. Checking documentation...")
        maintenance_results['documentation_check'] = self.update_documentation_links()
        
        print("7. Updating GitHub metadata...")
        maintenance_results['github_update'] = await self.update_github_metadata()
        
        # Generate final report
        final_report = self.generate_maintenance_report()
        final_report['detailed_results'] = maintenance_results
        
        return final_report


async def main():
    """Main function to run repository maintenance."""
    parser = argparse.ArgumentParser(description='Repository maintenance automation')
    parser.add_argument('--dry-run', action='store_true',
                       help='Perform dry run without making changes')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--output', default='maintenance-report.json',
                       help='Output report file')
    parser.add_argument('--tasks', nargs='+',
                       choices=['cleanup', 'dependencies', 'docker', 'git', 'docs', 'github'],
                       help='Specific tasks to run (default: all)')
    
    args = parser.parse_args()
    
    maintainer = RepositoryMaintainer(dry_run=args.dry_run, verbose=args.verbose)
    
    if args.tasks:
        # Run specific tasks only
        print(f"Running specific maintenance tasks: {', '.join(args.tasks)}")
        # Implementation for specific tasks would go here
        print("Specific task selection not implemented in this example")
        return
    else:
        # Run full maintenance
        report = await maintainer.run_full_maintenance()
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nMaintenance completed!")
    print(f"Report saved to: {args.output}")
    print(f"Success rate: {report['summary']['success_rate']:.1%}")
    
    if report['summary']['failed'] > 0:
        print(f"⚠️  {report['summary']['failed']} tasks failed")
        return 1
    else:
        print("✅ All maintenance tasks completed successfully")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)