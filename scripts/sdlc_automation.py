#!/usr/bin/env python3
"""Autonomous SDLC automation orchestrator."""

import argparse
import json
import logging
import sys
import time
import signal
from pathlib import Path
from typing import Dict, Any, List

from continual_transformer.sdlc.core import SDLCManager, WorkflowTask, TaskPriority
from continual_transformer.sdlc.automation import (
    AutomatedWorkflow, AutomationConfig, TriggerType
)
from continual_transformer.sdlc.monitoring import SDLCMonitor
from continual_transformer.sdlc.reliability import ReliabilityManager
from continual_transformer.sdlc.optimization import (
    OptimizedWorkflowEngine, OptimizationStrategy
)


class SDLCOrchestrator:
    """Main orchestrator for autonomous SDLC operations."""
    
    def __init__(self, project_path: str, config: Dict[str, Any] = None):
        self.project_path = Path(project_path)
        self.config = config or {}
        self.running = False
        
        # Initialize components
        self.sdlc_manager = SDLCManager(str(project_path), self.config.get('sdlc', {}))
        self.monitor = SDLCMonitor(str(project_path))
        self.reliability_manager = ReliabilityManager(str(project_path))
        
        # Automation components
        automation_config = AutomationConfig(
            level=self.config.get('automation_level', 'semi_auto'),
            triggers=self.config.get('triggers', [TriggerType.MANUAL]),
            max_concurrent=self.config.get('max_concurrent', 4),
            retry_failed=self.config.get('retry_failed', True)
        )
        self.automated_workflow = AutomatedWorkflow(str(project_path), automation_config)
        
        # Optimization
        optimization_strategy = OptimizationStrategy(
            self.config.get('optimization_strategy', 'intelligent')
        )
        self.optimized_engine = OptimizedWorkflowEngine(
            max_workers=self.config.get('max_workers', 4),
            optimization_strategy=optimization_strategy
        )
        
        self.logger = self._setup_logging()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_level = self.config.get('log_level', 'INFO')
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.project_path / 'sdlc_automation.log')
            ]
        )
        
        return logging.getLogger(__name__)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.shutdown()
        sys.exit(0)
    
    def start(self):
        """Start the SDLC orchestrator."""
        self.logger.info("Starting SDLC Orchestrator")
        self.running = True
        
        try:
            # Start monitoring
            self.monitor.start_monitoring(
                interval_seconds=self.config.get('monitoring_interval', 60)
            )
            
            # Start automated workflows
            self.automated_workflow.start()
            
            self.logger.info("SDLC Orchestrator started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start SDLC Orchestrator: {e}")
            self.shutdown()
            raise
    
    def shutdown(self):
        """Shutdown the SDLC orchestrator."""
        if not self.running:
            return
        
        self.logger.info("Shutting down SDLC Orchestrator")
        self.running = False
        
        try:
            # Stop components in reverse order
            self.automated_workflow.stop()
            self.monitor.stop_monitoring()
            self.optimized_engine.shutdown()
            
            # Cleanup
            self.monitor.cleanup()
            
            self.logger.info("SDLC Orchestrator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def execute_ci_pipeline(self) -> Dict[str, Any]:
        """Execute CI pipeline."""
        self.logger.info("Executing CI pipeline")
        
        try:
            # Create CI workflow
            ci_workflow_id = self.automated_workflow.create_ci_workflow()
            
            # Execute workflow
            execution_id = self.automated_workflow.execute_workflow(
                ci_workflow_id, TriggerType.MANUAL
            )
            
            # Monitor execution
            start_time = time.time()
            timeout = self.config.get('ci_timeout', 1800)  # 30 minutes
            
            while time.time() - start_time < timeout:
                status = self.automated_workflow.get_workflow_status(ci_workflow_id)
                
                if not status.get('running', True):
                    break
                
                time.sleep(5)  # Check every 5 seconds
            
            # Get final results
            final_status = self.automated_workflow.get_workflow_status(ci_workflow_id)
            
            self.logger.info("CI pipeline execution completed")
            
            return {
                "workflow_id": ci_workflow_id,
                "execution_id": execution_id,
                "status": final_status,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"CI pipeline execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def execute_deployment_pipeline(self) -> Dict[str, Any]:
        """Execute deployment pipeline."""
        self.logger.info("Executing deployment pipeline")
        
        try:
            # Create deployment workflow
            deploy_workflow_id = self.automated_workflow.create_deployment_workflow()
            
            # Execute workflow
            execution_id = self.automated_workflow.execute_workflow(
                deploy_workflow_id, TriggerType.MANUAL
            )
            
            # Monitor execution (similar to CI)
            start_time = time.time()
            timeout = self.config.get('deploy_timeout', 3600)  # 1 hour
            
            while time.time() - start_time < timeout:
                status = self.automated_workflow.get_workflow_status(deploy_workflow_id)
                
                if not status.get('running', True):
                    break
                
                time.sleep(10)  # Check every 10 seconds for deployment
            
            final_status = self.automated_workflow.get_workflow_status(deploy_workflow_id)
            
            self.logger.info("Deployment pipeline execution completed")
            
            return {
                "workflow_id": deploy_workflow_id,
                "execution_id": execution_id,
                "status": final_status,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Deployment pipeline execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def execute_custom_workflow(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute custom workflow from task definitions."""
        self.logger.info(f"Executing custom workflow with {len(tasks)} tasks")
        
        try:
            # Convert task definitions to WorkflowTask objects
            workflow_tasks = []
            for task_def in tasks:
                task = WorkflowTask(
                    id=task_def['id'],
                    name=task_def['name'],
                    command=task_def['command'],
                    description=task_def.get('description', ''),
                    dependencies=task_def.get('dependencies', []),
                    priority=TaskPriority(task_def.get('priority', 2)),  # NORMAL
                    timeout=task_def.get('timeout', 300),
                    environment=task_def.get('environment', {}),
                    working_dir=task_def.get('working_dir')
                )
                workflow_tasks.append(task)
            
            # Execute with optimization
            results = self.optimized_engine.execute_optimized_workflow(workflow_tasks)
            
            self.logger.info("Custom workflow execution completed")
            
            return {
                "results": {
                    task_id: {
                        "status": result.status.value,
                        "duration": result.duration,
                        "exit_code": result.exit_code,
                        "output": result.output[:500],  # Truncate for response
                        "error": result.error[:500] if result.error else None
                    }
                    for task_id, result in results.items()
                },
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Custom workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Get status from all components
            dashboard_data = self.monitor.get_dashboard_data()
            reliability_report = self.reliability_manager.get_reliability_report()
            automation_status = self.automated_workflow.get_workflow_status("ci_workflow")
            performance_metrics = self.optimized_engine.get_performance_metrics()
            
            return {
                "orchestrator": {
                    "running": self.running,
                    "project_path": str(self.project_path)
                },
                "monitoring": dashboard_data,
                "reliability": reliability_report,
                "automation": automation_status,
                "performance": performance_metrics,
                "timestamp": time.time(),
                "status": "healthy" if self.running else "stopped"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def generate_comprehensive_report(self, output_path: str) -> bool:
        """Generate comprehensive SDLC report."""
        try:
            self.logger.info(f"Generating comprehensive report: {output_path}")
            
            # Get all system data
            system_status = self.get_system_status()
            
            # Add additional analysis
            report = {
                "report_type": "comprehensive_sdlc",
                "project_path": str(self.project_path),
                "generated_at": time.time(),
                "system_status": system_status,
                "configuration": self.config,
                "summary": {
                    "orchestrator_healthy": system_status.get("status") == "healthy",
                    "monitoring_active": system_status.get("monitoring", {}).get("system_status", {}).get("monitoring_active", False),
                    "reliability_score": system_status.get("reliability", {}).get("reliability_score", 0),
                    "total_workflows": system_status.get("performance", {}).get("total_workflows", 0)
                }
            }
            
            # Write report
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Comprehensive report generated: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            return False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith(('.yml', '.yaml')):
                import yaml
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path}")
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def main():
    """Main entry point for SDLC automation."""
    
    parser = argparse.ArgumentParser(
        description="Autonomous SDLC Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  ci        - Execute CI pipeline
  deploy    - Execute deployment pipeline
  custom    - Execute custom workflow from JSON file
  monitor   - Start monitoring mode (runs until interrupted)
  status    - Get current system status
  report    - Generate comprehensive report

Examples:
  python sdlc_automation.py /path/to/project ci
  python sdlc_automation.py . deploy --config config.json
  python sdlc_automation.py . custom --tasks tasks.json
  python sdlc_automation.py . monitor --config automation_config.yml
        """)
    
    parser.add_argument(
        "project_path",
        help="Path to the project directory"
    )
    
    parser.add_argument(
        "command",
        choices=["ci", "deploy", "custom", "monitor", "status", "report"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Configuration file (JSON or YAML)"
    )
    
    parser.add_argument(
        "--tasks", "-t",
        help="Tasks definition file (JSON) for custom command"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file for reports or status"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout for operations in seconds (default: 3600)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Validate project path
    project_path = Path(args.project_path).resolve()
    if not project_path.exists():
        print(f"Error: Project path does not exist: {project_path}")
        sys.exit(1)
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override config with command line args
    config['log_level'] = args.log_level
    if args.timeout:
        config['ci_timeout'] = args.timeout
        config['deploy_timeout'] = args.timeout
    
    # Initialize orchestrator
    try:
        orchestrator = SDLCOrchestrator(str(project_path), config)
        
        # Execute command
        if args.command == "monitor":
            orchestrator.start()
            try:
                print("SDLC monitoring started. Press Ctrl+C to stop.")
                while orchestrator.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
                orchestrator.shutdown()
            
        elif args.command == "status":
            status = orchestrator.get_system_status()
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(status, f, indent=2, default=str)
                print(f"Status saved to {args.output}")
            else:
                print(json.dumps(status, indent=2, default=str))
        
        elif args.command == "report":
            if not args.output:
                args.output = "sdlc_comprehensive_report.json"
            
            success = orchestrator.generate_comprehensive_report(args.output)
            if success:
                print(f"Comprehensive report generated: {args.output}")
            else:
                print("Failed to generate report")
                sys.exit(1)
        
        elif args.command == "ci":
            orchestrator.start()
            try:
                result = orchestrator.execute_ci_pipeline()
                
                if result["success"]:
                    print("CI pipeline completed successfully")
                    if args.output:
                        with open(args.output, 'w') as f:
                            json.dump(result, f, indent=2, default=str)
                else:
                    print(f"CI pipeline failed: {result.get('error', 'Unknown error')}")
                    sys.exit(1)
                    
            finally:
                orchestrator.shutdown()
        
        elif args.command == "deploy":
            orchestrator.start()
            try:
                result = orchestrator.execute_deployment_pipeline()
                
                if result["success"]:
                    print("Deployment pipeline completed successfully")
                    if args.output:
                        with open(args.output, 'w') as f:
                            json.dump(result, f, indent=2, default=str)
                else:
                    print(f"Deployment pipeline failed: {result.get('error', 'Unknown error')}")
                    sys.exit(1)
                    
            finally:
                orchestrator.shutdown()
        
        elif args.command == "custom":
            if not args.tasks:
                print("Error: --tasks file required for custom command")
                sys.exit(1)
            
            # Load tasks definition
            try:
                with open(args.tasks, 'r') as f:
                    tasks_def = json.load(f)
                
                if not isinstance(tasks_def, list):
                    tasks_def = tasks_def.get('tasks', [])
                
            except Exception as e:
                print(f"Error loading tasks file: {e}")
                sys.exit(1)
            
            orchestrator.start()
            try:
                result = orchestrator.execute_custom_workflow(tasks_def)
                
                if result["success"]:
                    print("Custom workflow completed successfully")
                    if args.output:
                        with open(args.output, 'w') as f:
                            json.dump(result, f, indent=2, default=str)
                else:
                    print(f"Custom workflow failed: {result.get('error', 'Unknown error')}")
                    sys.exit(1)
                    
            finally:
                orchestrator.shutdown()
        
    except Exception as e:
        print(f"SDLC Orchestrator error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()