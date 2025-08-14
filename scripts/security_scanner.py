#!/usr/bin/env python3
"""Autonomous security scanning script for SDLC integration."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Import SDLC security components
from continual_transformer.sdlc.security import (
    SecurityValidator, SecurityLevel, VulnerabilityLevel
)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('security_scan.log')
        ]
    )
    return logging.getLogger(__name__)


def run_security_scan(
    project_path: str,
    security_level: SecurityLevel,
    output_format: str = "json",
    output_file: str = None,
    fail_on_critical: bool = True,
    fail_on_high: bool = False
) -> Dict[str, Any]:
    """Run comprehensive security scan."""
    
    logger = setup_logging()
    logger.info(f"Starting security scan on {project_path}")
    logger.info(f"Security level: {security_level.value}")
    
    try:
        # Initialize security validator
        validator = SecurityValidator(project_path, security_level)
        
        # Run comprehensive scan
        start_time = time.time()
        scan_result = validator.run_comprehensive_scan()
        scan_duration = time.time() - start_time
        
        logger.info(f"Security scan completed in {scan_duration:.2f} seconds")
        
        # Log scan summary
        counts = scan_result.vulnerability_count_by_severity
        logger.info(
            f"Vulnerabilities found - Critical: {counts['critical']}, "
            f"High: {counts['high']}, Medium: {counts['medium']}, Low: {counts['low']}"
        )
        logger.info(f"Risk score: {scan_result.risk_score:.1f}/100")
        
        # Validate against security policy
        is_valid, validation_message = validator.validate_scan_result(scan_result)
        logger.info(f"Security validation: {'PASSED' if is_valid else 'FAILED'}")
        if not is_valid:
            logger.warning(f"Validation failure reason: {validation_message}")
        
        # Prepare results
        results = {
            "scan_summary": {
                "scan_id": scan_result.scan_id,
                "duration": scan_duration,
                "files_scanned": scan_result.files_scanned,
                "security_level": security_level.value,
                "risk_score": scan_result.risk_score,
                "validation_passed": is_valid,
                "validation_message": validation_message
            },
            "vulnerability_counts": counts,
            "vulnerabilities": [
                {
                    "id": vuln.id,
                    "title": vuln.title,
                    "description": vuln.description,
                    "severity": vuln.severity.value,
                    "file_path": vuln.file_path,
                    "line_number": vuln.line_number,
                    "rule_id": vuln.rule_id,
                    "confidence": vuln.confidence,
                    "remediation": vuln.remediation
                }
                for vuln in scan_result.vulnerabilities
            ],
            "recommendations": validator._generate_recommendations(
                {"success_rate": 100 - scan_result.risk_score, **counts}
            ),
            "scan_config": scan_result.scan_config,
            "timestamp": time.time()
        }
        
        # Save results if output file specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Results saved to {output_path}")
            else:
                logger.error(f"Unsupported output format: {output_format}")
        
        # Determine exit code based on findings
        exit_code = 0
        
        if fail_on_critical and counts['critical'] > 0:
            logger.error(f"Exiting with error due to {counts['critical']} critical vulnerabilities")
            exit_code = 1
        elif fail_on_high and counts['high'] > 0:
            logger.error(f"Exiting with error due to {counts['high']} high severity vulnerabilities")
            exit_code = 1
        elif not is_valid:
            logger.error("Exiting with error due to security policy validation failure")
            exit_code = 1
        
        return {
            "results": results,
            "exit_code": exit_code,
            "validator": validator
        }
        
    except Exception as e:
        logger.error(f"Security scan failed: {str(e)}")
        return {
            "results": {"error": str(e)},
            "exit_code": 2,
            "validator": None
        }


def generate_security_report(
    scan_results: Dict[str, Any],
    validator: SecurityValidator,
    report_path: str
) -> None:
    """Generate detailed security report."""
    
    logger = logging.getLogger(__name__)
    
    try:
        # Generate comprehensive report
        if validator:
            # Get the last scan result from validator history
            if validator.scan_history:
                latest_scan = validator.scan_history[-1]
                validator.generate_security_report(latest_scan, report_path)
                logger.info(f"Detailed security report generated: {report_path}")
            else:
                logger.warning("No scan history available for detailed report")
        else:
            logger.error("Validator not available for report generation")
            
    except Exception as e:
        logger.error(f"Failed to generate security report: {e}")


def main():
    """Main entry point for security scanner."""
    
    parser = argparse.ArgumentParser(
        description="Autonomous Security Scanner for SDLC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Security Levels:
  basic     - Basic secret scanning only
  standard  - Secrets + dependency scanning + code security (default)
  strict    - All scans with strict validation (fail on high severity)
  paranoid  - All scans + additional checks with strictest validation

Examples:
  python security_scanner.py /path/to/project
  python security_scanner.py . --level strict --output security_report.json
  python security_scanner.py . --level paranoid --fail-on-high --report security_detailed.json
        """)
    
    parser.add_argument(
        "project_path",
        help="Path to the project directory to scan"
    )
    
    parser.add_argument(
        "--level", "-l",
        choices=["basic", "standard", "strict", "paranoid"],
        default="standard",
        help="Security scanning level (default: standard)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file for scan results (JSON format)"
    )
    
    parser.add_argument(
        "--report", "-r",
        help="Path for detailed security report"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["json"],
        default="json",
        help="Output format (default: json)"
    )
    
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        default=True,
        help="Exit with error code if critical vulnerabilities found (default: True)"
    )
    
    parser.add_argument(
        "--fail-on-high",
        action="store_true",
        default=False,
        help="Exit with error code if high severity vulnerabilities found (default: False)"
    )
    
    parser.add_argument(
        "--no-fail-on-critical",
        action="store_true",
        help="Don't exit with error code for critical vulnerabilities"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress console output (results still written to file)"
    )
    
    args = parser.parse_args()
    
    # Override fail settings
    fail_on_critical = args.fail_on_critical and not args.no_fail_on_critical
    
    # Setup logging level
    if args.quiet:
        log_level = "ERROR"
    else:
        log_level = args.log_level
    
    # Map security level string to enum
    security_level_map = {
        "basic": SecurityLevel.BASIC,
        "standard": SecurityLevel.STANDARD,
        "strict": SecurityLevel.STRICT,
        "paranoid": SecurityLevel.PARANOID
    }
    
    security_level = security_level_map[args.level]
    
    # Validate project path
    project_path = Path(args.project_path).resolve()
    if not project_path.exists():
        print(f"Error: Project path does not exist: {project_path}")
        sys.exit(2)
    
    if not project_path.is_dir():
        print(f"Error: Project path is not a directory: {project_path}")
        sys.exit(2)
    
    # Run security scan
    scan_output = run_security_scan(
        str(project_path),
        security_level,
        args.format,
        args.output,
        fail_on_critical,
        args.fail_on_high
    )
    
    results = scan_output["results"]
    exit_code = scan_output["exit_code"]
    validator = scan_output["validator"]
    
    # Generate detailed report if requested
    if args.report and validator:
        generate_security_report(results, validator, args.report)
    
    # Print summary to console (unless quiet)
    if not args.quiet:
        if "error" in results:
            print(f"Security scan failed: {results['error']}")
        else:
            summary = results["scan_summary"]
            counts = results["vulnerability_counts"]
            
            print("\n" + "="*60)
            print("SECURITY SCAN SUMMARY")
            print("="*60)
            print(f"Project: {project_path}")
            print(f"Security Level: {security_level.value.upper()}")
            print(f"Scan Duration: {summary['duration']:.2f} seconds")
            print(f"Files Scanned: {summary['files_scanned']}")
            print(f"Risk Score: {summary['risk_score']:.1f}/100")
            
            print(f"\nVulnerabilities Found:")
            print(f"  Critical: {counts['critical']}")
            print(f"  High:     {counts['high']}")
            print(f"  Medium:   {counts['medium']}")
            print(f"  Low:      {counts['low']}")
            print(f"  Total:    {sum(counts.values())}")
            
            print(f"\nValidation: {'PASSED' if summary['validation_passed'] else 'FAILED'}")
            if not summary['validation_passed']:
                print(f"Reason: {summary['validation_message']}")
            
            if results.get("recommendations"):
                print(f"\nRecommendations:")
                for i, rec in enumerate(results["recommendations"], 1):
                    print(f"  {i}. {rec}")
            
            if args.output:
                print(f"\nDetailed results saved to: {args.output}")
            
            if args.report:
                print(f"Comprehensive report saved to: {args.report}")
            
            print("="*60)
    
    # Exit with appropriate code
    sys.exit(exit_code)


if __name__ == "__main__":
    main()