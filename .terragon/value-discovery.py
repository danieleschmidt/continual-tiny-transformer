#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Value Discovery Engine
Continuous value discovery and intelligent prioritization system
"""

import json
import yaml
import subprocess
import re
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValueItem:
    """Represents a discovered work item with value scoring"""
    id: str
    title: str
    description: str
    category: str
    source: str
    files_affected: List[str]
    estimated_effort: float  # in hours
    
    # Scoring components
    wsjf_score: float = 0.0
    ice_score: float = 0.0
    technical_debt_score: float = 0.0
    security_score: float = 0.0
    composite_score: float = 0.0
    
    # Risk and confidence
    risk_level: float = 0.0
    confidence: float = 0.0
    
    # Metadata
    discovered_at: str = ""
    priority: str = "medium"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if not self.discovered_at:
            self.discovered_at = datetime.now().isoformat()

class ValueDiscoveryEngine:
    """Main engine for continuous value discovery and prioritization"""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.repo_root = Path.cwd()
        self.value_items: List[ValueItem] = []
        self.execution_history: List[Dict] = []
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration"""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            return {}
    
    def discover_value_items(self) -> List[ValueItem]:
        """Main value discovery orchestrator"""
        logger.info("🔍 Starting autonomous value discovery...")
        
        discovered_items = []
        
        # Execute all discovery sources
        for source in self.config.get('discovery', {}).get('sources', []):
            logger.info(f"Analyzing source: {source}")
            
            if source == 'git_history':
                discovered_items.extend(self._analyze_git_history())
            elif source == 'static_analysis':
                discovered_items.extend(self._analyze_static_code())
            elif source == 'security_scanning':
                discovered_items.extend(self._analyze_security())
            elif source == 'dependency_audit':
                discovered_items.extend(self._analyze_dependencies())
            elif source == 'performance_monitoring':
                discovered_items.extend(self._analyze_performance())
            elif source == 'issue_tracking':
                discovered_items.extend(self._analyze_github_issues())
            elif source == 'code_comments':
                discovered_items.extend(self._analyze_code_comments())
        
        # Score and prioritize items
        scored_items = [self._calculate_scores(item) for item in discovered_items]
        self.value_items = sorted(scored_items, key=lambda x: x.composite_score, reverse=True)
        
        logger.info(f"📊 Discovered {len(self.value_items)} value items")
        return self.value_items
    
    def _analyze_git_history(self) -> List[ValueItem]:
        """Analyze git history for TODO, FIXME, and technical debt markers"""
        items = []
        
        try:
            # Search for debt markers in recent commits
            result = subprocess.run([
                'git', 'log', '--oneline', '--grep=TODO\\|FIXME\\|HACK\\|XXX', 
                '--since=30.days.ago'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    commit_hash = line.split()[0]
                    commit_msg = ' '.join(line.split()[1:])
                    
                    items.append(ValueItem(
                        id=f"git-{commit_hash[:7]}",
                        title=f"Address technical debt: {commit_msg[:50]}...",
                        description=f"Technical debt identified in commit {commit_hash}: {commit_msg}",
                        category="technical_debt",
                        source="git_history",
                        files_affected=self._get_commit_files(commit_hash),
                        estimated_effort=2.0,
                        tags=["debt", "refactoring"]
                    ))
                    
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git history analysis failed: {e}")
            
        return items
    
    def _analyze_static_code(self) -> List[ValueItem]:
        """Run static analysis tools to identify code quality issues"""
        items = []
        
        # Analyze with Ruff for code quality issues
        try:
            result = subprocess.run([
                'ruff', 'check', '--output-format=json', '.'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                ruff_issues = json.loads(result.stdout)
                for issue in ruff_issues[:10]:  # Limit to top 10 issues
                    items.append(ValueItem(
                        id=f"ruff-{issue.get('code', 'unknown')}",
                        title=f"Fix {issue.get('code')}: {issue.get('message', '')[:50]}",
                        description=f"Ruff identified issue: {issue.get('message')}",
                        category="code_quality",
                        source="static_analysis",
                        files_affected=[issue.get('filename', '')],
                        estimated_effort=0.5,
                        tags=["quality", "linting"]
                    ))
                    
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            logger.info("Ruff analysis not available or no issues found")
        
        # Analyze with MyPy for type issues
        try:
            result = subprocess.run([
                'mypy', '--show-error-codes', '--json-report=/tmp/mypy-report', 'src/'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            # Parse MyPy output for type issues
            if result.stderr:
                type_issues = result.stderr.split('\n')
                for issue in type_issues[:5]:  # Limit to top 5 type issues
                    if 'error:' in issue:
                        items.append(ValueItem(
                            id=f"mypy-{hash(issue) % 10000}",
                            title=f"Fix type issue: {issue.split('error:')[1][:50]}",
                            description=f"MyPy type checking issue: {issue}",
                            category="type_safety",
                            source="static_analysis", 
                            files_affected=[issue.split(':')[0] if ':' in issue else ''],
                            estimated_effort=1.0,
                            tags=["types", "quality"]
                        ))
                        
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.info("MyPy analysis not available")
            
        return items
    
    def _analyze_security(self) -> List[ValueItem]:
        """Analyze security vulnerabilities and risks"""
        items = []
        
        # Check for security issues with Bandit
        try:
            result = subprocess.run([
                'bandit', '-r', 'src/', '-f', 'json'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                bandit_results = json.loads(result.stdout)
                for issue in bandit_results.get('results', [])[:5]:
                    severity = issue.get('issue_severity', 'MEDIUM')
                    items.append(ValueItem(
                        id=f"security-{issue.get('test_id', 'unknown')}",
                        title=f"Fix {severity} security issue: {issue.get('issue_text', '')[:50]}",
                        description=f"Security vulnerability: {issue.get('issue_text')}",
                        category="security",
                        source="security_scanning",
                        files_affected=[issue.get('filename', '')],
                        estimated_effort=3.0 if severity == 'HIGH' else 1.5,
                        tags=["security", "vulnerability"]
                    ))
                    
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            logger.info("Bandit security analysis not available")
            
        return items
    
    def _analyze_dependencies(self) -> List[ValueItem]:
        """Analyze dependency vulnerabilities and updates"""
        items = []
        
        # Check for dependency vulnerabilities with Safety
        try:
            result = subprocess.run([
                'safety', 'check', '--json'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                safety_results = json.loads(result.stdout)
                for vuln in safety_results[:3]:  # Top 3 vulnerabilities
                    items.append(ValueItem(
                        id=f"dep-vuln-{vuln.get('id', 'unknown')}",
                        title=f"Fix vulnerability in {vuln.get('package_name', 'unknown')}",
                        description=f"Security vulnerability: {vuln.get('advisory', '')}",
                        category="dependency_security",
                        source="dependency_audit",
                        files_affected=['requirements.txt', 'pyproject.toml'],
                        estimated_effort=1.0,
                        tags=["security", "dependencies"]
                    ))
                    
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            logger.info("Safety dependency check not available")
        
        # Check for outdated dependencies
        try:
            result = subprocess.run([
                'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                outdated = json.loads(result.stdout)
                if len(outdated) > 5:  # Only if many outdated packages
                    items.append(ValueItem(
                        id="dep-updates",
                        title=f"Update {len(outdated)} outdated dependencies",
                        description=f"Update outdated packages: {', '.join([p['name'] for p in outdated[:5]])}",
                        category="dependency_maintenance",
                        source="dependency_audit",
                        files_affected=['requirements.txt', 'pyproject.toml'],
                        estimated_effort=2.0,
                        tags=["maintenance", "dependencies"]
                    ))
                    
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            logger.info("Dependency update check not available")
            
        return items
    
    def _analyze_performance(self) -> List[ValueItem]:
        """Analyze performance issues and optimization opportunities"""
        items = []
        
        # Check if performance tests exist and can be run
        if (Path(self.repo_root) / 'tests' / 'test_performance.py').exists():
            items.append(ValueItem(
                id="perf-monitoring",
                title="Setup automated performance regression detection",
                description="Implement CI-based performance monitoring and trend analysis",
                category="performance",
                source="performance_monitoring",
                files_affected=['.github/workflows/ci.yml'],
                estimated_effort=4.0,
                tags=["performance", "automation"]
            ))
            
        return items
    
    def _analyze_github_issues(self) -> List[ValueItem]:
        """Analyze GitHub issues for work items"""
        items = []
        
        try:
            # Check if gh CLI is available
            result = subprocess.run([
                'gh', 'issue', 'list', '--json', 'number,title,labels,assignees'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.stdout:
                issues = json.loads(result.stdout)
                for issue in issues[:5]:  # Top 5 open issues
                    labels = [label['name'] for label in issue.get('labels', [])]
                    items.append(ValueItem(
                        id=f"issue-{issue['number']}",
                        title=f"Resolve #{issue['number']}: {issue['title'][:50]}",
                        description=f"GitHub issue: {issue['title']}",
                        category="feature_request" if 'enhancement' in labels else "bug_fix",
                        source="issue_tracking",
                        files_affected=[],  # Would need issue analysis
                        estimated_effort=4.0 if 'enhancement' in labels else 2.0,
                        tags=labels
                    ))
                    
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            logger.info("GitHub CLI not available for issue analysis")
            
        return items
    
    def _analyze_code_comments(self) -> List[ValueItem]:
        """Analyze code comments for TODO, FIXME markers"""
        items = []
        
        # Search for TODO/FIXME markers in source code
        patterns = ['TODO', 'FIXME', 'XXX', 'HACK', 'BUG']
        
        for pattern in patterns:
            try:
                result = subprocess.run([
                    'grep', '-r', '-n', '--include=*.py', pattern, 'src/'
                ], capture_output=True, text=True, cwd=self.repo_root)
                
                for line in result.stdout.split('\n')[:3]:  # Limit to 3 per pattern
                    if line.strip():
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            filename, line_num, comment = parts
                            items.append(ValueItem(
                                id=f"comment-{pattern.lower()}-{hash(line) % 1000}",
                                title=f"Address {pattern}: {comment.strip()[:50]}",
                                description=f"Code comment marker at {filename}:{line_num} - {comment.strip()}",
                                category="technical_debt",
                                source="code_comments",
                                files_affected=[filename],
                                estimated_effort=1.0,
                                tags=["debt", "comment"]
                            ))
                            
            except subprocess.CalledProcessError:
                pass  # Pattern not found
                
        return items
    
    def _calculate_scores(self, item: ValueItem) -> ValueItem:
        """Calculate comprehensive value scores for an item"""
        
        # WSJF (Weighted Shortest Job First) calculation
        user_value = self._score_user_value(item)
        time_criticality = self._score_time_criticality(item)
        risk_reduction = self._score_risk_reduction(item)
        opportunity = self._score_opportunity_enablement(item)
        
        cost_of_delay = user_value + time_criticality + risk_reduction + opportunity
        job_size = max(item.estimated_effort, 0.5)  # Minimum 0.5 hours
        item.wsjf_score = cost_of_delay / job_size
        
        # ICE (Impact-Confidence-Ease) calculation
        impact = self._score_impact(item)
        confidence = self._score_confidence(item)
        ease = 10 - min(item.estimated_effort, 10)  # Inverse of effort (easier = higher score)
        item.ice_score = impact * confidence * ease
        
        # Technical debt scoring
        item.technical_debt_score = self._score_technical_debt(item)
        
        # Security scoring
        item.security_score = self._score_security_impact(item)
        
        # Composite score calculation
        weights = self.config.get('scoring', {}).get('weights', {}).get('maturing', {})
        
        item.composite_score = (
            weights.get('wsjf', 0.6) * self._normalize_score(item.wsjf_score, 0, 100) +
            weights.get('ice', 0.1) * self._normalize_score(item.ice_score, 0, 1000) +
            weights.get('technicalDebt', 0.2) * self._normalize_score(item.technical_debt_score, 0, 100) +
            weights.get('security', 0.1) * self._normalize_score(item.security_score, 0, 100)
        )
        
        # Apply category boosts
        category_boosts = {
            'security': self.config.get('scoring', {}).get('thresholds', {}).get('securityBoost', 2.0),
            'dependency_security': 2.0,
            'performance': 1.5,
            'technical_debt': 1.2
        }
        
        if item.category in category_boosts:
            item.composite_score *= category_boosts[item.category]
        
        # Set priority based on composite score
        if item.composite_score >= 70:
            item.priority = "high"
        elif item.composite_score >= 40:
            item.priority = "medium"
        else:
            item.priority = "low"
            
        return item
    
    def _score_user_value(self, item: ValueItem) -> float:
        """Score user/business value (1-10 scale)"""
        category_values = {
            'security': 9,
            'dependency_security': 8,
            'performance': 7,
            'bug_fix': 8,
            'feature_request': 6,
            'code_quality': 5,
            'technical_debt': 4,  
            'documentation': 3
        }
        return category_values.get(item.category, 5)
    
    def _score_time_criticality(self, item: ValueItem) -> float:
        """Score time criticality (1-10 scale)"""
        if 'security' in item.category:
            return 9
        elif 'bug' in item.category:
            return 7
        elif 'performance' in item.category:
            return 6
        else:
            return 4
    
    def _score_risk_reduction(self, item: ValueItem) -> float:
        """Score risk reduction value (1-10 scale)"""
        risk_categories = {
            'security': 10,
            'dependency_security': 9,
            'technical_debt': 6,
            'code_quality': 5
        }
        return risk_categories.get(item.category, 3)
    
    def _score_opportunity_enablement(self, item: ValueItem) -> float:
        """Score opportunity enablement (1-10 scale)"""
        if 'automation' in item.tags:
            return 8
        elif item.category == 'feature_request':
            return 7
        elif item.category == 'performance':
            return 6
        else:
            return 3
    
    def _score_impact(self, item: ValueItem) -> float:
        """Score business impact (1-10 scale)"""
        return self._score_user_value(item)
    
    def _score_confidence(self, item: ValueItem) -> float:
        """Score execution confidence (1-10 scale)"""
        confidence_by_effort = {
            range(0, 2): 9,    # < 2 hours: high confidence
            range(2, 6): 7,    # 2-6 hours: medium-high confidence  
            range(6, 12): 5,   # 6-12 hours: medium confidence
            range(12, 24): 3,  # 12-24 hours: low confidence
        }
        
        for effort_range, confidence in confidence_by_effort.items():
            if item.estimated_effort in effort_range:
                return confidence
        return 2  # Very low confidence for >24 hours
    
    def _score_technical_debt(self, item: ValueItem) -> float:
        """Score technical debt impact (0-100 scale)"""
        if item.category == 'technical_debt':
            return 70
        elif item.category == 'code_quality':
            return 50
        elif 'debt' in item.tags:
            return 60
        else:
            return 20
    
    def _score_security_impact(self, item: ValueItem) -> float:
        """Score security impact (0-100 scale)"""
        if 'security' in item.category:
            return 90
        elif 'vulnerability' in item.tags:
            return 85
        else:
            return 10
    
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-100 range"""
        if max_val == min_val:
            return 50.0
        return min(100.0, max(0.0, ((score - min_val) / (max_val - min_val)) * 100))
    
    def _get_commit_files(self, commit_hash: str) -> List[str]:
        """Get files modified in a commit"""
        try:
            result = subprocess.run([
                'git', 'show', '--name-only', '--pretty=format:', commit_hash
            ], capture_output=True, text=True, cwd=self.repo_root)
            return [f.strip() for f in result.stdout.split('\n') if f.strip()]
        except subprocess.CalledProcessError:
            return []
    
    def get_next_best_value_item(self) -> Optional[ValueItem]:
        """Get the next highest-value item for execution"""
        if not self.value_items:
            self.discover_value_items()
        
        min_score = self.config.get('scoring', {}).get('thresholds', {}).get('minScore', 15)
        
        for item in self.value_items:
            if item.composite_score >= min_score:
                return item
                
        return None
    
    def save_backlog(self, filepath: str = "BACKLOG.md"):
        """Generate and save markdown backlog"""
        backlog_content = self._generate_backlog_markdown()
        
        with open(filepath, 'w') as f:
            f.write(backlog_content)
            
        logger.info(f"📝 Saved backlog to {filepath}")
    
    def _generate_backlog_markdown(self) -> str:
        """Generate markdown representation of value backlog"""
        now = datetime.now().isoformat()
        next_execution = (datetime.now() + timedelta(hours=1)).isoformat()
        
        content = f"""# 📊 Autonomous Value Backlog

Last Updated: {now}
Next Execution: {next_execution}

## 🎯 Next Best Value Item
"""
        
        next_item = self.get_next_best_value_item()
        if next_item:
            content += f"""**[{next_item.id.upper()}] {next_item.title}**
- **Composite Score**: {next_item.composite_score:.1f}
- **WSJF**: {next_item.wsjf_score:.1f} | **ICE**: {next_item.ice_score:.0f} | **Tech Debt**: {next_item.technical_debt_score:.0f}
- **Estimated Effort**: {next_item.estimated_effort} hours
- **Category**: {next_item.category.replace('_', ' ').title()}
- **Description**: {next_item.description}

"""
        else:
            content += "No high-value items currently available for execution.\n\n"
        
        content += """## 📋 Top 10 Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours |
|------|-----|--------|---------|----------|------------|
"""
        
        for i, item in enumerate(self.value_items[:10], 1):
            title = item.title[:40] + "..." if len(item.title) > 40 else item.title
            category = item.category.replace('_', ' ').title()
            content += f"| {i} | {item.id.upper()} | {title} | {item.composite_score:.1f} | {category} | {item.estimated_effort} |\n"
        
        # Value metrics section
        total_items = len(self.value_items)
        high_priority = len([i for i in self.value_items if i.priority == 'high'])
        avg_score = sum(i.composite_score for i in self.value_items) / max(len(self.value_items), 1)
        
        content += f"""
## 📈 Value Metrics
- **Total Items Discovered**: {total_items}
- **High Priority Items**: {high_priority}
- **Average Value Score**: {avg_score:.1f}
- **Security Items**: {len([i for i in self.value_items if 'security' in i.category])}
- **Technical Debt Items**: {len([i for i in self.value_items if 'debt' in i.category])}

## 🔄 Discovery Sources Summary
"""
        
        sources = {}
        for item in self.value_items:
            sources[item.source] = sources.get(item.source, 0) + 1
            
        for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / max(total_items, 1)) * 100
            content += f"- **{source.replace('_', ' ').title()}**: {count} items ({percentage:.1f}%)\n"
        
        content += f"""
## 🏷️ Category Distribution
"""
        
        categories = {}
        for item in self.value_items:
            categories[item.category] = categories.get(item.category, 0) + 1
            
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / max(total_items, 1)) * 100
            content += f"- **{category.replace('_', ' ').title()}**: {count} items ({percentage:.1f}%)\n"
        
        return content
    
    def save_metrics(self, filepath: str = ".terragon/value-metrics.json"):
        """Save value metrics and execution history"""
        metrics = {
            "last_updated": datetime.now().isoformat(),
            "repository": {
                "name": self.config.get('repository', {}).get('name', 'unknown'),
                "maturity_level": self.config.get('repository', {}).get('maturity_level', 'unknown'),
                "maturity_score": self.config.get('repository', {}).get('maturity_score', 0)
            },
            "discovery_summary": {
                "total_items": len(self.value_items),
                "high_priority_items": len([i for i in self.value_items if i.priority == 'high']),
                "average_score": sum(i.composite_score for i in self.value_items) / max(len(self.value_items), 1),
                "categories": {cat: len([i for i in self.value_items if i.category == cat]) 
                             for cat in set(i.category for i in self.value_items)},
                "sources": {src: len([i for i in self.value_items if i.source == src]) 
                           for src in set(i.source for i in self.value_items)}
            },
            "next_best_item": asdict(self.get_next_best_value_item()) if self.get_next_best_value_item() else None,
            "backlog_items": [asdict(item) for item in self.value_items[:20]],  # Top 20 items
            "execution_history": self.execution_history
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"💾 Saved metrics to {filepath}")

def main():
    """Main execution function"""
    engine = ValueDiscoveryEngine()
    
    # Discover and score value items
    items = engine.discover_value_items()
    
    # Generate outputs
    engine.save_backlog()
    engine.save_metrics()
    
    # Display summary
    next_item = engine.get_next_best_value_item()
    if next_item:
        print(f"🎯 Next Best Value Item: {next_item.title}")
        print(f"   Score: {next_item.composite_score:.1f} | Effort: {next_item.estimated_effort}h")
        print(f"   Category: {next_item.category} | Priority: {next_item.priority}")
    else:
        print("🎯 No high-value items currently available for execution")
    
    print(f"📊 Total items discovered: {len(items)}")
    print(f"📝 Backlog saved to BACKLOG.md")
    print(f"💾 Metrics saved to .terragon/value-metrics.json")

if __name__ == "__main__":
    main()