#!/usr/bin/env python3
"""
Documentation Maintenance Automation Script

This script runs regular maintenance checks on documentation based on
the maintenance schedule defined in documentation_config.json.
"""

import os
import json
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


class DocumentationMaintenance:
    """Automated documentation maintenance system"""
    
    def __init__(self, config_path: str = "scripts/documentation_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.root_path = Path(".")
        
    def _load_config(self) -> Dict:
        """Load maintenance configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if config file is missing"""
        return {
            "maintenance_schedule": {
                "daily_checks": ["broken_links", "file_errors"],
                "weekly_checks": ["heading_hierarchy", "code_syntax", "content_quality"],
                "monthly_checks": ["comprehensive_review"],
                "quarterly_checks": ["structure_review"]
            },
            "validation_settings": {
                "severity_levels": {
                    "broken_link": "error",
                    "heading_hierarchy": "warning",
                    "code_syntax": "warning",
                    "todo_comment": "info",
                    "content_length": "warning",
                    "file_error": "error"
                }
            }
        }
    
    def run_daily_checks(self) -> bool:
        """Run daily maintenance checks"""
        print("=== Running Daily Documentation Checks ===")
        
        # Run validation with error-level issues only
        result = self._run_validation(severity="error")
        
        if result["success"]:
            print("✅ Daily checks passed - no critical issues found")
            return True
        else:
            print("❌ Daily checks failed - critical issues found")
            self._report_issues(result["issues"])
            return False
    
    def run_weekly_checks(self) -> bool:
        """Run weekly maintenance checks"""
        print("=== Running Weekly Documentation Checks ===")
        
        # Run validation with warning-level issues
        result = self._run_validation(severity="warning")
        
        # Generate weekly report
        self._generate_weekly_report(result)
        
        if result["success"]:
            print("✅ Weekly checks completed successfully")
            return True
        else:
            print("⚠️  Weekly checks found issues requiring attention")
            return False
    
    def run_monthly_checks(self) -> bool:
        """Run monthly comprehensive checks"""
        print("=== Running Monthly Documentation Checks ===")
        
        # Run full validation
        result = self._run_validation(severity="info")
        
        # Additional monthly checks
        self._check_content_freshness()
        self._validate_user_journeys()
        
        # Generate monthly report
        self._generate_monthly_report(result)
        
        print("📊 Monthly comprehensive check completed")
        return True
    
    def run_quarterly_checks(self) -> bool:
        """Run quarterly structural review"""
        print("=== Running Quarterly Documentation Review ===")
        
        # Full validation
        result = self._run_validation(severity="info")
        
        # Structural analysis
        self._analyze_documentation_structure()
        self._audit_cross_references()
        
        # Generate quarterly report
        self._generate_quarterly_report(result)
        
        print("📈 Quarterly review completed")
        return True
    
    def _run_validation(self, severity: str = "warning") -> Dict:
        """Run documentation validation script"""
        try:
            cmd = [
                "python", "scripts/validate_documentation.py",
                "--severity", severity,
                "--format", "json"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.stdout:
                validation_data = json.loads(result.stdout)
                return {
                    "success": result.returncode == 0,
                    "issues": validation_data.get("issues", []),
                    "summary": validation_data.get("summary", {}),
                    "total_issues": validation_data.get("total_issues", 0)
                }
            else:
                return {"success": False, "issues": [], "error": result.stderr}
                
        except Exception as e:
            return {"success": False, "issues": [], "error": str(e)}
    
    def _report_issues(self, issues: List[Dict]):
        """Report validation issues"""
        if not issues:
            return
        
        print(f"\n📋 Found {len(issues)} issues:")
        
        # Group issues by severity
        by_severity = {}
        for issue in issues:
            severity = issue.get("severity", "unknown")
            if severity not in by_severity:
                by_severity[severity] = []
            by_severity[severity].append(issue)
        
        # Report by severity
        for severity in ["error", "warning", "info"]:
            if severity in by_severity:
                print(f"\n{severity.upper()} ({len(by_severity[severity])} issues):")
                for issue in by_severity[severity][:5]:  # Show first 5
                    location = issue.get("file_path", "unknown")
                    if issue.get("line_number"):
                        location += f":{issue['line_number']}"
                    print(f"  • {location}: {issue.get('issue_description', 'Unknown issue')}")
                
                if len(by_severity[severity]) > 5:
                    print(f"  ... and {len(by_severity[severity]) - 5} more")
    
    def _check_content_freshness(self):
        """Check if content is up to date"""
        print("🔍 Checking content freshness...")
        
        # Check for files that haven't been updated recently
        cutoff_date = datetime.now() - timedelta(days=90)  # 3 months
        old_files = []
        
        for doc_file in self.root_path.rglob("*.md"):
            if any(excluded in str(doc_file) for excluded in [".git", "node_modules"]):
                continue
                
            try:
                mtime = datetime.fromtimestamp(doc_file.stat().st_mtime)
                if mtime < cutoff_date:
                    old_files.append((str(doc_file), mtime))
            except:
                continue
        
        if old_files:
            print(f"⚠️  Found {len(old_files)} files not updated in 3+ months:")
            for file_path, mtime in old_files[:5]:
                print(f"  • {file_path} (last updated: {mtime.strftime('%Y-%m-%d')})")
        else:
            print("✅ All documentation appears to be recently updated")
    
    def _validate_user_journeys(self):
        """Validate that user journeys are complete"""
        print("🚶 Validating user journeys...")
        
        # Check key user journey files exist and are linked properly
        key_journeys = [
            "quick-start.md",
            "docs/user-guide/README.md",
            "docs/tutorials/first-video.md",
            "docs/developer/README.md"
        ]
        
        missing_journeys = []
        for journey in key_journeys:
            if not (self.root_path / journey).exists():
                missing_journeys.append(journey)
        
        if missing_journeys:
            print(f"❌ Missing key user journey files: {missing_journeys}")
        else:
            print("✅ All key user journey files present")
    
    def _analyze_documentation_structure(self):
        """Analyze overall documentation structure"""
        print("🏗️  Analyzing documentation structure...")
        
        # Count files by category
        structure_stats = {
            "total_files": 0,
            "user_guide": 0,
            "developer_guide": 0,
            "tutorials": 0,
            "support": 0,
            "other": 0
        }
        
        for doc_file in self.root_path.rglob("*.md"):
            if any(excluded in str(doc_file) for excluded in [".git", "node_modules"]):
                continue
                
            structure_stats["total_files"] += 1
            
            file_str = str(doc_file)
            if "user-guide" in file_str:
                structure_stats["user_guide"] += 1
            elif "developer" in file_str:
                structure_stats["developer_guide"] += 1
            elif "tutorials" in file_str:
                structure_stats["tutorials"] += 1
            elif "support" in file_str:
                structure_stats["support"] += 1
            else:
                structure_stats["other"] += 1
        
        print("📊 Documentation structure:")
        for category, count in structure_stats.items():
            print(f"  • {category}: {count} files")
    
    def _audit_cross_references(self):
        """Audit cross-reference integrity"""
        print("🔗 Auditing cross-references...")
        
        # This would be a more comprehensive check than daily validation
        # For now, just run the validation and report cross-reference specific issues
        result = self._run_validation(severity="info")
        
        link_issues = [issue for issue in result.get("issues", []) 
                      if issue.get("issue_type") == "broken_link"]
        
        if link_issues:
            print(f"⚠️  Found {len(link_issues)} cross-reference issues")
        else:
            print("✅ Cross-reference integrity looks good")
    
    def _generate_weekly_report(self, validation_result: Dict):
        """Generate weekly maintenance report"""
        report_path = Path("reports") / f"weekly-maintenance-{datetime.now().strftime('%Y-%m-%d')}.md"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(f"# Weekly Documentation Maintenance Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- Total issues found: {validation_result.get('total_issues', 0)}\n")
            f.write(f"- Validation passed: {'✅ Yes' if validation_result.get('success') else '❌ No'}\n\n")
            
            if validation_result.get("issues"):
                f.write(f"## Issues by Type\n\n")
                issue_types = {}
                for issue in validation_result["issues"]:
                    issue_type = issue.get("issue_type", "unknown")
                    issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
                
                for issue_type, count in sorted(issue_types.items()):
                    f.write(f"- {issue_type}: {count}\n")
        
        print(f"📄 Weekly report generated: {report_path}")
    
    def _generate_monthly_report(self, validation_result: Dict):
        """Generate monthly maintenance report"""
        report_path = Path("reports") / f"monthly-maintenance-{datetime.now().strftime('%Y-%m')}.md"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(f"# Monthly Documentation Maintenance Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Comprehensive Analysis\n\n")
            f.write(f"This report includes validation results, content freshness analysis, ")
            f.write(f"and user journey validation.\n\n")
            f.write(f"### Validation Summary\n\n")
            f.write(f"- Total issues: {validation_result.get('total_issues', 0)}\n")
            f.write(f"- Status: {'✅ Passed' if validation_result.get('success') else '❌ Issues found'}\n\n")
        
        print(f"📄 Monthly report generated: {report_path}")
    
    def _generate_quarterly_report(self, validation_result: Dict):
        """Generate quarterly maintenance report"""
        report_path = Path("reports") / f"quarterly-maintenance-{datetime.now().strftime('%Y-Q%q')}.md"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(f"# Quarterly Documentation Review Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Strategic Review\n\n")
            f.write(f"This quarterly review includes structural analysis, ")
            f.write(f"cross-reference auditing, and strategic recommendations.\n\n")
        
        print(f"📄 Quarterly report generated: {report_path}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Run documentation maintenance checks")
    parser.add_argument("check_type", choices=["daily", "weekly", "monthly", "quarterly"],
                       help="Type of maintenance check to run")
    parser.add_argument("--config", default="scripts/documentation_config.json",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Initialize maintenance system
    maintenance = DocumentationMaintenance(args.config)
    
    # Run appropriate check
    if args.check_type == "daily":
        success = maintenance.run_daily_checks()
    elif args.check_type == "weekly":
        success = maintenance.run_weekly_checks()
    elif args.check_type == "monthly":
        success = maintenance.run_monthly_checks()
    elif args.check_type == "quarterly":
        success = maintenance.run_quarterly_checks()
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()