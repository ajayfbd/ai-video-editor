#!/usr/bin/env python3
"""
Documentation Quality Assurance and Validation Script

This script provides automated validation for cross-references, link integrity,
and documentation quality standards for the AI Video Editor project.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ValidationResult:
    """Results from documentation validation"""
    file_path: str
    issue_type: str
    issue_description: str
    line_number: Optional[int] = None
    severity: str = "warning"  # error, warning, info


@dataclass
class ValidationReport:
    """Complete validation report"""
    timestamp: str
    total_files_checked: int
    total_issues: int
    issues_by_severity: Dict[str, int]
    issues: List[ValidationResult]
    summary: Dict[str, any]


class DocumentationValidator:
    """Main validation class for documentation quality assurance"""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.docs_path = self.root_path / "docs"
        self.issues: List[ValidationResult] = []
        
        # Documentation file patterns
        self.doc_patterns = ["*.md", "*.rst", "*.txt"]
        
        # Link patterns for validation
        self.internal_link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        self.heading_pattern = re.compile(r'^#+\s+(.+)$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```[\s\S]*?```')
        
    def validate_all(self) -> ValidationReport:
        """Run all validation checks"""
        print("Starting documentation validation...")
        
        # Get all documentation files
        doc_files = self._get_documentation_files()
        
        # Run validation checks
        for file_path in doc_files:
            self._validate_file(file_path)
        
        # Generate report
        report = self._generate_report(len(doc_files))
        return report
    
    def _get_documentation_files(self) -> List[Path]:
        """Get all documentation files in the project"""
        doc_files = []
        
        # Check docs directory
        if self.docs_path.exists():
            for pattern in self.doc_patterns:
                doc_files.extend(self.docs_path.rglob(pattern))
        
        # Check root level documentation
        for pattern in self.doc_patterns:
            doc_files.extend(self.root_path.glob(pattern))
        
        # Filter out excluded directories
        excluded_dirs = {'.git', 'node_modules', '__pycache__', '.pytest_cache'}
        doc_files = [f for f in doc_files if not any(part in excluded_dirs for part in f.parts)]
        
        return sorted(doc_files)
    
    def _validate_file(self, file_path: Path):
        """Validate a single documentation file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Run validation checks
            self._check_internal_links(file_path, content)
            self._check_heading_structure(file_path, content)
            self._check_code_examples(file_path, content)
            self._check_content_quality(file_path, content)
            
        except Exception as e:
            self.issues.append(ValidationResult(
                file_path=str(file_path),
                issue_type="file_error",
                issue_description=f"Error reading file: {str(e)}",
                severity="error"
            ))
    
    def _check_internal_links(self, file_path: Path, content: str):
        """Check internal link integrity"""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            matches = self.internal_link_pattern.findall(line)
            
            for link_text, link_url in matches:
                # Skip external links
                if link_url.startswith(('http://', 'https://', 'mailto:')):
                    continue
                
                # Check if internal link target exists
                if not self._validate_internal_link(file_path, link_url):
                    self.issues.append(ValidationResult(
                        file_path=str(file_path),
                        issue_type="broken_link",
                        issue_description=f"Broken internal link: '{link_text}' -> '{link_url}'",
                        line_number=line_num,
                        severity="error"
                    ))
    
    def _validate_internal_link(self, current_file: Path, link_url: str) -> bool:
        """Validate that an internal link target exists"""
        # Remove anchor fragments
        if '#' in link_url:
            file_part, anchor = link_url.split('#', 1)
        else:
            file_part, anchor = link_url, None
        
        # Resolve relative path
        if file_part:
            target_path = (current_file.parent / file_part).resolve()
            if not target_path.exists():
                return False
        else:
            target_path = current_file
        
        # If there's an anchor, check if the heading exists
        if anchor and target_path.exists():
            return self._check_anchor_exists(target_path, anchor)
        
        return target_path.exists() if file_part else True
    
    def _check_anchor_exists(self, file_path: Path, anchor: str) -> bool:
        """Check if a heading anchor exists in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            headings = self.heading_pattern.findall(content)
            # Convert heading to anchor format (lowercase, spaces to hyphens)
            anchor_headings = [h.lower().replace(' ', '-').replace('/', '').replace('.', '') for h in headings]
            
            return anchor.lower() in anchor_headings
        except:
            return False
    
    def _check_heading_structure(self, file_path: Path, content: str):
        """Check heading structure and hierarchy"""
        lines = content.split('\n')
        heading_levels = []
        
        for line_num, line in enumerate(lines, 1):
            if line.strip().startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                heading_levels.append((level, line_num))
                
                # Check for proper heading hierarchy
                if len(heading_levels) > 1:
                    prev_level = heading_levels[-2][0]
                    if level > prev_level + 1:
                        self.issues.append(ValidationResult(
                            file_path=str(file_path),
                            issue_type="heading_hierarchy",
                            issue_description=f"Heading level jumps from {prev_level} to {level}",
                            line_number=line_num,
                            severity="warning"
                        ))
    
    def _check_code_examples(self, file_path: Path, content: str):
        """Check code examples for basic syntax issues"""
        code_blocks = self.code_block_pattern.findall(content)
        
        for i, block in enumerate(code_blocks):
            # Check for common issues in code blocks
            if '```python' in block and 'import' in block:
                # Basic Python syntax check
                if block.count('(') != block.count(')'):
                    self.issues.append(ValidationResult(
                        file_path=str(file_path),
                        issue_type="code_syntax",
                        issue_description=f"Unmatched parentheses in code block {i+1}",
                        severity="warning"
                    ))
    
    def _check_content_quality(self, file_path: Path, content: str):
        """Check content quality and consistency"""
        lines = content.split('\n')
        
        # Check for TODO/FIXME comments
        for line_num, line in enumerate(lines, 1):
            if re.search(r'\b(TODO|FIXME|XXX)\b', line, re.IGNORECASE):
                self.issues.append(ValidationResult(
                    file_path=str(file_path),
                    issue_type="todo_comment",
                    issue_description=f"TODO/FIXME comment found: {line.strip()}",
                    line_number=line_num,
                    severity="info"
                ))
        
        # Check for very short files (might be incomplete)
        if len(content.strip()) < 100:
            self.issues.append(ValidationResult(
                file_path=str(file_path),
                issue_type="content_length",
                issue_description="File appears to be very short or incomplete",
                severity="warning"
            ))
    
    def _generate_report(self, total_files: int) -> ValidationReport:
        """Generate validation report"""
        issues_by_severity = {"error": 0, "warning": 0, "info": 0}
        
        for issue in self.issues:
            issues_by_severity[issue.severity] += 1
        
        summary = {
            "files_with_issues": len(set(issue.file_path for issue in self.issues)),
            "most_common_issues": self._get_most_common_issues(),
            "validation_passed": issues_by_severity["error"] == 0
        }
        
        return ValidationReport(
            timestamp=datetime.now().isoformat(),
            total_files_checked=total_files,
            total_issues=len(self.issues),
            issues_by_severity=issues_by_severity,
            issues=self.issues,
            summary=summary
        )
    
    def _get_most_common_issues(self) -> Dict[str, int]:
        """Get most common issue types"""
        issue_counts = {}
        for issue in self.issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
        
        return dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5])


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Validate documentation quality")
    parser.add_argument("--root", default=".", help="Root directory to validate")
    parser.add_argument("--output", help="Output file for validation report")
    parser.add_argument("--format", choices=["json", "text"], default="text", help="Output format")
    parser.add_argument("--severity", choices=["error", "warning", "info"], default="warning", 
                       help="Minimum severity level to report")
    
    args = parser.parse_args()
    
    # Run validation
    validator = DocumentationValidator(args.root)
    report = validator.validate_all()
    
    # Filter by severity
    severity_levels = {"error": 3, "warning": 2, "info": 1}
    min_level = severity_levels[args.severity]
    filtered_issues = [issue for issue in report.issues 
                      if severity_levels[issue.severity] >= min_level]
    
    # Output report
    if args.format == "json":
        output_data = asdict(report)
        output_data["issues"] = [asdict(issue) for issue in filtered_issues]
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
        else:
            print(json.dumps(output_data, indent=2))
    else:
        # Text format
        print(f"\n=== Documentation Validation Report ===")
        print(f"Timestamp: {report.timestamp}")
        print(f"Files checked: {report.total_files_checked}")
        print(f"Total issues: {len(filtered_issues)}")
        print(f"Validation passed: {report.summary['validation_passed']}")
        
        if filtered_issues:
            print(f"\n=== Issues Found ===")
            for issue in filtered_issues:
                location = f"{issue.file_path}"
                if issue.line_number:
                    location += f":{issue.line_number}"
                
                print(f"[{issue.severity.upper()}] {location}")
                print(f"  {issue.issue_type}: {issue.issue_description}")
                print()
        
        if report.summary["most_common_issues"]:
            print("=== Most Common Issues ===")
            for issue_type, count in report.summary["most_common_issues"].items():
                print(f"  {issue_type}: {count}")
    
    # Exit with error code if validation failed
    if not report.summary["validation_passed"]:
        exit(1)


if __name__ == "__main__":
    main()