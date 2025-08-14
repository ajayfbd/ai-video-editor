#!/usr/bin/env python3
"""
Final Documentation Validation Script
Comprehensive validation for task 12: Final validation and user experience testing
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ValidationResult:
    category: str
    test_name: str
    status: str  # "PASS", "FAIL", "WARNING"
    message: str
    details: List[str] = None

class DocumentationValidator:
    def __init__(self, docs_root: str = "docs"):
        self.docs_root = Path(docs_root)
        self.project_root = Path(".")
        self.results: List[ValidationResult] = []
        
        # Define user journeys to validate
        self.user_journeys = {
            "new_user": [
                "README.md",
                "quick-start.md", 
                "docs/tutorials/first-video.md",
                "docs/tutorials/understanding-output.md"
            ],
            "developer": [
                "README.md",
                "docs/developer/README.md",
                "docs/developer/architecture.md",
                "docs/developer/api-reference.md",
                "docs/developer/contributing.md"
            ],
            "content_creator": [
                "quick-start.md",
                "docs/tutorials/first-video.md",
                "docs/tutorials/workflows/educational-content.md",
                "docs/tutorials/workflows/music-videos.md",
                "docs/tutorials/workflows/general-content.md",
                "docs/tutorials/advanced/performance-tuning.md"
            ],
            "troubleshooter": [
                "docs/support/troubleshooting-unified.md",
                "docs/support/faq-unified.md",
                "docs/support/performance-unified.md",
                "docs/support/error-handling-unified.md"
            ]
        }
        
        # Expected documentation structure
        self.expected_structure = {
            "docs/README.md": "Documentation hub",
            "docs/NAVIGATION.md": "Navigation index",
            "docs/user-guide/README.md": "Complete user guide",
            "docs/tutorials/README.md": "Tutorial index",
            "docs/tutorials/first-video.md": "First video tutorial",
            "docs/developer/README.md": "Developer documentation",
            "docs/developer/architecture.md": "Architecture guide",
            "docs/developer/api-reference.md": "API reference",
            "docs/developer/contributing.md": "Contributing guide",
            "docs/developer/testing.md": "Testing guide",
            "docs/support/troubleshooting-unified.md": "Troubleshooting guide",
            "docs/support/faq-unified.md": "FAQ",
            "docs/support/performance-unified.md": "Performance guide",
            "docs/support/error-handling-unified.md": "Error handling guide"
        }

    def validate_all(self) -> Dict[str, List[ValidationResult]]:
        """Run all validation tests"""
        print("🔍 Starting comprehensive documentation validation...")
        
        # 1. Verify all user journeys have complete documentation coverage
        self.validate_user_journeys()
        
        # 2. Test navigation paths for different user types
        self.validate_navigation_paths()
        
        # 3. Ensure consistent formatting and terminology
        self.validate_formatting_consistency()
        
        # 4. Validate that all essential information has been preserved
        self.validate_information_preservation()
        
        # 5. Check cross-references and link integrity
        self.validate_cross_references()
        
        # 6. Validate content completeness
        self.validate_content_completeness()
        
        # 7. Check for accessibility and usability
        self.validate_accessibility()
        
        return self.categorize_results()

    def validate_user_journeys(self):
        """Validate that all user journeys have complete documentation coverage"""
        print("📋 Validating user journeys...")
        
        for journey_name, journey_files in self.user_journeys.items():
            missing_files = []
            incomplete_files = []
            
            for file_path in journey_files:
                full_path = self.project_root / file_path
                
                if not full_path.exists():
                    missing_files.append(file_path)
                    continue
                
                # Check if file has substantial content
                try:
                    content = full_path.read_text(encoding='utf-8')
                    if len(content.strip()) < 500:  # Minimum content threshold
                        incomplete_files.append(file_path)
                    
                    # Check for essential sections based on file type
                    if not self.has_essential_sections(file_path, content):
                        incomplete_files.append(f"{file_path} (missing sections)")
                        
                except Exception as e:
                    self.results.append(ValidationResult(
                        "user_journeys", f"read_{journey_name}",
                        "FAIL", f"Cannot read {file_path}: {e}"
                    ))
            
            if missing_files:
                self.results.append(ValidationResult(
                    "user_journeys", f"{journey_name}_missing",
                    "FAIL", f"Missing files for {journey_name} journey",
                    missing_files
                ))
            
            if incomplete_files:
                self.results.append(ValidationResult(
                    "user_journeys", f"{journey_name}_incomplete", 
                    "WARNING", f"Incomplete files for {journey_name} journey",
                    incomplete_files
                ))
            
            if not missing_files and not incomplete_files:
                self.results.append(ValidationResult(
                    "user_journeys", f"{journey_name}_complete",
                    "PASS", f"{journey_name.title()} journey has complete documentation"
                ))

    def has_essential_sections(self, file_path: str, content: str) -> bool:
        """Check if file has essential sections based on its type"""
        essential_patterns = {
            "README.md": [r"#.*[Gg]etting [Ss]tarted", r"#.*[Ii]nstallation", r"#.*[Uu]sage"],
            "quick-start.md": [r"#.*[Ss]etup", r"#.*[Ii]nstallation", r"#.*[Ff]irst"],
            "first-video.md": [r"#.*[Ss]tep", r"#.*[Pp]rocess", r"#.*[Rr]esult"],
            "architecture.md": [r"#.*[Aa]rchitecture", r"#.*[Cc]omponent", r"#.*[Dd]esign"],
            "api-reference.md": [r"#.*API", r"#.*[Ee]ndpoint", r"#.*[Rr]eference"],
            "troubleshooting": [r"#.*[Tt]roubleshooting", r"#.*[Ii]ssue", r"#.*[Ss]olution"],
            "faq": [r"#.*FAQ", r"Q:", r"A:"]
        }
        
        filename = Path(file_path).name.lower()
        for pattern_key, patterns in essential_patterns.items():
            if pattern_key in filename or pattern_key in file_path.lower():
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        return True
                return False
        
        return True  # Default to true for files without specific requirements

    def validate_navigation_paths(self):
        """Test navigation paths for different user types"""
        print("🧭 Validating navigation paths...")
        
        # Check main navigation documents
        nav_files = ["docs/README.md", "docs/NAVIGATION.md", "README.md", "quick-start.md"]
        
        for nav_file in nav_files:
            full_path = self.project_root / nav_file
            if not full_path.exists():
                self.results.append(ValidationResult(
                    "navigation", f"missing_{nav_file.replace('/', '_')}",
                    "FAIL", f"Missing navigation file: {nav_file}"
                ))
                continue
            
            try:
                content = full_path.read_text(encoding='utf-8')
                
                # Check for navigation elements
                nav_elements = self.extract_navigation_elements(content)
                
                if len(nav_elements) < 5:  # Minimum navigation links
                    self.results.append(ValidationResult(
                        "navigation", f"sparse_{nav_file.replace('/', '_')}",
                        "WARNING", f"Sparse navigation in {nav_file}",
                        [f"Only {len(nav_elements)} navigation links found"]
                    ))
                else:
                    self.results.append(ValidationResult(
                        "navigation", f"complete_{nav_file.replace('/', '_')}",
                        "PASS", f"Good navigation structure in {nav_file}"
                    ))
                
            except Exception as e:
                self.results.append(ValidationResult(
                    "navigation", f"error_{nav_file.replace('/', '_')}",
                    "FAIL", f"Error reading {nav_file}: {e}"
                ))

    def extract_navigation_elements(self, content: str) -> List[str]:
        """Extract navigation links from content"""
        # Find markdown links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links = re.findall(link_pattern, content)
        
        # Find table of contents entries
        toc_pattern = r'^#+\s+(.+)$'
        headers = re.findall(toc_pattern, content, re.MULTILINE)
        
        return links + [(h, "") for h in headers]

    def validate_formatting_consistency(self):
        """Ensure consistent formatting and terminology across all documents"""
        print("📝 Validating formatting consistency...")
        
        # Collect all markdown files
        md_files = list(self.project_root.glob("**/*.md"))
        
        # Track formatting patterns
        formatting_issues = []
        terminology_issues = []
        
        # Define expected terminology
        expected_terms = {
            "AI Video Editor": ["ai video editor", "ai-video-editor", "aivideoeditor"],
            "ContentContext": ["content context", "content-context", "contentcontext"],
            "AI Director": ["ai director", "ai-director", "aidirector"],
            "Gemini API": ["gemini api", "gemini-api", "geminiapi"]
        }
        
        for md_file in md_files:
            if any(exclude in str(md_file) for exclude in ["archive", ".git", ".agent.md"]):
                continue
                
            try:
                content = md_file.read_text(encoding='utf-8')
                
                # Check formatting consistency
                format_issues = self.check_formatting_issues(content, str(md_file))
                formatting_issues.extend(format_issues)
                
                # Check terminology consistency
                term_issues = self.check_terminology_issues(content, str(md_file), expected_terms)
                terminology_issues.extend(term_issues)
                
            except Exception as e:
                self.results.append(ValidationResult(
                    "formatting", f"read_error_{md_file.name}",
                    "FAIL", f"Cannot read {md_file}: {e}"
                ))
        
        # Report results
        if formatting_issues:
            self.results.append(ValidationResult(
                "formatting", "formatting_inconsistencies",
                "WARNING", f"Found {len(formatting_issues)} formatting inconsistencies",
                formatting_issues[:10]  # Limit to first 10
            ))
        else:
            self.results.append(ValidationResult(
                "formatting", "formatting_consistent",
                "PASS", "Formatting is consistent across documents"
            ))
        
        if terminology_issues:
            self.results.append(ValidationResult(
                "formatting", "terminology_inconsistencies", 
                "WARNING", f"Found {len(terminology_issues)} terminology inconsistencies",
                terminology_issues[:10]  # Limit to first 10
            ))
        else:
            self.results.append(ValidationResult(
                "formatting", "terminology_consistent",
                "PASS", "Terminology is consistent across documents"
            ))

    def check_formatting_issues(self, content: str, file_path: str) -> List[str]:
        """Check for formatting inconsistencies"""
        issues = []
        
        # Check header formatting
        headers = re.findall(r'^(#+)\s*(.+)$', content, re.MULTILINE)
        for level, text in headers:
            if text.strip() != text:
                issues.append(f"{file_path}: Header has extra whitespace: '{text}'")
        
        # Check code block formatting
        code_blocks = re.findall(r'```(\w*)\n(.*?)\n```', content, re.DOTALL)
        for lang, code in code_blocks:
            if not lang and 'bash' in code.lower() or 'python' in code.lower():
                issues.append(f"{file_path}: Code block missing language specification")
        
        # Check list formatting
        list_items = re.findall(r'^(\s*)([-*+]|\d+\.)\s+(.+)$', content, re.MULTILINE)
        for indent, marker, text in list_items:
            if len(indent) % 2 != 0:  # Should be even number of spaces
                issues.append(f"{file_path}: Inconsistent list indentation")
        
        return issues

    def check_terminology_issues(self, content: str, file_path: str, expected_terms: Dict[str, List[str]]) -> List[str]:
        """Check for terminology inconsistencies"""
        issues = []
        
        for correct_term, variations in expected_terms.items():
            for variation in variations:
                if variation in content.lower() and correct_term not in content:
                    issues.append(f"{file_path}: Use '{correct_term}' instead of '{variation}'")
        
        return issues

    def validate_information_preservation(self):
        """Validate that all essential information has been preserved through consolidation"""
        print("🔍 Validating information preservation...")
        
        # Check for essential topics that should be covered
        essential_topics = {
            "installation": ["install", "setup", "requirements", "dependencies"],
            "configuration": ["config", "api key", "environment", ".env"],
            "usage": ["process", "command", "cli", "usage"],
            "troubleshooting": ["error", "issue", "problem", "troubleshoot"],
            "api_reference": ["api", "endpoint", "method", "parameter"],
            "architecture": ["architecture", "design", "component", "module"],
            "testing": ["test", "mock", "coverage", "validation"]
        }
        
        topic_coverage = defaultdict(list)
        
        # Scan all documentation files
        for md_file in self.project_root.glob("**/*.md"):
            if any(exclude in str(md_file) for exclude in ["archive", ".git", ".agent.md"]):
                continue
                
            try:
                content = md_file.read_text(encoding='utf-8').lower()
                
                for topic, keywords in essential_topics.items():
                    for keyword in keywords:
                        if keyword in content:
                            topic_coverage[topic].append(str(md_file))
                            break
                            
            except Exception:
                continue
        
        # Check coverage
        for topic, keywords in essential_topics.items():
            if not topic_coverage[topic]:
                self.results.append(ValidationResult(
                    "information_preservation", f"missing_{topic}",
                    "FAIL", f"No documentation found for essential topic: {topic}",
                    keywords
                ))
            elif len(topic_coverage[topic]) == 1:
                self.results.append(ValidationResult(
                    "information_preservation", f"single_{topic}",
                    "WARNING", f"Only one document covers {topic}",
                    topic_coverage[topic]
                ))
            else:
                self.results.append(ValidationResult(
                    "information_preservation", f"covered_{topic}",
                    "PASS", f"Topic '{topic}' is well covered",
                    [f"{len(topic_coverage[topic])} documents"]
                ))

    def validate_cross_references(self):
        """Check cross-references and link integrity"""
        print("🔗 Validating cross-references and links...")
        
        broken_links = []
        circular_refs = []
        missing_targets = []
        
        for md_file in self.project_root.glob("**/*.md"):
            if "archive" in str(md_file) or ".git" in str(md_file):
                continue
                
            try:
                content = md_file.read_text(encoding='utf-8')
                
                # Find all markdown links
                links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
                
                for link_text, link_target in links:
                    # Skip external links
                    if link_target.startswith(('http://', 'https://', 'mailto:')):
                        continue
                    
                    # Skip anchors for now (would need more complex validation)
                    if link_target.startswith('#'):
                        continue
                    
                    # Resolve relative path
                    if link_target.startswith('./') or link_target.startswith('../'):
                        target_path = (md_file.parent / link_target).resolve()
                    else:
                        target_path = self.project_root / link_target
                    
                    # Check if target exists
                    if not target_path.exists():
                        broken_links.append(f"{md_file}: {link_text} -> {link_target}")
                
            except Exception as e:
                self.results.append(ValidationResult(
                    "cross_references", f"read_error_{md_file.name}",
                    "FAIL", f"Cannot read {md_file}: {e}"
                ))
        
        # Report results
        if broken_links:
            self.results.append(ValidationResult(
                "cross_references", "broken_links",
                "FAIL", f"Found {len(broken_links)} broken internal links",
                broken_links[:10]  # Limit to first 10
            ))
        else:
            self.results.append(ValidationResult(
                "cross_references", "links_valid",
                "PASS", "All internal links are valid"
            ))

    def validate_content_completeness(self):
        """Validate content completeness and quality"""
        print("📚 Validating content completeness...")
        
        # Check expected structure
        missing_files = []
        incomplete_files = []
        
        for file_path, description in self.expected_structure.items():
            full_path = self.project_root / file_path
            
            if not full_path.exists():
                missing_files.append(f"{file_path} ({description})")
                continue
            
            try:
                content = full_path.read_text(encoding='utf-8')
                
                # Check content length (minimum threshold)
                if len(content.strip()) < 1000:
                    incomplete_files.append(f"{file_path} ({len(content)} chars)")
                
                # Check for placeholder content
                if "TODO" in content or "PLACEHOLDER" in content:
                    incomplete_files.append(f"{file_path} (has placeholders)")
                
            except Exception:
                incomplete_files.append(f"{file_path} (read error)")
        
        # Report results
        if missing_files:
            self.results.append(ValidationResult(
                "completeness", "missing_files",
                "FAIL", f"Missing {len(missing_files)} expected files",
                missing_files
            ))
        
        if incomplete_files:
            self.results.append(ValidationResult(
                "completeness", "incomplete_files",
                "WARNING", f"Found {len(incomplete_files)} incomplete files",
                incomplete_files
            ))
        
        if not missing_files and not incomplete_files:
            self.results.append(ValidationResult(
                "completeness", "structure_complete",
                "PASS", "All expected documentation files are present and complete"
            ))

    def validate_accessibility(self):
        """Check for accessibility and usability"""
        print("♿ Validating accessibility and usability...")
        
        accessibility_issues = []
        
        for md_file in self.project_root.glob("**/*.md"):
            if "archive" in str(md_file) or ".git" in str(md_file):
                continue
                
            try:
                content = md_file.read_text(encoding='utf-8')
                
                # Check for alt text on images
                images = re.findall(r'!\[([^\]]*)\]\([^)]+\)', content)
                for alt_text in images:
                    if not alt_text.strip():
                        accessibility_issues.append(f"{md_file}: Image missing alt text")
                
                # Check for descriptive link text
                links = re.findall(r'\[([^\]]+)\]\([^)]+\)', content)
                for link_text in links:
                    if link_text.lower() in ['here', 'click here', 'link', 'read more']:
                        accessibility_issues.append(f"{md_file}: Non-descriptive link text: '{link_text}'")
                
                # Check for proper heading hierarchy
                headers = re.findall(r'^(#+)\s*(.+)$', content, re.MULTILINE)
                prev_level = 0
                for level_str, text in headers:
                    level = len(level_str)
                    if level > prev_level + 1:
                        accessibility_issues.append(f"{md_file}: Heading level skip: {text}")
                    prev_level = level
                
            except Exception:
                continue
        
        # Report results
        if accessibility_issues:
            self.results.append(ValidationResult(
                "accessibility", "accessibility_issues",
                "WARNING", f"Found {len(accessibility_issues)} accessibility issues",
                accessibility_issues[:10]  # Limit to first 10
            ))
        else:
            self.results.append(ValidationResult(
                "accessibility", "accessibility_good",
                "PASS", "No accessibility issues found"
            ))

    def categorize_results(self) -> Dict[str, List[ValidationResult]]:
        """Categorize results by category"""
        categorized = defaultdict(list)
        for result in self.results:
            categorized[result.category].append(result)
        return dict(categorized)

    def generate_report(self) -> str:
        """Generate a comprehensive validation report"""
        categorized = self.categorize_results()
        
        # Count results by status
        total_tests = len(self.results)
        passed = len([r for r in self.results if r.status == "PASS"])
        failed = len([r for r in self.results if r.status == "FAIL"])
        warnings = len([r for r in self.results if r.status == "WARNING"])
        
        report = []
        report.append("# Final Documentation Validation Report")
        report.append("")
        report.append(f"**Overall Status**: {passed}/{total_tests} tests passed")
        report.append(f"- ✅ **Passed**: {passed}")
        report.append(f"- ❌ **Failed**: {failed}")
        report.append(f"- ⚠️ **Warnings**: {warnings}")
        report.append("")
        
        # Success rate
        success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
        report.append(f"**Success Rate**: {success_rate:.1f}%")
        report.append("")
        
        # Category breakdown
        for category, results in categorized.items():
            category_passed = len([r for r in results if r.status == "PASS"])
            category_total = len(results)
            
            report.append(f"## {category.replace('_', ' ').title()}")
            report.append(f"**Status**: {category_passed}/{category_total} passed")
            report.append("")
            
            for result in results:
                status_icon = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️"}[result.status]
                report.append(f"### {status_icon} {result.test_name}")
                report.append(f"**Message**: {result.message}")
                
                if result.details:
                    report.append("**Details**:")
                    for detail in result.details[:5]:  # Limit details
                        report.append(f"- {detail}")
                    if len(result.details) > 5:
                        report.append(f"- ... and {len(result.details) - 5} more")
                
                report.append("")
        
        return "\n".join(report)

def main():
    """Run the validation and generate report"""
    validator = DocumentationValidator()
    
    # Run all validations
    results = validator.validate_all()
    
    # Generate and save report
    report = validator.generate_report()
    
    # Save to file
    report_path = Path("validation_report.md")
    report_path.write_text(report, encoding='utf-8')
    
    # Print summary
    total_tests = len(validator.results)
    passed = len([r for r in validator.results if r.status == "PASS"])
    failed = len([r for r in validator.results if r.status == "FAIL"])
    warnings = len([r for r in validator.results if r.status == "WARNING"])
    
    print("\n" + "="*60)
    print("📊 FINAL VALIDATION SUMMARY")
    print("="*60)
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️ Warnings: {warnings}")
    print(f"Success Rate: {(passed/total_tests)*100:.1f}%")
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    if failed > 0:
        print("\n❌ CRITICAL ISSUES FOUND:")
        for result in validator.results:
            if result.status == "FAIL":
                print(f"  - {result.category}: {result.message}")
    
    if warnings > 0:
        print(f"\n⚠️ {warnings} warnings found - see report for details")
    
    if failed == 0:
        print("\n🎉 All critical validations passed!")
        print("Documentation consolidation task 12 is COMPLETE!")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)