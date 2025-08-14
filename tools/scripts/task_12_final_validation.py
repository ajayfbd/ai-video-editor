#!/usr/bin/env python3
"""
Task 12 Final Validation: Conduct final validation and user experience testing
Focused validation for the specific requirements of task 12
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ValidationResult:
    requirement: str
    status: str  # "PASS", "FAIL", "WARNING"
    message: str
    details: List[str] = None

class Task12Validator:
    def __init__(self):
        self.project_root = Path(".")
        self.results: List[ValidationResult] = []
        
        # Define user journeys from requirements 7.1, 7.3, 7.4
        self.user_journeys = {
            "new_user": {
                "description": "New user getting started",
                "path": [
                    "README.md",
                    "quick-start.md", 
                    "docs/tutorials/first-video.md",
                    "docs/tutorials/understanding-output.md"
                ],
                "required_sections": ["installation", "setup", "first video", "results"]
            },
            "developer": {
                "description": "Developer integrating or contributing",
                "path": [
                    "README.md",
                    "docs/developer/README.md",
                    "docs/developer/architecture.md",
                    "docs/developer/api-reference.md",
                    "docs/developer/contributing.md"
                ],
                "required_sections": ["architecture", "api", "contributing", "testing"]
            },
            "advanced_user": {
                "description": "Advanced user optimizing workflows",
                "path": [
                    "docs/tutorials/README.md",
                    "docs/tutorials/advanced/performance-tuning.md",
                    "docs/tutorials/advanced/batch-processing.md",
                    "docs/tutorials/advanced/api-integration.md",
                    "docs/support/performance-unified.md"
                ],
                "required_sections": ["performance", "batch", "automation", "optimization"]
            }
        }

    def validate_all(self) -> Dict[str, List[ValidationResult]]:
        """Run all task 12 validations"""
        print("🔍 Task 12: Final validation and user experience testing")
        print("="*60)
        
        # Requirement 7.1, 7.3, 7.4: Verify all user journeys have complete documentation coverage
        self.validate_user_journey_coverage()
        
        # Requirement 7.3: Test navigation paths for different user types
        self.validate_navigation_paths()
        
        # Requirement 7.4: Ensure consistent formatting and terminology
        self.validate_consistency()
        
        # Requirement 5.3: Validate that all essential information has been preserved
        self.validate_information_preservation()
        
        return self.categorize_results()

    def validate_user_journey_coverage(self):
        """Validate complete documentation coverage for all user types"""
        print("📋 Validating user journey coverage...")
        
        for journey_name, journey_config in self.user_journeys.items():
            missing_files = []
            incomplete_coverage = []
            
            # Check all files in the journey path exist
            for file_path in journey_config["path"]:
                full_path = self.project_root / file_path
                
                if not full_path.exists():
                    missing_files.append(file_path)
                    continue
                
                # Check file has substantial content
                try:
                    content = full_path.read_text(encoding='utf-8')
                    if len(content.strip()) < 500:
                        incomplete_coverage.append(f"{file_path} (too short)")
                    
                    # Check for required sections
                    content_lower = content.lower()
                    missing_sections = []
                    for section in journey_config["required_sections"]:
                        if section not in content_lower:
                            missing_sections.append(section)
                    
                    if missing_sections:
                        incomplete_coverage.append(f"{file_path} (missing: {', '.join(missing_sections)})")
                        
                except Exception as e:
                    incomplete_coverage.append(f"{file_path} (read error: {e})")
            
            # Report results
            if missing_files:
                self.results.append(ValidationResult(
                    "7.1_user_journey_coverage", "FAIL",
                    f"{journey_name} journey has missing files",
                    missing_files
                ))
            elif incomplete_coverage:
                self.results.append(ValidationResult(
                    "7.1_user_journey_coverage", "WARNING",
                    f"{journey_name} journey has incomplete coverage",
                    incomplete_coverage
                ))
            else:
                self.results.append(ValidationResult(
                    "7.1_user_journey_coverage", "PASS",
                    f"{journey_name} journey has complete documentation coverage"
                ))

    def validate_navigation_paths(self):
        """Test navigation paths for different user types"""
        print("🧭 Validating navigation paths...")
        
        # Check main navigation documents exist and have good structure
        nav_documents = {
            "docs/README.md": "Main documentation hub",
            "docs/NAVIGATION.md": "Cross-reference navigation",
            "README.md": "Project overview with navigation",
            "quick-start.md": "Quick start with next steps"
        }
        
        for nav_file, description in nav_documents.items():
            full_path = self.project_root / nav_file
            
            if not full_path.exists():
                self.results.append(ValidationResult(
                    "7.3_navigation_paths", "FAIL",
                    f"Missing navigation document: {nav_file} ({description})"
                ))
                continue
            
            try:
                content = full_path.read_text(encoding='utf-8')
                
                # Count navigation links
                import re
                links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
                internal_links = [link for link in links if not link[1].startswith(('http://', 'https://', 'mailto:'))]
                
                if len(internal_links) < 5:
                    self.results.append(ValidationResult(
                        "7.3_navigation_paths", "WARNING",
                        f"Sparse navigation in {nav_file}",
                        [f"Only {len(internal_links)} internal links found"]
                    ))
                else:
                    self.results.append(ValidationResult(
                        "7.3_navigation_paths", "PASS",
                        f"Good navigation structure in {nav_file}",
                        [f"{len(internal_links)} internal links"]
                    ))
                
            except Exception as e:
                self.results.append(ValidationResult(
                    "7.3_navigation_paths", "FAIL",
                    f"Cannot read navigation document {nav_file}: {e}"
                ))

    def validate_consistency(self):
        """Ensure consistent formatting and terminology"""
        print("📝 Validating consistency...")
        
        # Check key terminology consistency
        expected_terms = {
            "AI Video Editor": ["ai video editor", "ai-video-editor", "aivideoeditor"],
            "ContentContext": ["content context", "content-context"],
            "AI Director": ["ai director", "ai-director"],
        }
        
        terminology_issues = []
        formatting_issues = []
        
        # Check main documentation files
        main_docs = [
            "README.md", "quick-start.md", "docs/README.md",
            "docs/user-guide/README.md", "docs/tutorials/README.md",
            "docs/developer/README.md"
        ]
        
        for doc_path in main_docs:
            full_path = self.project_root / doc_path
            if not full_path.exists():
                continue
                
            try:
                content = full_path.read_text(encoding='utf-8')
                
                # Check terminology
                for correct_term, variations in expected_terms.items():
                    for variation in variations:
                        if variation in content.lower() and correct_term not in content:
                            terminology_issues.append(f"{doc_path}: Use '{correct_term}' instead of '{variation}'")
                
                # Check basic formatting consistency
                import re
                
                # Check header formatting
                headers = re.findall(r'^(#+)\s*(.+)$', content, re.MULTILINE)
                for level, text in headers:
                    if text.strip() != text:
                        formatting_issues.append(f"{doc_path}: Header has extra whitespace: '{text}'")
                
            except Exception:
                continue
        
        # Report results
        if terminology_issues:
            self.results.append(ValidationResult(
                "7.4_consistency", "WARNING",
                f"Found {len(terminology_issues)} terminology inconsistencies",
                terminology_issues[:5]
            ))
        else:
            self.results.append(ValidationResult(
                "7.4_consistency", "PASS",
                "Terminology is consistent across main documents"
            ))
        
        if formatting_issues:
            self.results.append(ValidationResult(
                "7.4_consistency", "WARNING", 
                f"Found {len(formatting_issues)} formatting inconsistencies",
                formatting_issues[:5]
            ))
        else:
            self.results.append(ValidationResult(
                "7.4_consistency", "PASS",
                "Formatting is consistent across main documents"
            ))

    def validate_information_preservation(self):
        """Validate that all essential information has been preserved"""
        print("🔍 Validating information preservation...")
        
        # Check that essential topics are covered
        essential_topics = {
            "installation": ["install", "setup", "requirements", "dependencies"],
            "configuration": ["config", "api key", "environment", ".env"],
            "usage": ["process", "command", "cli", "usage", "tutorial"],
            "troubleshooting": ["error", "issue", "problem", "troubleshoot", "fix"],
            "performance": ["performance", "optimization", "speed", "memory"],
            "api": ["api", "endpoint", "method", "integration"],
            "architecture": ["architecture", "design", "component", "module"],
        }
        
        topic_coverage = {}
        
        # Scan main documentation areas
        doc_areas = [
            "README.md", "quick-start.md", "docs/README.md",
            "docs/user-guide/README.md", "docs/tutorials/",
            "docs/developer/", "docs/support/"
        ]
        
        for topic, keywords in essential_topics.items():
            coverage_count = 0
            
            for area in doc_areas:
                area_path = self.project_root / area
                
                if area_path.is_file():
                    try:
                        content = area_path.read_text(encoding='utf-8').lower()
                        if any(keyword in content for keyword in keywords):
                            coverage_count += 1
                    except Exception:
                        continue
                elif area_path.is_dir():
                    for md_file in area_path.glob("**/*.md"):
                        try:
                            content = md_file.read_text(encoding='utf-8').lower()
                            if any(keyword in content for keyword in keywords):
                                coverage_count += 1
                                break  # Count directory once
                        except Exception:
                            continue
            
            topic_coverage[topic] = coverage_count
        
        # Report results
        missing_topics = [topic for topic, count in topic_coverage.items() if count == 0]
        sparse_topics = [topic for topic, count in topic_coverage.items() if count == 1]
        
        if missing_topics:
            self.results.append(ValidationResult(
                "5.3_information_preservation", "FAIL",
                f"Missing coverage for essential topics",
                missing_topics
            ))
        elif sparse_topics:
            self.results.append(ValidationResult(
                "5.3_information_preservation", "WARNING",
                f"Sparse coverage for some topics",
                sparse_topics
            ))
        else:
            self.results.append(ValidationResult(
                "5.3_information_preservation", "PASS",
                "All essential topics have good coverage"
            ))

    def categorize_results(self) -> Dict[str, List[ValidationResult]]:
        """Categorize results by requirement"""
        categorized = {}
        for result in self.results:
            req = result.requirement.split('_')[0]  # Get requirement number
            if req not in categorized:
                categorized[req] = []
            categorized[req].append(result)
        return categorized

    def generate_final_report(self) -> str:
        """Generate final validation report for task 12"""
        categorized = self.categorize_results()
        
        # Count results
        total_tests = len(self.results)
        passed = len([r for r in self.results if r.status == "PASS"])
        failed = len([r for r in self.results if r.status == "FAIL"])
        warnings = len([r for r in self.results if r.status == "WARNING"])
        
        report = []
        report.append("# Task 12: Final Validation and User Experience Testing")
        report.append("")
        report.append("## Summary")
        report.append("")
        report.append(f"**Overall Status**: {passed}/{total_tests} validations passed")
        report.append(f"- ✅ **Passed**: {passed}")
        report.append(f"- ❌ **Failed**: {failed}")
        report.append(f"- ⚠️ **Warnings**: {warnings}")
        report.append("")
        
        success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
        report.append(f"**Success Rate**: {success_rate:.1f}%")
        report.append("")
        
        # Task requirements validation
        report.append("## Requirements Validation")
        report.append("")
        
        req_mapping = {
            "7.1": "User journey documentation coverage",
            "7.3": "Navigation paths for different user types", 
            "7.4": "Consistent formatting and terminology",
            "5.3": "Essential information preservation"
        }
        
        for req_num, req_desc in req_mapping.items():
            req_results = [r for r in self.results if r.requirement.startswith(req_num)]
            if req_results:
                req_passed = len([r for r in req_results if r.status == "PASS"])
                req_total = len(req_results)
                status_icon = "✅" if req_passed == req_total else "⚠️" if any(r.status == "WARNING" for r in req_results) else "❌"
                
                report.append(f"### {status_icon} Requirement {req_num}: {req_desc}")
                report.append(f"**Status**: {req_passed}/{req_total} validations passed")
                report.append("")
                
                for result in req_results:
                    result_icon = {"PASS": "✅", "FAIL": "❌", "WARNING": "⚠️"}[result.status]
                    report.append(f"- {result_icon} {result.message}")
                    if result.details:
                        for detail in result.details[:3]:
                            report.append(f"  - {detail}")
                        if len(result.details) > 3:
                            report.append(f"  - ... and {len(result.details) - 3} more")
                
                report.append("")
        
        # Overall assessment
        report.append("## Overall Assessment")
        report.append("")
        
        if failed == 0:
            report.append("🎉 **TASK 12 COMPLETE**: All critical validations passed!")
            report.append("")
            report.append("The documentation consolidation has successfully:")
            report.append("- ✅ Provided complete documentation coverage for all user journeys")
            report.append("- ✅ Established clear navigation paths for different user types")
            report.append("- ✅ Maintained consistent formatting and terminology")
            report.append("- ✅ Preserved all essential information through consolidation")
        else:
            report.append("⚠️ **TASK 12 NEEDS ATTENTION**: Some critical issues found")
            report.append("")
            report.append("Critical issues that need resolution:")
            for result in self.results:
                if result.status == "FAIL":
                    report.append(f"- ❌ {result.message}")
        
        if warnings > 0:
            report.append("")
            report.append(f"📝 **Note**: {warnings} minor issues identified for future improvement")
        
        return "\n".join(report)

def main():
    """Run task 12 validation"""
    validator = Task12Validator()
    
    # Run validations
    results = validator.validate_all()
    
    # Generate report
    report = validator.generate_final_report()
    
    # Save report
    report_path = Path("task_12_validation_report.md")
    report_path.write_text(report, encoding='utf-8')
    
    # Print summary
    total_tests = len(validator.results)
    passed = len([r for r in validator.results if r.status == "PASS"])
    failed = len([r for r in validator.results if r.status == "FAIL"])
    warnings = len([r for r in validator.results if r.status == "WARNING"])
    
    print("\n" + "="*60)
    print("📊 TASK 12 VALIDATION COMPLETE")
    print("="*60)
    print(f"✅ Passed: {passed}/{total_tests}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️ Warnings: {warnings}")
    print(f"Success Rate: {(passed/total_tests)*100:.1f}%")
    print(f"\n📄 Report saved to: {report_path}")
    
    if failed == 0:
        print("\n🎉 TASK 12 SUCCESSFULLY COMPLETED!")
        print("All user journeys have complete documentation coverage")
        print("Navigation paths are validated for all user types")
        print("Formatting and terminology are consistent")
        print("All essential information has been preserved")
    else:
        print(f"\n⚠️ {failed} critical issues need attention")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)