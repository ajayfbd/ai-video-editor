#!/usr/bin/env python3
"""
Test Code Examples Script

This script tests all code examples in the documentation to ensure they work
with the current system implementation.
"""

import os
import sys
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re

class CodeExampleTester:
    """Tests code examples from documentation."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.test_results = []
        
    def test_all_examples(self):
        """Test all code examples in documentation."""
        print("🧪 Testing Code Examples...")
        print("=" * 50)
        
        # Test CLI commands
        self._test_cli_commands()
        
        # Test Python imports
        self._test_python_imports()
        
        # Test configuration examples
        self._test_configuration_examples()
        
        # Test API usage examples
        self._test_api_examples()
        
        # Generate report
        self._generate_test_report()
    
    def _test_cli_commands(self):
        """Test CLI commands from documentation."""
        print("\n🖥️  Testing CLI Commands...")
        
        # Test basic CLI functionality
        cli_tests = [
            {
                "name": "CLI Help",
                "command": [sys.executable, "-m", "ai_video_editor.cli.main", "--help"],
                "expected_in_output": ["AI Video Editor", "Commands:"]
            },
            {
                "name": "CLI Status",
                "command": [sys.executable, "-m", "ai_video_editor.cli.main", "status"],
                "expected_in_output": ["Status"]
            },
            {
                "name": "CLI Init Help",
                "command": [sys.executable, "-m", "ai_video_editor.cli.main", "init", "--help"],
                "expected_in_output": ["Initialize", "configuration"]
            },
            {
                "name": "CLI Process Help",
                "command": [sys.executable, "-m", "ai_video_editor.cli.main", "process", "--help"],
                "expected_in_output": ["Process", "video"]
            }
        ]
        
        for test in cli_tests:
            self._run_cli_test(test)
    
    def _run_cli_test(self, test: Dict[str, Any]):
        """Run a single CLI test."""
        try:
            result = subprocess.run(
                test["command"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            # Check for expected content
            if success and "expected_in_output" in test:
                for expected in test["expected_in_output"]:
                    if expected.lower() not in output.lower():
                        success = False
                        break
            
            self.test_results.append({
                "category": "CLI",
                "name": test["name"],
                "success": success,
                "details": output[:200] + "..." if len(output) > 200 else output,
                "error": result.stderr if result.stderr else None
            })
            
            status = "✅" if success else "❌"
            print(f"   {status} {test['name']}")
            
        except subprocess.TimeoutExpired:
            self.test_results.append({
                "category": "CLI",
                "name": test["name"],
                "success": False,
                "details": "Command timed out",
                "error": "Timeout"
            })
            print(f"   ❌ {test['name']} (timeout)")
            
        except Exception as e:
            self.test_results.append({
                "category": "CLI",
                "name": test["name"],
                "success": False,
                "details": str(e),
                "error": str(e)
            })
            print(f"   ❌ {test['name']} (error: {e})")
    
    def _test_python_imports(self):
        """Test Python imports from documentation."""
        print("\n📦 Testing Python Imports...")
        
        # Key imports that should work
        import_tests = [
            "ai_video_editor",
            "ai_video_editor.core.config",
            "ai_video_editor.core.content_context",
            "ai_video_editor.core.workflow_orchestrator",
            "ai_video_editor.core.exceptions",
            "ai_video_editor.cli.main"
        ]
        
        for import_name in import_tests:
            self._test_import(import_name)
    
    def _test_import(self, import_name: str):
        """Test a single import."""
        try:
            # Test import in a subprocess to avoid polluting current namespace
            test_code = f"import {import_name}"
            result = subprocess.run([
                sys.executable, "-c", test_code
            ], capture_output=True, text=True, timeout=10)
            
            success = result.returncode == 0
            
            self.test_results.append({
                "category": "Import",
                "name": import_name,
                "success": success,
                "details": result.stderr if result.stderr else "Import successful",
                "error": result.stderr if result.stderr else None
            })
            
            status = "✅" if success else "❌"
            print(f"   {status} {import_name}")
            
        except Exception as e:
            self.test_results.append({
                "category": "Import",
                "name": import_name,
                "success": False,
                "details": str(e),
                "error": str(e)
            })
            print(f"   ❌ {import_name} (error: {e})")
    
    def _test_configuration_examples(self):
        """Test configuration examples."""
        print("\n⚙️  Testing Configuration Examples...")
        
        # Test environment variable loading
        config_tests = [
            {
                "name": "Config Module Load",
                "code": """
from ai_video_editor.core.config import get_settings
settings = get_settings()
print("Config loaded successfully")
""",
                "expected_output": "Config loaded successfully"
            },
            {
                "name": "Project Settings Creation",
                "code": """
from ai_video_editor.core.config import ProjectSettings, ContentType, VideoQuality
settings = ProjectSettings(
    content_type=ContentType.EDUCATIONAL,
    quality=VideoQuality.HIGH
)
print("Project settings created")
""",
                "expected_output": "Project settings created"
            }
        ]
        
        for test in config_tests:
            self._run_python_test(test)
    
    def _test_api_examples(self):
        """Test API usage examples."""
        print("\n🔌 Testing API Examples...")
        
        api_tests = [
            {
                "name": "ContentContext Creation",
                "code": """
from ai_video_editor.core.content_context import ContentContext, ContentType
from ai_video_editor.core.content_context import UserPreferences
context = ContentContext(
    project_id="test_project",
    video_files=["test.mp4"],
    content_type=ContentType.GENERAL,
    user_preferences=UserPreferences()
)
print("ContentContext created successfully")
""",
                "expected_output": "ContentContext created successfully"
            },
            {
                "name": "Workflow Configuration",
                "code": """
from ai_video_editor.core.workflow_orchestrator import WorkflowConfiguration, ProcessingMode
from pathlib import Path
config = WorkflowConfiguration(
    processing_mode=ProcessingMode.BALANCED,
    enable_parallel_processing=True,
    max_memory_usage_gb=8.0,
    output_directory=Path("./test_output")
)
print("Workflow configuration created")
""",
                "expected_output": "Workflow configuration created"
            }
        ]
        
        for test in api_tests:
            self._run_python_test(test)
    
    def _run_python_test(self, test: Dict[str, Any]):
        """Run a Python code test."""
        try:
            result = subprocess.run([
                sys.executable, "-c", test["code"]
            ], capture_output=True, text=True, timeout=15)
            
            success = result.returncode == 0
            if success and "expected_output" in test:
                success = test["expected_output"] in result.stdout
            
            self.test_results.append({
                "category": "Python",
                "name": test["name"],
                "success": success,
                "details": result.stdout + result.stderr,
                "error": result.stderr if result.stderr else None
            })
            
            status = "✅" if success else "❌"
            print(f"   {status} {test['name']}")
            
        except Exception as e:
            self.test_results.append({
                "category": "Python",
                "name": test["name"],
                "success": False,
                "details": str(e),
                "error": str(e)
            })
            print(f"   ❌ {test['name']} (error: {e})")
    
    def _generate_test_report(self):
        """Generate a comprehensive test report."""
        print("\n📊 Generating Test Report...")
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate report
        report = f"""# Code Examples Test Report

## Summary
- **Total Tests**: {total_tests}
- **Passed**: {passed_tests}
- **Failed**: {failed_tests}
- **Success Rate**: {success_rate:.1f}%

## Test Results by Category

"""
        
        # Group results by category
        categories = {}
        for result in self.test_results:
            category = result["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        for category, results in categories.items():
            category_passed = sum(1 for r in results if r["success"])
            category_total = len(results)
            
            report += f"### {category} Tests ({category_passed}/{category_total})\n\n"
            
            for result in results:
                status = "✅" if result["success"] else "❌"
                report += f"- {status} **{result['name']}**\n"
                if not result["success"] and result["error"]:
                    report += f"  - Error: {result['error']}\n"
                report += "\n"
        
        # Add recommendations
        if failed_tests > 0:
            report += """## Recommendations

### High Priority Issues
"""
            cli_failures = [r for r in self.test_results if r["category"] == "CLI" and not r["success"]]
            if cli_failures:
                report += "- Fix CLI command issues - these affect basic functionality\n"
            
            import_failures = [r for r in self.test_results if r["category"] == "Import" and not r["success"]]
            if import_failures:
                report += "- Fix import issues - these affect API documentation accuracy\n"
            
            report += """
### Next Steps
1. Review failed tests and fix underlying issues
2. Update documentation to match current implementation
3. Re-run tests to verify fixes
"""
        else:
            report += """## ✅ All Tests Passed!

The code examples in the documentation are working correctly with the current system implementation.
"""
        
        # Save report
        report_file = Path("code_examples_test_report.md")
        report_file.write_text(report, encoding='utf-8')
        
        # Print summary
        print(f"\n📊 Test Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Success Rate: {success_rate:.1f}%")
        
        if failed_tests > 0:
            print(f"\n❌ {failed_tests} tests failed. See code_examples_test_report.md for details.")
        else:
            print(f"\n✅ All tests passed! Code examples are working correctly.")
        
        return failed_tests == 0

def main():
    """Main function."""
    tester = CodeExampleTester()
    success = tester.test_all_examples()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())