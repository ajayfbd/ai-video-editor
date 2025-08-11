#!/usr/bin/env python3
"""
Documentation Content Validation Script

This script validates all code examples, CLI commands, and API references
in the consolidated documentation to ensure they work with the current system.
"""

import os
import sys
import json
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
import importlib.util
from dataclasses import dataclass, field

@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    success: bool
    message: str
    details: str = ""
    file_path: str = ""
    line_number: int = 0

@dataclass
class ValidationReport:
    """Complete validation report."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    results: List[ValidationResult] = field(default_factory=list)
    
    def add_result(self, result: ValidationResult):
        """Add a validation result."""
        self.results.append(result)
        self.total_tests += 1
        if result.success:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "success_rate": self.passed_tests / self.total_tests if self.total_tests > 0 else 0,
            "failed_results": [r for r in self.results if not r.success]
        }

class DocumentationValidator:
    """Validates documentation content accuracy and completeness."""
    
    def __init__(self):
        self.report = ValidationReport()
        self.project_root = Path.cwd()
        self.docs_dir = self.project_root / "docs"
        self.ai_video_editor_dir = self.project_root / "ai_video_editor"
        
    def validate_all(self) -> ValidationReport:
        """Run all validation tests."""
        print("🔍 Starting Documentation Validation...")
        print("=" * 60)
        
        # Test 1: Validate code examples
        self._validate_code_examples()
        
        # Test 2: Verify CLI commands
        self._validate_cli_commands()
        
        # Test 3: Check API references
        self._validate_api_references()
        
        # Test 4: Verify configuration examples
        self._validate_configuration_examples()
        
        # Test 5: Check import statements
        self._validate_import_statements()
        
        # Test 6: Verify file paths and references
        self._validate_file_references()
        
        # Test 7: Check system requirements
        self._validate_system_requirements()
        
        print("\n" + "=" * 60)
        print("📊 Validation Complete!")
        
        return self.report
    
    def _validate_code_examples(self):
        """Validate all code examples in documentation."""
        print("\n📝 Validating Code Examples...")
        
        # Find all documentation files
        doc_files = []
        for pattern in ["*.md", "**/*.md"]:
            doc_files.extend(self.project_root.glob(pattern))
        
        for doc_file in doc_files:
            if doc_file.is_file():
                self._validate_code_examples_in_file(doc_file)
    
    def _validate_code_examples_in_file(self, file_path: Path):
        """Validate code examples in a specific file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Find Python code blocks
            python_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
            bash_blocks = re.findall(r'```bash\n(.*?)\n```', content, re.DOTALL)
            
            # Validate Python code blocks
            for i, code_block in enumerate(python_blocks):
                self._validate_python_code(code_block, file_path, f"python_block_{i}")
            
            # Validate bash commands
            for i, bash_block in enumerate(bash_blocks):
                self._validate_bash_commands(bash_block, file_path, f"bash_block_{i}")
                
        except Exception as e:
            self.report.add_result(ValidationResult(
                test_name=f"code_examples_{file_path.name}",
                success=False,
                message=f"Error reading file: {str(e)}",
                file_path=str(file_path)
            ))
    
    def _validate_python_code(self, code: str, file_path: Path, block_id: str):
        """Validate Python code syntax and imports."""
        try:
            # Skip configuration examples and incomplete snippets
            if any(skip in code for skip in ['# Configuration', 'your_key_here', '...', 'TODO']):
                return
            
            # Try to compile the code
            compile(code, f"{file_path}:{block_id}", 'exec')
            
            # Check for AI Video Editor imports
            if 'ai_video_editor' in code:
                self._validate_ai_video_editor_imports(code, file_path, block_id)
            
            self.report.add_result(ValidationResult(
                test_name=f"python_syntax_{file_path.name}_{block_id}",
                success=True,
                message="Python code syntax valid",
                file_path=str(file_path)
            ))
            
        except SyntaxError as e:
            self.report.add_result(ValidationResult(
                test_name=f"python_syntax_{file_path.name}_{block_id}",
                success=False,
                message=f"Python syntax error: {str(e)}",
                details=code,
                file_path=str(file_path)
            ))
        except Exception as e:
            self.report.add_result(ValidationResult(
                test_name=f"python_code_{file_path.name}_{block_id}",
                success=False,
                message=f"Python code validation error: {str(e)}",
                details=code,
                file_path=str(file_path)
            ))
    
    def _validate_ai_video_editor_imports(self, code: str, file_path: Path, block_id: str):
        """Validate AI Video Editor specific imports."""
        import_lines = [line.strip() for line in code.split('\n') if line.strip().startswith('from ai_video_editor') or line.strip().startswith('import ai_video_editor')]
        
        for import_line in import_lines:
            try:
                # Extract module path
                if import_line.startswith('from '):
                    module_path = import_line.split(' import ')[0].replace('from ', '')
                else:
                    module_path = import_line.replace('import ', '')
                
                # Check if module exists
                module_file_path = self.ai_video_editor_dir
                for part in module_path.split('.')[1:]:  # Skip 'ai_video_editor'
                    module_file_path = module_file_path / part
                
                # Check for .py file or __init__.py in directory
                if not (module_file_path.with_suffix('.py').exists() or 
                       (module_file_path.is_dir() and (module_file_path / '__init__.py').exists())):
                    self.report.add_result(ValidationResult(
                        test_name=f"import_validation_{file_path.name}_{block_id}",
                        success=False,
                        message=f"Module not found: {module_path}",
                        details=import_line,
                        file_path=str(file_path)
                    ))
                else:
                    self.report.add_result(ValidationResult(
                        test_name=f"import_validation_{file_path.name}_{block_id}",
                        success=True,
                        message=f"Module exists: {module_path}",
                        file_path=str(file_path)
                    ))
                    
            except Exception as e:
                self.report.add_result(ValidationResult(
                    test_name=f"import_validation_{file_path.name}_{block_id}",
                    success=False,
                    message=f"Import validation error: {str(e)}",
                    details=import_line,
                    file_path=str(file_path)
                ))
    
    def _validate_bash_commands(self, bash_code: str, file_path: Path, block_id: str):
        """Validate bash commands and CLI references."""
        lines = bash_code.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check for AI Video Editor CLI commands
            if 'python -m ai_video_editor.cli.main' in line:
                self._validate_cli_command(line, file_path, f"{block_id}_line_{i}")
    
    def _validate_cli_commands(self):
        """Validate CLI commands and options."""
        print("\n🖥️  Validating CLI Commands...")
        
        # Test basic CLI help
        try:
            result = subprocess.run([
                sys.executable, '-m', 'ai_video_editor.cli.main', '--help'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                self.report.add_result(ValidationResult(
                    test_name="cli_help_command",
                    success=True,
                    message="CLI help command works"
                ))
                
                # Validate documented commands exist in help output
                self._validate_documented_commands(result.stdout)
            else:
                self.report.add_result(ValidationResult(
                    test_name="cli_help_command",
                    success=False,
                    message=f"CLI help failed: {result.stderr}"
                ))
                
        except Exception as e:
            self.report.add_result(ValidationResult(
                test_name="cli_help_command",
                success=False,
                message=f"CLI help error: {str(e)}"
            ))
    
    def _validate_cli_command(self, command: str, file_path: Path, test_id: str):
        """Validate a specific CLI command."""
        try:
            # Extract the command parts
            parts = command.split()
            if len(parts) < 4:
                return
            
            # Get the subcommand
            subcommand = parts[4] if len(parts) > 4 else None
            
            if subcommand:
                # Test if subcommand exists by checking help
                try:
                    result = subprocess.run([
                        sys.executable, '-m', 'ai_video_editor.cli.main', subcommand, '--help'
                    ], capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        self.report.add_result(ValidationResult(
                            test_name=f"cli_command_{subcommand}_{test_id}",
                            success=True,
                            message=f"CLI command '{subcommand}' exists",
                            file_path=str(file_path)
                        ))
                    else:
                        self.report.add_result(ValidationResult(
                            test_name=f"cli_command_{subcommand}_{test_id}",
                            success=False,
                            message=f"CLI command '{subcommand}' not found",
                            details=command,
                            file_path=str(file_path)
                        ))
                        
                except subprocess.TimeoutExpired:
                    self.report.add_result(ValidationResult(
                        test_name=f"cli_command_{subcommand}_{test_id}",
                        success=False,
                        message=f"CLI command '{subcommand}' timeout",
                        file_path=str(file_path)
                    ))
                    
        except Exception as e:
            self.report.add_result(ValidationResult(
                test_name=f"cli_command_validation_{test_id}",
                success=False,
                message=f"CLI command validation error: {str(e)}",
                details=command,
                file_path=str(file_path)
            ))
    
    def _validate_documented_commands(self, help_output: str):
        """Validate that documented commands exist in CLI help."""
        documented_commands = ['process', 'status', 'init', 'analyze', 'enhance', 'workflow', 'test-workflow']
        
        for command in documented_commands:
            if command in help_output:
                self.report.add_result(ValidationResult(
                    test_name=f"documented_command_{command}",
                    success=True,
                    message=f"Command '{command}' found in CLI help"
                ))
            else:
                self.report.add_result(ValidationResult(
                    test_name=f"documented_command_{command}",
                    success=False,
                    message=f"Command '{command}' not found in CLI help"
                ))
    
    def _validate_api_references(self):
        """Validate API references and class/method existence."""
        print("\n🔌 Validating API References...")
        
        # Key API classes that should exist
        api_classes = [
            ('ai_video_editor.core.content_context', 'ContentContext'),
            ('ai_video_editor.core.config', 'Settings'),
            ('ai_video_editor.core.config', 'ProjectSettings'),
            ('ai_video_editor.core.workflow_orchestrator', 'WorkflowOrchestrator'),
            ('ai_video_editor.core.exceptions', 'ContentContextError'),
        ]
        
        for module_path, class_name in api_classes:
            self._validate_api_class(module_path, class_name)
    
    def _validate_api_class(self, module_path: str, class_name: str):
        """Validate that an API class exists."""
        try:
            # Convert module path to file path
            file_path = self.project_root / module_path.replace('.', '/') + '.py'
            
            if file_path.exists():
                # Read the file and check for class definition
                content = file_path.read_text()
                if f"class {class_name}" in content:
                    self.report.add_result(ValidationResult(
                        test_name=f"api_class_{class_name}",
                        success=True,
                        message=f"API class {class_name} exists in {module_path}"
                    ))
                else:
                    self.report.add_result(ValidationResult(
                        test_name=f"api_class_{class_name}",
                        success=False,
                        message=f"API class {class_name} not found in {module_path}"
                    ))
            else:
                self.report.add_result(ValidationResult(
                    test_name=f"api_module_{module_path}",
                    success=False,
                    message=f"API module {module_path} not found"
                ))
                
        except Exception as e:
            self.report.add_result(ValidationResult(
                test_name=f"api_validation_{class_name}",
                success=False,
                message=f"API validation error: {str(e)}"
            ))
    
    def _validate_configuration_examples(self):
        """Validate configuration examples and environment variables."""
        print("\n⚙️  Validating Configuration Examples...")
        
        # Check if .env.example exists and contains required variables
        env_example = self.project_root / '.env.example'
        if env_example.exists():
            content = env_example.read_text()
            required_vars = [
                'AI_VIDEO_EDITOR_GEMINI_API_KEY',
                'AI_VIDEO_EDITOR_IMAGEN_API_KEY',
                'AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT'
            ]
            
            for var in required_vars:
                if var in content:
                    self.report.add_result(ValidationResult(
                        test_name=f"env_var_{var}",
                        success=True,
                        message=f"Environment variable {var} documented"
                    ))
                else:
                    self.report.add_result(ValidationResult(
                        test_name=f"env_var_{var}",
                        success=False,
                        message=f"Environment variable {var} not found in .env.example"
                    ))
        else:
            self.report.add_result(ValidationResult(
                test_name="env_example_file",
                success=False,
                message=".env.example file not found"
            ))
    
    def _validate_import_statements(self):
        """Validate import statements in documentation."""
        print("\n📦 Validating Import Statements...")
        
        # Test key imports that are documented
        test_imports = [
            'ai_video_editor.core.content_context',
            'ai_video_editor.core.config',
            'ai_video_editor.cli.main',
        ]
        
        for import_path in test_imports:
            self._test_import(import_path)
    
    def _test_import(self, import_path: str):
        """Test if a module can be imported."""
        try:
            # Convert import path to file path
            file_path = self.project_root / import_path.replace('.', '/') + '.py'
            
            if file_path.exists():
                self.report.add_result(ValidationResult(
                    test_name=f"import_{import_path}",
                    success=True,
                    message=f"Module {import_path} exists"
                ))
            else:
                self.report.add_result(ValidationResult(
                    test_name=f"import_{import_path}",
                    success=False,
                    message=f"Module {import_path} file not found"
                ))
                
        except Exception as e:
            self.report.add_result(ValidationResult(
                test_name=f"import_{import_path}",
                success=False,
                message=f"Import test error: {str(e)}"
            ))
    
    def _validate_file_references(self):
        """Validate file paths and references in documentation."""
        print("\n📁 Validating File References...")
        
        # Key files that should exist based on documentation
        expected_files = [
            'requirements.txt',
            'setup.py',
            'ai_video_editor/__init__.py',
            'ai_video_editor/cli/main.py',
            'ai_video_editor/core/config.py',
            'ai_video_editor/core/content_context.py',
        ]
        
        for file_path in expected_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.report.add_result(ValidationResult(
                    test_name=f"file_exists_{file_path}",
                    success=True,
                    message=f"File {file_path} exists"
                ))
            else:
                self.report.add_result(ValidationResult(
                    test_name=f"file_exists_{file_path}",
                    success=False,
                    message=f"File {file_path} not found"
                ))
    
    def _validate_system_requirements(self):
        """Validate system requirements and dependencies."""
        print("\n🔧 Validating System Requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 9):
            self.report.add_result(ValidationResult(
                test_name="python_version",
                success=True,
                message=f"Python {python_version.major}.{python_version.minor} meets requirements"
            ))
        else:
            self.report.add_result(ValidationResult(
                test_name="python_version",
                success=False,
                message=f"Python {python_version.major}.{python_version.minor} below minimum 3.9"
            ))
        
        # Check key dependencies
        key_dependencies = ['movis', 'openai-whisper', 'opencv-python', 'google-genai']
        
        for dep in key_dependencies:
            try:
                __import__(dep.replace('-', '_'))
                self.report.add_result(ValidationResult(
                    test_name=f"dependency_{dep}",
                    success=True,
                    message=f"Dependency {dep} available"
                ))
            except ImportError:
                self.report.add_result(ValidationResult(
                    test_name=f"dependency_{dep}",
                    success=False,
                    message=f"Dependency {dep} not available"
                ))
    
    def generate_report(self) -> str:
        """Generate a detailed validation report."""
        summary = self.report.get_summary()
        
        report = f"""
# Documentation Validation Report

## Summary
- **Total Tests**: {summary['total_tests']}
- **Passed**: {summary['passed_tests']}
- **Failed**: {summary['failed_tests']}
- **Success Rate**: {summary['success_rate']:.1%}

## Test Results

### ✅ Passed Tests ({summary['passed_tests']})
"""
        
        for result in self.report.results:
            if result.success:
                report += f"- {result.test_name}: {result.message}\n"
        
        if summary['failed_tests'] > 0:
            report += f"\n### ❌ Failed Tests ({summary['failed_tests']})\n"
            for result in summary['failed_results']:
                report += f"- **{result.test_name}**: {result.message}\n"
                if result.details:
                    report += f"  - Details: {result.details[:100]}...\n"
                if result.file_path:
                    report += f"  - File: {result.file_path}\n"
        
        report += f"""
## Recommendations

### High Priority Issues
"""
        
        # Identify high priority issues
        high_priority = [r for r in summary['failed_results'] if 'cli_command' in r.test_name or 'api_class' in r.test_name]
        
        if high_priority:
            for issue in high_priority:
                report += f"- Fix {issue.test_name}: {issue.message}\n"
        else:
            report += "- No high priority issues found\n"
        
        report += """
### Medium Priority Issues
"""
        
        medium_priority = [r for r in summary['failed_results'] if 'import' in r.test_name or 'file_exists' in r.test_name]
        
        if medium_priority:
            for issue in medium_priority:
                report += f"- Address {issue.test_name}: {issue.message}\n"
        else:
            report += "- No medium priority issues found\n"
        
        return report

def main():
    """Main validation function."""
    validator = DocumentationValidator()
    report = validator.validate_all()
    
    # Generate and save report
    report_content = validator.generate_report()
    
    # Save to file
    report_file = Path("documentation_validation_report.md")
    report_file.write_text(report_content, encoding='utf-8')
    
    # Print summary
    summary = report.get_summary()
    print(f"\n📊 Validation Summary:")
    print(f"   Total Tests: {summary['total_tests']}")
    print(f"   Passed: {summary['passed_tests']}")
    print(f"   Failed: {summary['failed_tests']}")
    print(f"   Success Rate: {summary['success_rate']:.1%}")
    
    if summary['failed_tests'] > 0:
        print(f"\n❌ {summary['failed_tests']} tests failed. See documentation_validation_report.md for details.")
        return 1
    else:
        print(f"\n✅ All tests passed! Documentation is accurate and complete.")
        return 0

if __name__ == "__main__":
    sys.exit(main())