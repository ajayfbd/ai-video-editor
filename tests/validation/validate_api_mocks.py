# tests/validation/validate_api_mocks.py
"""
API mock completeness validation for pre-commit hooks.
Ensures all external API dependencies are properly mocked.
"""

import sys
import ast
import traceback
from typing import Dict, List, Set, Any
from pathlib import Path


class APIMockValidator:
    """Validates that all external API calls are properly mocked."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.project_root = Path(__file__).parent.parent.parent
        self.ai_video_editor_path = self.project_root / "ai_video_editor"
        
        # Known external APIs that should be mocked
        self.external_apis = {
            "gemini_api": ["analyze_content", "research_keywords", "analyze_competitors", "predict_engagement"],
            "imagen_api": ["generate_background", "generate_variations"],
            "whisper_api": ["transcribe", "transcribe_with_timestamps", "identify_speakers"],
            "openai_api": ["chat_completion", "embedding"],
            "requests": ["get", "post", "put", "delete"],
            "urllib": ["urlopen", "urlretrieve"]
        }
        
        # Modules that are expected to make external API calls
        self.api_modules = [
            "modules/content_analysis",
            "modules/audio_analysis", 
            "modules/thumbnail_generation",
            "modules/intelligence"
        ]
    
    def validate_all(self) -> bool:
        """Run all API mock validations."""
        print("🔍 Validating API mock completeness...")
        
        try:
            self.scan_for_external_api_calls()
            self.validate_mock_coverage()
            self.validate_mock_implementations()
            self.validate_test_isolation()
            
            return self.report_results()
            
        except Exception as e:
            self.errors.append(f"API mock validation failed with exception: {str(e)}")
            print(f"❌ Validation failed: {e}")
            traceback.print_exc()
            return False
    
    def scan_for_external_api_calls(self):
        """Scan source code for external API calls."""
        print("Scanning for external API calls...")
        
        found_api_calls = {}
        
        for module_path in self.api_modules:
            full_path = self.ai_video_editor_path / module_path
            if not full_path.exists():
                self.warnings.append(f"Expected API module not found: {module_path}")
                continue
            
            # Scan Python files in the module
            for py_file in full_path.rglob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                
                try:
                    api_calls = self._extract_api_calls_from_file(py_file)
                    if api_calls:
                        found_api_calls[str(py_file.relative_to(self.project_root))] = api_calls
                except Exception as e:
                    self.warnings.append(f"Failed to scan {py_file}: {str(e)}")
        
        self.found_api_calls = found_api_calls
        
        if found_api_calls:
            print(f"✅ Found API calls in {len(found_api_calls)} files")
        else:
            self.warnings.append("No external API calls found - this might indicate incomplete implementation")
    
    def validate_mock_coverage(self):
        """Validate that all found API calls have corresponding mocks."""
        print("Checking mock coverage...")
        
        # Load mock implementations
        mock_implementations = self._load_mock_implementations()
        
        uncovered_apis = []
        
        for file_path, api_calls in self.found_api_calls.items():
            for api_call in api_calls:
                api_name = api_call["api"]
                method_name = api_call["method"]
                
                if api_name not in mock_implementations:
                    uncovered_apis.append(f"{api_name} (used in {file_path})")
                elif method_name not in mock_implementations[api_name]:
                    uncovered_apis.append(f"{api_name}.{method_name} (used in {file_path})")
        
        if uncovered_apis:
            self.errors.extend([f"Missing mock for: {api}" for api in uncovered_apis])
        else:
            print("✅ All API calls have mock coverage")
    
    def validate_mock_implementations(self):
        """Validate that mock implementations are complete and realistic."""
        print("Checking mock implementations...")
        
        try:
            # Test that mock classes can be imported
            from tests.mocks.api_mocks import GeminiAPIMock, ImagenAPIMock, WhisperAPIMock, ComprehensiveAPIMocker
            
            # Test Gemini mock
            gemini_mock = GeminiAPIMock()
            required_methods = ["analyze_content", "research_keywords", "analyze_competitors", "predict_engagement"]
            
            for method in required_methods:
                if not hasattr(gemini_mock, method):
                    self.errors.append(f"GeminiAPIMock missing method: {method}")
                elif not callable(getattr(gemini_mock, method)):
                    self.errors.append(f"GeminiAPIMock.{method} is not callable")
            
            # Test Imagen mock
            imagen_mock = ImagenAPIMock()
            required_methods = ["generate_background", "generate_variations"]
            
            for method in required_methods:
                if not hasattr(imagen_mock, method):
                    self.errors.append(f"ImagenAPIMock missing method: {method}")
            
            # Test Whisper mock
            whisper_mock = WhisperAPIMock()
            required_methods = ["transcribe", "transcribe_with_timestamps", "identify_speakers"]
            
            for method in required_methods:
                if not hasattr(whisper_mock, method):
                    self.errors.append(f"WhisperAPIMock missing method: {method}")
            
            # Test comprehensive mocker
            comprehensive_mocker = ComprehensiveAPIMocker()
            required_methods = ["mock_all_apis", "mock_gemini_api", "mock_imagen_api", "mock_whisper_api"]
            
            for method in required_methods:
                if not hasattr(comprehensive_mocker, method):
                    self.errors.append(f"ComprehensiveAPIMocker missing method: {method}")
            
            print("✅ Mock implementations validated")
            
        except ImportError as e:
            self.errors.append(f"Failed to import mock classes: {str(e)}")
        except Exception as e:
            self.errors.append(f"Mock implementation validation failed: {str(e)}")
    
    def validate_test_isolation(self):
        """Validate that tests properly isolate external dependencies."""
        print("Checking test isolation...")
        
        test_files = list((self.project_root / "tests").rglob("test_*.py"))
        
        tests_with_mocking = 0
        tests_without_mocking = []
        
        for test_file in test_files:
            if test_file.name in ["__init__.py", "conftest.py"]:
                continue
            
            try:
                content = test_file.read_text()
                
                # Check for mocking patterns
                has_mocking = any(pattern in content for pattern in [
                    "@patch", "mock.patch", "api_mocker", "mock_", 
                    "ComprehensiveAPIMocker", "GeminiAPIMock", "ImagenAPIMock", "WhisperAPIMock"
                ])
                
                # Check for external API usage
                has_external_calls = any(api in content for api in self.external_apis.keys())
                
                if has_external_calls and not has_mocking:
                    tests_without_mocking.append(str(test_file.relative_to(self.project_root)))
                elif has_mocking:
                    tests_with_mocking += 1
                    
            except Exception as e:
                self.warnings.append(f"Failed to analyze test file {test_file}: {str(e)}")
        
        if tests_without_mocking:
            self.warnings.extend([f"Test file may need mocking: {test}" for test in tests_without_mocking])
        
        print(f"✅ Found {tests_with_mocking} tests with proper mocking")
    
    def _extract_api_calls_from_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Extract external API calls from a Python file."""
        try:
            content = file_path.read_text()
            tree = ast.parse(content)
            
            api_calls = []
            
            class APICallVisitor(ast.NodeVisitor):
                def visit_Call(self, node):
                    # Look for function calls that might be external APIs
                    if isinstance(node.func, ast.Attribute):
                        # Handle module.function() calls
                        if isinstance(node.func.value, ast.Name):
                            module_name = node.func.value.id
                            function_name = node.func.attr
                            
                            # Check if this looks like an external API call
                            for api_name, methods in self.external_apis.items():
                                if (api_name in module_name.lower() or 
                                    module_name.lower() in api_name or
                                    function_name in methods):
                                    api_calls.append({
                                        "api": api_name,
                                        "method": function_name,
                                        "module": module_name,
                                        "line": node.lineno
                                    })
                    
                    elif isinstance(node.func, ast.Name):
                        # Handle direct function calls
                        function_name = node.func.id
                        
                        for api_name, methods in self.external_apis.items():
                            if function_name in methods:
                                api_calls.append({
                                    "api": api_name,
                                    "method": function_name,
                                    "module": "direct",
                                    "line": node.lineno
                                })
                    
                    self.generic_visit(node)
            
            visitor = APICallVisitor()
            visitor.visit(tree)
            
            return api_calls
            
        except Exception as e:
            raise Exception(f"Failed to parse {file_path}: {str(e)}")
    
    def _load_mock_implementations(self) -> Dict[str, List[str]]:
        """Load available mock implementations."""
        mock_implementations = {}
        
        try:
            # Scan mock files for available implementations
            mock_files = list((self.project_root / "tests" / "mocks").rglob("*.py"))
            
            for mock_file in mock_files:
                if mock_file.name == "__init__.py":
                    continue
                
                try:
                    content = mock_file.read_text()
                    tree = ast.parse(content)
                    
                    class MockMethodVisitor(ast.NodeVisitor):
                        def __init__(self):
                            self.current_class = None
                            self.methods = {}
                        
                        def visit_ClassDef(self, node):
                            self.current_class = node.name
                            self.methods[self.current_class] = []
                            self.generic_visit(node)
                            self.current_class = None
                        
                        def visit_FunctionDef(self, node):
                            if self.current_class and not node.name.startswith('_'):
                                self.methods[self.current_class].append(node.name)
                    
                    visitor = MockMethodVisitor()
                    visitor.visit(tree)
                    
                    # Map mock classes to API names
                    for class_name, methods in visitor.methods.items():
                        if "gemini" in class_name.lower():
                            mock_implementations["gemini_api"] = methods
                        elif "imagen" in class_name.lower():
                            mock_implementations["imagen_api"] = methods
                        elif "whisper" in class_name.lower():
                            mock_implementations["whisper_api"] = methods
                
                except Exception as e:
                    self.warnings.append(f"Failed to analyze mock file {mock_file}: {str(e)}")
        
        except Exception as e:
            self.warnings.append(f"Failed to load mock implementations: {str(e)}")
        
        return mock_implementations
    
    def report_results(self) -> bool:
        """Report validation results."""
        print("\n" + "="*60)
        print("API MOCK VALIDATION RESULTS")
        print("="*60)
        
        if self.found_api_calls:
            print(f"\n📊 FOUND API CALLS:")
            for file_path, api_calls in self.found_api_calls.items():
                print(f"  {file_path}:")
                for call in api_calls:
                    print(f"    - {call['api']}.{call['method']} (line {call['line']})")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ All API mock validations passed!")
        elif not self.errors:
            print(f"\n✅ API mock validation passed with {len(self.warnings)} warnings")
        
        print("="*60)
        
        return len(self.errors) == 0


def main():
    """Main validation function for pre-commit hook."""
    validator = APIMockValidator()
    success = validator.validate_all()
    
    if not success:
        print("\n💥 API mock validation failed!")
        print("Please fix the errors above before committing.")
        sys.exit(1)
    else:
        print("\n🎉 API mock validation passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()