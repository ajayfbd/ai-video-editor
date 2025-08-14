#!/usr/bin/env python3
"""
Configuration Validation Script for AI Video Editor

This script validates the project configuration system to ensure:
- All required configuration files exist
- Environment variables are properly set
- Dependencies are correctly specified in pyproject.toml
- No security issues (API keys in templates)

Usage:
    python tools/scripts/validate_config.py
    python tools/scripts/validate_config.py --environment development
    python tools/scripts/validate_config.py --fix-issues
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import tomllib
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class ConfigValidator:
    """Validates AI Video Editor configuration system."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues: List[str] = []
        self.warnings: List[str] = []
        
    def validate_all(self, environment: Optional[str] = None, fix_issues: bool = False) -> bool:
        """Run all configuration validations."""
        print("🔍 AI Video Editor Configuration Validation")
        print("=" * 50)
        
        # Core validation checks
        self.validate_pyproject_toml()
        self.validate_environment_files()
        self.validate_security()
        self.validate_directory_structure()
        
        if environment:
            self.validate_environment_specific(environment)
            
        # Report results
        self.report_results(fix_issues)
        
        return len(self.issues) == 0
    
    def validate_pyproject_toml(self):
        """Validate pyproject.toml configuration."""
        print("\n📋 Validating pyproject.toml...")
        
        pyproject_path = self.project_root / "pyproject.toml"
        if not pyproject_path.exists():
            self.issues.append("Missing pyproject.toml file")
            return
            
        try:
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)
        except Exception as e:
            self.issues.append(f"Invalid pyproject.toml syntax: {e}")
            return
            
        # Check required sections
        required_sections = ["build-system", "project", "tool.pytest.ini_options"]
        for section in required_sections:
            if not self._get_nested_key(config, section.split(".")):
                self.issues.append(f"Missing required section: {section}")
                
        # Check project metadata
        project = config.get("project", {})
        required_fields = ["name", "version", "description", "dependencies"]
        for field in required_fields:
            if field not in project:
                self.issues.append(f"Missing project.{field} in pyproject.toml")
                
        # Check CLI entry points
        scripts = project.get("scripts", {})
        expected_scripts = ["video-editor", "ai-ve"]
        for script in expected_scripts:
            if script not in scripts:
                self.warnings.append(f"Missing CLI entry point: {script}")
                
        # Validate dependencies
        dependencies = project.get("dependencies", [])
        required_deps = ["movis", "openai-whisper", "opencv-python", "google-genai"]
        for dep in required_deps:
            if not any(dep in d for d in dependencies):
                self.issues.append(f"Missing required dependency: {dep}")
                
        print("✅ pyproject.toml validation complete")
    
    def validate_environment_files(self):
        """Validate environment configuration files."""
        print("\n🌍 Validating environment files...")
        
        # Check config directory structure
        config_dir = self.project_root / "config"
        if not config_dir.exists():
            self.issues.append("Missing config/ directory")
            return
            
        # Check required template files
        required_templates = [
            "config/.env.example",
            "config/development.env", 
            "config/testing.env"
        ]
        
        for template in required_templates:
            template_path = self.project_root / template
            if not template_path.exists():
                self.issues.append(f"Missing template: {template}")
            else:
                self._validate_env_file(template_path, is_template=True)
                
        # Check root .env if it exists
        root_env = self.project_root / ".env"
        if root_env.exists():
            self._validate_env_file(root_env, is_template=False)
        else:
            self.warnings.append("No .env file found (copy from config/ templates)")
            
        print("✅ Environment files validation complete")
    
    def _validate_env_file(self, env_path: Path, is_template: bool):
        """Validate individual environment file."""
        try:
            with open(env_path, 'r') as f:
                content = f.read()
        except Exception as e:
            self.issues.append(f"Cannot read {env_path}: {e}")
            return
            
        # Check for required variables
        required_vars = [
            "AI_VIDEO_EDITOR_GEMINI_API_KEY",
            "AI_VIDEO_EDITOR_DEBUG",
            "AI_VIDEO_EDITOR_LOG_LEVEL",
            "AI_VIDEO_EDITOR_MAX_MEMORY_USAGE_GB",
            "AI_VIDEO_EDITOR_WHISPER_MODEL_SIZE"
        ]
        
        for var in required_vars:
            if var not in content:
                self.issues.append(f"Missing variable {var} in {env_path}")
                
        # Security check: templates should not have real API keys
        if is_template:
            # Look for actual Google API keys (start with AIza and are 39 chars)
            google_api_pattern = r'AI_VIDEO_EDITOR_\w*API_KEY\s*=\s*["\']?AIza[A-Za-z0-9_-]{35}["\']?'
            if re.search(google_api_pattern, content):
                self.issues.append(f"Template {env_path} contains real Google API key!")
    
    def validate_security(self):
        """Validate security configuration."""
        print("\n🔒 Validating security configuration...")
        
        # Check .gitignore
        gitignore_path = self.project_root / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()
                
            if ".env" not in gitignore_content:
                self.issues.append(".env not found in .gitignore")
        else:
            self.issues.append("Missing .gitignore file")
            
        # Check for accidentally committed API keys
        env_files = [
            self.project_root / ".env",
            self.project_root / "config" / ".env.example",
            self.project_root / "config" / "development.env",
            self.project_root / "config" / "testing.env"
        ]
        
        for env_file in env_files:
            if env_file.exists():
                self._check_for_real_api_keys(env_file)
                
        print("✅ Security validation complete")
    
    def _check_for_real_api_keys(self, env_path: Path):
        """Check if file contains real API keys."""
        try:
            with open(env_path, 'r') as f:
                content = f.read()
        except:
            return
            
        # Pattern for real API keys (not placeholders)
        real_key_patterns = [
            r'GEMINI_API_KEY\s*=\s*["\']?AIza[A-Za-z0-9_-]{35}["\']?',  # Google API key pattern
        ]
        
        for pattern in real_key_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # Skip obvious placeholders
                if not any(placeholder in match.lower() for placeholder in 
                          ["your_", "test_", "mock", "example", "placeholder", "here"]):
                    self.issues.append(f"Potential real API key found in {env_path}: {match}")
    
    def validate_directory_structure(self):
        """Validate project directory structure."""
        print("\n📁 Validating directory structure...")
        
        # Check required directories
        required_dirs = [
            "ai_video_editor",
            "tests", 
            "config",
            "docs",
            "workspace",
            "tools"
        ]
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                self.issues.append(f"Missing required directory: {dir_name}")
                
        # Check for old redundant files
        redundant_files = [
            "requirements.txt",  # Should use pyproject.toml
            "setup.py",         # Should use pyproject.toml
            "pytest.ini"        # Should use pyproject.toml
        ]
        
        for file_name in redundant_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                self.warnings.append(f"Redundant file found: {file_name} (use pyproject.toml)")
                
        print("✅ Directory structure validation complete")
    
    def validate_environment_specific(self, environment: str):
        """Validate specific environment configuration."""
        print(f"\n🎯 Validating {environment} environment...")
        
        env_file = self.project_root / "config" / f"{environment}.env"
        if not env_file.exists():
            self.issues.append(f"Missing {environment}.env template")
            return
            
        # Environment-specific validation
        if environment == "testing":
            self._validate_testing_config(env_file)
        elif environment == "development":
            self._validate_development_config(env_file)
            
        print(f"✅ {environment} environment validation complete")
    
    def _validate_testing_config(self, env_path: Path):
        """Validate testing environment configuration."""
        with open(env_path, 'r') as f:
            content = f.read()
            
        # Testing should use minimal resources
        if "MAX_MEMORY_USAGE_GB=2.0" not in content:
            self.warnings.append("Testing environment should use minimal memory (2.0GB)")
            
        if "WHISPER_MODEL_SIZE=tiny" not in content:
            self.warnings.append("Testing environment should use tiny Whisper model")
    
    def _validate_development_config(self, env_path: Path):
        """Validate development environment configuration."""
        with open(env_path, 'r') as f:
            content = f.read()
            
        # Development should have debug enabled
        if "DEBUG=true" not in content:
            self.warnings.append("Development environment should have DEBUG=true")
    
    def _get_nested_key(self, data: Dict[str, Any], keys: List[str]) -> Any:
        """Get nested dictionary key."""
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                return None
        return data
    
    def report_results(self, fix_issues: bool = False):
        """Report validation results."""
        print("\n" + "=" * 50)
        print("📊 VALIDATION RESULTS")
        print("=" * 50)
        
        if not self.issues and not self.warnings:
            print("✅ All configuration checks passed!")
            print("🎉 AI Video Editor configuration is properly set up.")
            return
            
        if self.issues:
            print(f"❌ {len(self.issues)} ISSUES FOUND:")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")
                
        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
                
        if fix_issues:
            print("\n🔧 Attempting to fix issues...")
            self._fix_common_issues()
        else:
            print("\n💡 Run with --fix-issues to attempt automatic fixes")
            
    def _fix_common_issues(self):
        """Attempt to fix common configuration issues."""
        fixed_count = 0
        
        # Create missing .env from template
        if not (self.project_root / ".env").exists():
            template_path = self.project_root / "config" / ".env.example"
            if template_path.exists():
                import shutil
                shutil.copy(template_path, self.project_root / ".env")
                print("✅ Created .env from template")
                fixed_count += 1
                
        # Remove redundant requirements.txt
        requirements_txt = self.project_root / "requirements.txt"
        if requirements_txt.exists():
            requirements_txt.unlink()
            print("✅ Removed redundant requirements.txt")
            fixed_count += 1
            
        print(f"\n🎯 Fixed {fixed_count} issues automatically")
        if fixed_count > 0:
            print("🔄 Re-run validation to check remaining issues")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate AI Video Editor configuration")
    parser.add_argument("--environment", choices=["development", "testing", "production"],
                       help="Validate specific environment")
    parser.add_argument("--fix-issues", action="store_true",
                       help="Attempt to fix common issues automatically")
    
    args = parser.parse_args()
    
    # Find project root
    current_dir = Path(__file__).parent
    while current_dir.parent != current_dir:
        if (current_dir / "pyproject.toml").exists():
            project_root = current_dir
            break
        current_dir = current_dir.parent
    else:
        print("❌ Could not find project root (no pyproject.toml found)")
        sys.exit(1)
    
    # Run validation
    validator = ConfigValidator(project_root)
    success = validator.validate_all(args.environment, args.fix_issues)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()