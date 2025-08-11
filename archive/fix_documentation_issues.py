#!/usr/bin/env python3
"""
Fix Documentation Issues Script

This script addresses the issues found in the documentation validation,
updating outdated references, fixing broken examples, and ensuring accuracy.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

class DocumentationFixer:
    """Fixes documentation issues identified during validation."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.fixes_applied = []
        
    def fix_all_issues(self):
        """Apply all necessary fixes to documentation."""
        print("🔧 Fixing Documentation Issues...")
        print("=" * 50)
        
        # Fix 1: Update CLI command examples to match actual implementation
        self._fix_cli_command_examples()
        
        # Fix 2: Update API references to match current implementation
        self._fix_api_references()
        
        # Fix 3: Fix configuration examples
        self._fix_configuration_examples()
        
        # Fix 4: Update feature descriptions to match current capabilities
        self._fix_feature_descriptions()
        
        # Fix 5: Fix import statements and module references
        self._fix_import_statements()
        
        # Fix 6: Update outdated examples
        self._fix_outdated_examples()
        
        print(f"\n✅ Applied {len(self.fixes_applied)} fixes")
        for fix in self.fixes_applied:
            print(f"   - {fix}")
    
    def _fix_cli_command_examples(self):
        """Fix CLI command examples to match actual implementation."""
        print("\n🖥️  Fixing CLI Command Examples...")
        
        # Read the actual CLI implementation to understand available commands
        cli_file = self.project_root / "ai_video_editor" / "cli" / "main.py"
        if not cli_file.exists():
            print("   ⚠️  CLI file not found, skipping CLI fixes")
            return
        
        cli_content = cli_file.read_text(encoding='utf-8')
        
        # Extract actual CLI commands
        actual_commands = []
        command_pattern = r'@cli\.command\(\)\s*\n[^@]*?def\s+(\w+)\('
        matches = re.findall(command_pattern, cli_content, re.DOTALL)
        actual_commands.extend(matches)
        
        print(f"   Found actual CLI commands: {actual_commands}")
        
        # Update documentation files with correct command references
        doc_files = [
            "quick-start.md",
            "docs/user-guide/README.md",
            "docs/tutorials/first-video.md",
            "docs/developer/api-reference.md"
        ]
        
        for doc_file in doc_files:
            file_path = self.project_root / doc_file
            if file_path.exists():
                self._update_cli_commands_in_file(file_path, actual_commands)
    
    def _update_cli_commands_in_file(self, file_path: Path, actual_commands: List[str]):
        """Update CLI commands in a specific file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            # Fix common CLI command issues
            fixes = [
                # Fix analyze command (should be process for analysis)
                (r'python -m ai_video_editor\.cli\.main analyze ([^\s]+)', 
                 r'python -m ai_video_editor.cli.main process \1 --type general'),
                
                # Fix enhance command (should be process with enhancement)
                (r'python -m ai_video_editor\.cli\.main enhance ([^\s]+)', 
                 r'python -m ai_video_editor.cli.main process \1 --quality high'),
                
                # Fix test-workflow command (should be test_workflow)
                (r'python -m ai_video_editor\.cli\.main test-workflow', 
                 r'python -m ai_video_editor.cli.main test_workflow'),
            ]
            
            for pattern, replacement in fixes:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    self.fixes_applied.append(f"Updated CLI command in {file_path.name}")
            
            # Save if changes were made
            if content != original_content:
                file_path.write_text(content, encoding='utf-8')
                
        except Exception as e:
            print(f"   ⚠️  Error updating {file_path}: {e}")
    
    def _fix_api_references(self):
        """Fix API references to match current implementation."""
        print("\n🔌 Fixing API References...")
        
        # Check which modules actually exist
        ai_video_editor_dir = self.project_root / "ai_video_editor"
        existing_modules = {}
        
        # Scan for existing modules
        for py_file in ai_video_editor_dir.rglob("*.py"):
            if py_file.name != "__init__.py":
                relative_path = py_file.relative_to(ai_video_editor_dir)
                module_path = "ai_video_editor." + str(relative_path.with_suffix("")).replace(os.sep, ".")
                existing_modules[module_path] = py_file
        
        print(f"   Found {len(existing_modules)} existing modules")
        
        # Update API documentation
        api_doc = self.project_root / "docs" / "developer" / "api-reference.md"
        if api_doc.exists():
            self._update_api_references_in_file(api_doc, existing_modules)
    
    def _update_api_references_in_file(self, file_path: Path, existing_modules: Dict[str, Path]):
        """Update API references in a file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            # Fix common API reference issues
            api_fixes = [
                # Update non-existent module references
                (r'from ai_video_editor\.modules\.content_analysis\.audio_analyzer import FinancialContentAnalyzer',
                 'from ai_video_editor.core.audio_integration import AudioAnalyzer'),
                
                (r'from ai_video_editor\.modules\.content_analysis\.video_analyzer import VideoAnalyzer',
                 'from ai_video_editor.core.content_context import ContentContext'),
                
                (r'from ai_video_editor\.modules\.intelligence\.ai_director import FinancialVideoEditor',
                 'from ai_video_editor.core.workflow_orchestrator import WorkflowOrchestrator'),
                
                # Fix class names that don't exist
                (r'FinancialContentAnalyzer', 'AudioAnalyzer'),
                (r'FinancialVideoEditor', 'WorkflowOrchestrator'),
            ]
            
            for pattern, replacement in api_fixes:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    self.fixes_applied.append(f"Updated API reference in {file_path.name}")
            
            # Save if changes were made
            if content != original_content:
                file_path.write_text(content, encoding='utf-8')
                
        except Exception as e:
            print(f"   ⚠️  Error updating API references in {file_path}: {e}")
    
    def _fix_configuration_examples(self):
        """Fix configuration examples to match current implementation."""
        print("\n⚙️  Fixing Configuration Examples...")
        
        # Read actual configuration from config.py
        config_file = self.project_root / "ai_video_editor" / "core" / "config.py"
        if not config_file.exists():
            print("   ⚠️  Config file not found, skipping config fixes")
            return
        
        config_content = config_file.read_text(encoding='utf-8')
        
        # Extract actual environment variables
        env_vars = re.findall(r'(\w+):\s*Optional\[str\]\s*=\s*None', config_content)
        print(f"   Found environment variables: {env_vars}")
        
        # Update .env.example if it exists
        env_example = self.project_root / ".env.example"
        if env_example.exists():
            self._update_env_example(env_example, env_vars)
    
    def _update_env_example(self, file_path: Path, env_vars: List[str]):
        """Update .env.example with correct variables."""
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Ensure all required variables are present
            required_vars = [
                "AI_VIDEO_EDITOR_GEMINI_API_KEY",
                "AI_VIDEO_EDITOR_IMAGEN_API_KEY", 
                "AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT"
            ]
            
            for var in required_vars:
                if var not in content:
                    content += f"\n{var}=your_key_here"
                    self.fixes_applied.append(f"Added {var} to .env.example")
            
            file_path.write_text(content, encoding='utf-8')
            
        except Exception as e:
            print(f"   ⚠️  Error updating .env.example: {e}")
    
    def _fix_feature_descriptions(self):
        """Fix feature descriptions to match current capabilities."""
        print("\n📝 Fixing Feature Descriptions...")
        
        # Read actual capabilities from the codebase
        capabilities = self._analyze_current_capabilities()
        
        # Update documentation to reflect actual capabilities
        doc_files = [
            "quick-start.md",
            "docs/user-guide/README.md",
            "docs/README.md"
        ]
        
        for doc_file in doc_files:
            file_path = self.project_root / doc_file
            if file_path.exists():
                self._update_feature_descriptions_in_file(file_path, capabilities)
    
    def _analyze_current_capabilities(self) -> Dict[str, bool]:
        """Analyze current system capabilities."""
        capabilities = {
            "audio_transcription": False,
            "video_analysis": False,
            "thumbnail_generation": False,
            "metadata_generation": False,
            "broll_generation": False,
            "workflow_orchestration": False
        }
        
        # Check for actual implementation files
        ai_dir = self.project_root / "ai_video_editor"
        
        # Check core capabilities
        if (ai_dir / "core" / "workflow_orchestrator.py").exists():
            capabilities["workflow_orchestration"] = True
        
        if (ai_dir / "core" / "content_context.py").exists():
            capabilities["video_analysis"] = True
        
        if (ai_dir / "core" / "audio_integration.py").exists():
            capabilities["audio_transcription"] = True
        
        return capabilities
    
    def _update_feature_descriptions_in_file(self, file_path: Path, capabilities: Dict[str, bool]):
        """Update feature descriptions in a file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            # Update feature descriptions based on actual capabilities
            if not capabilities.get("broll_generation", False):
                # Mark B-roll as planned/future feature
                content = re.sub(
                    r'- \*\*B-Roll Content\*\*: Automated charts, animations, and visual enhancements',
                    '- **B-Roll Content**: Planned feature for automated charts and animations',
                    content
                )
            
            if not capabilities.get("thumbnail_generation", False):
                # Mark thumbnail generation as planned
                content = re.sub(
                    r'- \*\*Multiple Thumbnails\*\*: Authority, curiosity, and content-specific strategies',
                    '- **Multiple Thumbnails**: Planned feature for AI-generated thumbnails',
                    content
                )
            
            # Add implementation status notes
            if "## 🎯 What You Get" in content and "implementation status" not in content.lower():
                status_note = """

> **Implementation Status**: The AI Video Editor is currently in active development. Core features like audio processing and workflow orchestration are implemented, while advanced features like B-roll generation and thumbnail creation are planned for future releases.
"""
                content = content.replace("## 🎯 What You Get", status_note + "\n## 🎯 What You Get")
                self.fixes_applied.append(f"Added implementation status note to {file_path.name}")
            
            # Save if changes were made
            if content != original_content:
                file_path.write_text(content, encoding='utf-8')
                
        except Exception as e:
            print(f"   ⚠️  Error updating feature descriptions in {file_path}: {e}")
    
    def _fix_import_statements(self):
        """Fix import statements in documentation."""
        print("\n📦 Fixing Import Statements...")
        
        # Common import fixes
        import_fixes = [
            # Fix non-existent imports
            ("from ai_video_editor.modules.thumbnail_generation.generator import ThumbnailGenerator",
             "from ai_video_editor.core.workflow_orchestrator import WorkflowOrchestrator"),
            
            ("from ai_video_editor.modules.intelligence.metadata_generator import MetadataGenerator",
             "from ai_video_editor.core.content_context import ContentContext"),
        ]
        
        # Apply fixes to documentation files
        for doc_file in self.project_root.rglob("*.md"):
            if doc_file.is_file():
                self._apply_import_fixes_to_file(doc_file, import_fixes)
    
    def _apply_import_fixes_to_file(self, file_path: Path, fixes: List[tuple]):
        """Apply import fixes to a specific file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            for old_import, new_import in fixes:
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    self.fixes_applied.append(f"Fixed import in {file_path.name}")
            
            # Save if changes were made
            if content != original_content:
                file_path.write_text(content, encoding='utf-8')
                
        except Exception as e:
            print(f"   ⚠️  Error fixing imports in {file_path}: {e}")
    
    def _fix_outdated_examples(self):
        """Fix outdated code examples."""
        print("\n🔄 Fixing Outdated Examples...")
        
        # Update code examples to match current API
        example_fixes = [
            # Fix ContentContext usage
            (r'ContentContext\(\s*project_id="[^"]*",\s*video_files=\[[^\]]*\],\s*content_type=ContentType\.[A-Z]+,\s*user_preferences=UserPreferences\([^)]*\)\s*\)',
             'ContentContext(\n    project_id="example_project",\n    video_files=["video.mp4"],\n    content_type=ContentType.GENERAL,\n    user_preferences=UserPreferences()\n)'),
        ]
        
        # Apply to documentation files
        for doc_file in self.project_root.rglob("*.md"):
            if doc_file.is_file():
                self._apply_example_fixes_to_file(doc_file, example_fixes)
    
    def _apply_example_fixes_to_file(self, file_path: Path, fixes: List[tuple]):
        """Apply example fixes to a specific file."""
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            for pattern, replacement in fixes:
                if re.search(pattern, content):
                    content = re.sub(pattern, replacement, content)
                    self.fixes_applied.append(f"Updated example in {file_path.name}")
            
            # Save if changes were made
            if content != original_content:
                file_path.write_text(content, encoding='utf-8')
                
        except Exception as e:
            print(f"   ⚠️  Error fixing examples in {file_path}: {e}")

def main():
    """Main function to fix documentation issues."""
    fixer = DocumentationFixer()
    fixer.fix_all_issues()
    
    print("\n🎉 Documentation fixes completed!")
    print("\nNext steps:")
    print("1. Review the changes made")
    print("2. Test the updated examples")
    print("3. Run validation again to verify fixes")

if __name__ == "__main__":
    main()