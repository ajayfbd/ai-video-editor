#!/usr/bin/env python3
"""
Fix Validation Issues Script

This script fixes the specific issues found during validation testing.
"""

import re
from pathlib import Path

def fix_cli_unicode_issue():
    """Fix Unicode encoding issue in CLI status command."""
    print("🔧 Fixing CLI Unicode Issue...")
    
    cli_file = Path("ai_video_editor/cli/main.py")
    if not cli_file.exists():
        print("   ⚠️  CLI file not found")
        return
    
    content = cli_file.read_text(encoding='utf-8')
    
    # Replace Unicode checkmarks with ASCII equivalents
    replacements = [
        ('✅', '[OK]'),
        ('❌', '[ERROR]'),
        ('⚠️', '[WARNING]'),
        ('📊', '[INFO]'),
        ('📁', '[DIR]'),
        ('💾', '[MEM]'),
        ('🔗', '[API]'),
        ('⏱️', '[TIME]'),
        ('⏹️', '[STOP]'),
        ('🎬', '[VIDEO]'),
        ('🎵', '[AUDIO]'),
        ('📋', '[LIST]'),
        ('🧪', '[TEST]'),
    ]
    
    original_content = content
    for unicode_char, ascii_replacement in replacements:
        content = content.replace(unicode_char, ascii_replacement)
    
    if content != original_content:
        cli_file.write_text(content, encoding='utf-8')
        print("   ✅ Fixed Unicode characters in CLI")
    else:
        print("   ℹ️  No Unicode characters found to fix")

def fix_documentation_imports():
    """Fix import statements in documentation."""
    print("🔧 Fixing Documentation Imports...")
    
    # Fix the test code to use correct import
    test_file = Path("test_code_examples.py")
    if test_file.exists():
        content = test_file.read_text(encoding='utf-8')
        
        # Fix the ContentContext test
        old_import = "from ai_video_editor.core.config import UserPreferences"
        new_import = "from ai_video_editor.core.content_context import UserPreferences"
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            test_file.write_text(content, encoding='utf-8')
            print("   ✅ Fixed UserPreferences import in test file")
    
    # Fix documentation files
    doc_files = [
        "docs/developer/api-reference.md",
        "docs/user-guide/README.md",
        "quick-start.md"
    ]
    
    for doc_file_path in doc_files:
        doc_file = Path(doc_file_path)
        if doc_file.exists():
            content = doc_file.read_text(encoding='utf-8')
            original_content = content
            
            # Fix UserPreferences import
            content = re.sub(
                r'from ai_video_editor\.core\.config import ([^,\n]*,\s*)?UserPreferences',
                r'from ai_video_editor.core.content_context import UserPreferences',
                content
            )
            
            # Fix combined imports
            content = re.sub(
                r'from ai_video_editor\.core\.config import ([^,\n]+), UserPreferences',
                r'from ai_video_editor.core.config import \1\nfrom ai_video_editor.core.content_context import UserPreferences',
                content
            )
            
            if content != original_content:
                doc_file.write_text(content, encoding='utf-8')
                print(f"   ✅ Fixed imports in {doc_file_path}")

def update_api_examples():
    """Update API examples to match current implementation."""
    print("🔧 Updating API Examples...")
    
    # Update the test file with correct example
    test_file = Path("test_code_examples.py")
    if test_file.exists():
        content = test_file.read_text(encoding='utf-8')
        
        # Update ContentContext creation example
        old_example = '''from ai_video_editor.core.content_context import ContentContext, ContentType
from ai_video_editor.core.config import UserPreferences
context = ContentContext(
    project_id="test_project",
    video_files=["test.mp4"],
    content_type=ContentType.GENERAL,
    user_preferences=UserPreferences()
)'''
        
        new_example = '''from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
context = ContentContext(
    project_id="test_project",
    video_files=["test.mp4"],
    content_type=ContentType.GENERAL,
    user_preferences=UserPreferences()
)'''
        
        if old_example in content:
            content = content.replace(old_example, new_example)
            test_file.write_text(content, encoding='utf-8')
            print("   ✅ Updated ContentContext example in test file")

def add_missing_classes():
    """Add any missing classes that are referenced in documentation."""
    print("🔧 Checking for Missing Classes...")
    
    # Check if ProcessingMode enum exists
    workflow_file = Path("ai_video_editor/core/workflow_orchestrator.py")
    if workflow_file.exists():
        content = workflow_file.read_text(encoding='utf-8')
        
        if "class ProcessingMode" not in content and "ProcessingMode" not in content:
            # Add ProcessingMode enum
            enum_code = '''
from enum import Enum

class ProcessingMode(Enum):
    """Processing mode enumeration."""
    FAST = "fast"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"

'''
            # Insert after imports
            import_end = content.find('\n\n')
            if import_end != -1:
                content = content[:import_end] + enum_code + content[import_end:]
                workflow_file.write_text(content, encoding='utf-8')
                print("   ✅ Added ProcessingMode enum to workflow_orchestrator.py")

def create_env_example():
    """Create .env.example file if it doesn't exist."""
    print("🔧 Creating .env.example...")
    
    env_example = Path(".env.example")
    if not env_example.exists():
        env_content = """# AI Video Editor Configuration

# Required API Keys
AI_VIDEO_EDITOR_GEMINI_API_KEY=your_gemini_api_key_here
AI_VIDEO_EDITOR_IMAGEN_API_KEY=your_imagen_api_key_here
AI_VIDEO_EDITOR_GOOGLE_CLOUD_PROJECT=your_project_id_here

# Optional Settings
AI_VIDEO_EDITOR_DEBUG=false
AI_VIDEO_EDITOR_LOG_LEVEL=INFO
AI_VIDEO_EDITOR_MAX_MEMORY_USAGE_GB=8.0
AI_VIDEO_EDITOR_MAX_CONCURRENT_PROCESSES=2
AI_VIDEO_EDITOR_ENABLE_GPU_ACCELERATION=true

# Default Project Settings
AI_VIDEO_EDITOR_DEFAULT_CONTENT_TYPE=general
AI_VIDEO_EDITOR_DEFAULT_QUALITY=high
AI_VIDEO_EDITOR_DEFAULT_OUTPUT_FORMAT=mp4

# Performance Settings
AI_VIDEO_EDITOR_WHISPER_MODEL_SIZE=large-v3
AI_VIDEO_EDITOR_VIDEO_PROCESSING_TIMEOUT=3600
AI_VIDEO_EDITOR_API_REQUEST_TIMEOUT=30
AI_VIDEO_EDITOR_MAX_RETRIES=3
"""
        env_example.write_text(env_content, encoding='utf-8')
        print("   ✅ Created .env.example file")
    else:
        print("   ℹ️  .env.example already exists")

def main():
    """Main function to apply all fixes."""
    print("🔧 Applying Validation Fixes...")
    print("=" * 40)
    
    fix_cli_unicode_issue()
    fix_documentation_imports()
    update_api_examples()
    add_missing_classes()
    create_env_example()
    
    print("\n✅ All validation fixes applied!")
    print("\nNext steps:")
    print("1. Run the test suite again: python test_code_examples.py")
    print("2. Verify all examples work correctly")
    print("3. Update any remaining documentation issues")

if __name__ == "__main__":
    main()