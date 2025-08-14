#!/usr/bin/env python3
"""
Fix broken internal links in documentation
"""

import re
from pathlib import Path

def fix_broken_links():
    """Fix common broken link patterns"""
    
    # Define link fixes
    link_fixes = {
        # Fix anchor links that exist but are referenced incorrectly
        "docs/tutorials/README.md#complete-workflow-guides": "docs/tutorials/README.md#-complete-workflow-guides",
        "docs/user-guide/README.md#configuration": "docs/user-guide/README.md#-configuration",
        "docs/user-guide/README.md#content-types": "docs/user-guide/README.md#-content-types",
        
        # Fix relative path issues
        "user-guide/README.md": "../user-guide/README.md",
        "tutorials/README.md": "../tutorials/README.md",
        "developer/README.md": "../developer/README.md",
        "support/troubleshooting-unified.md": "../support/troubleshooting-unified.md",
        "support/faq-unified.md": "../support/faq-unified.md",
        "support/performance-unified.md": "../support/performance-unified.md",
    }
    
    # Process all markdown files
    project_root = Path(".")
    fixed_count = 0
    
    for md_file in project_root.glob("**/*.md"):
        if "archive" in str(md_file) or ".git" in str(md_file) or ".agent.md" in str(md_file):
            continue
            
        try:
            content = md_file.read_text(encoding='utf-8')
            original_content = content
            
            # Apply fixes
            for broken_link, fixed_link in link_fixes.items():
                if broken_link in content:
                    content = content.replace(broken_link, fixed_link)
                    print(f"Fixed link in {md_file}: {broken_link} -> {fixed_link}")
                    fixed_count += 1
            
            # Write back if changed
            if content != original_content:
                md_file.write_text(content, encoding='utf-8')
                
        except Exception as e:
            print(f"Error processing {md_file}: {e}")
    
    print(f"\nFixed {fixed_count} broken links")

if __name__ == "__main__":
    fix_broken_links()