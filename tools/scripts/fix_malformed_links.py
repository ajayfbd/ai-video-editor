#!/usr/bin/env python3
"""
Fix malformed links with docs/../ patterns
"""

import re
from pathlib import Path

def fix_malformed_links():
    """Fix malformed docs/../ link patterns"""
    
    project_root = Path(".")
    fixed_count = 0
    
    for md_file in project_root.glob("**/*.md"):
        if any(exclude in str(md_file) for exclude in ["archive", ".git", ".agent.md"]):
            continue
            
        try:
            content = md_file.read_text(encoding='utf-8')
            original_content = content
            
            # Fix docs/../ patterns
            content = re.sub(r'docs/\.\./([^)]+)', r'docs/\1', content)
            
            # Write back if changed
            if content != original_content:
                md_file.write_text(content, encoding='utf-8')
                print(f"Fixed malformed links in {md_file}")
                fixed_count += 1
                
        except Exception as e:
            print(f"Error processing {md_file}: {e}")
    
    print(f"\nFixed malformed links in {fixed_count} files")

if __name__ == "__main__":
    fix_malformed_links()