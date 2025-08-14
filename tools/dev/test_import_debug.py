#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

print("Testing import step by step...")

try:
    # Test if we can import the module
    print("1. Importing module...")
    import ai_video_editor.modules.thumbnail_generation.generator as gen_module
    print(f"   Module imported: {gen_module}")
    print(f"   Module file: {gen_module.__file__}")
    print(f"   Module attributes: {[attr for attr in dir(gen_module) if not attr.startswith('_')]}")
    
    # Test if we can access the class
    print("2. Checking for ThumbnailGenerator class...")
    if hasattr(gen_module, 'ThumbnailGenerator'):
        print("   ✓ ThumbnailGenerator found")
        ThumbnailGenerator = gen_module.ThumbnailGenerator
        print(f"   Class: {ThumbnailGenerator}")
    else:
        print("   ✗ ThumbnailGenerator not found")
        
        # Try to execute the file manually
        print("3. Executing file manually...")
        with open('ai_video_editor/modules/thumbnail_generation/generator.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"   File size: {len(content)} characters")
        print(f"   First 100 chars: {repr(content[:100])}")
        print(f"   Last 100 chars: {repr(content[-100:])}")
        
        # Create a new namespace and execute
        namespace = {}
        exec(content, namespace)
        print(f"   Namespace after exec: {[k for k in namespace.keys() if not k.startswith('_')]}")
        
        if 'ThumbnailGenerator' in namespace:
            print("   ✓ ThumbnailGenerator found in executed namespace")
        else:
            print("   ✗ ThumbnailGenerator not found in executed namespace")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()