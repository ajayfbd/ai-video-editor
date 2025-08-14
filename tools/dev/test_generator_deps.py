print("Testing generator.py dependencies...")

# Test each import from generator.py individually
imports = [
    "import logging",
    "from typing import List, Dict, Any, Optional",
    "from datetime import datetime",
    "from ai_video_editor.modules.thumbnail_generation.thumbnail_models import ThumbnailPackage, ThumbnailVariation, ThumbnailConcept, ThumbnailGenerationStats",
    "from ai_video_editor.modules.thumbnail_generation.concept_analyzer import ThumbnailConceptAnalyzer",
    "from ai_video_editor.modules.thumbnail_generation.image_generator import ThumbnailImageGenerator",
    "from ai_video_editor.modules.thumbnail_generation.synchronizer import ThumbnailMetadataSynchronizer",
    "from ai_video_editor.core.content_context import ContentContext",
    "from ai_video_editor.core.cache_manager import CacheManager",
    "from ai_video_editor.core.exceptions import ContentContextError",
    "from ai_video_editor.modules.intelligence.gemini_client import GeminiClient"
]

for i, import_stmt in enumerate(imports, 1):
    try:
        exec(import_stmt)
        print(f"✓ {i}. {import_stmt}")
    except Exception as e:
        print(f"✗ {i}. {import_stmt}")
        print(f"   Error: {e}")
        break

print("\nTesting file execution...")
try:
    with open('ai_video_editor/modules/thumbnail_generation/generator.py', 'r') as f:
        content = f.read()
    
    print(f"File size: {len(content)} characters")
    
    if len(content) == 0:
        print("✗ File is empty!")
    else:
        print("✓ File has content")
        
        # Try to compile it
        try:
            compile(content, 'generator.py', 'exec')
            print("✓ File compiles successfully")
        except Exception as e:
            print(f"✗ Compilation error: {e}")
            
        # Try to execute it
        try:
            namespace = {}
            exec(content, namespace)
            classes = [name for name, obj in namespace.items() if isinstance(obj, type)]
            print(f"✓ File executes successfully, classes: {classes}")
        except Exception as e:
            print(f"✗ Execution error: {e}")
            import traceback
            traceback.print_exc()

except Exception as e:
    print(f"✗ Failed to read file: {e}")