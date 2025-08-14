#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '.')

# Test individual imports
print("Testing individual imports...")

try:
    from ai_video_editor.modules.thumbnail_generation.thumbnail_models import ThumbnailPackage, ThumbnailVariation, ThumbnailConcept, ThumbnailGenerationStats
    print("✓ thumbnail_models imports OK")
except Exception as e:
    print(f"✗ thumbnail_models import failed: {e}")

try:
    from ai_video_editor.modules.thumbnail_generation.concept_analyzer import ThumbnailConceptAnalyzer, ConceptAnalysisError
    print("✓ concept_analyzer imports OK")
except Exception as e:
    print(f"✗ concept_analyzer import failed: {e}")

try:
    from ai_video_editor.modules.thumbnail_generation.image_generator import ThumbnailImageGenerator, ImageGenerationError
    print("✓ image_generator imports OK")
except Exception as e:
    print(f"✗ image_generator import failed: {e}")

try:
    from ai_video_editor.modules.thumbnail_generation.synchronizer import ThumbnailMetadataSynchronizer, SynchronizationError
    print("✓ synchronizer imports OK")
except Exception as e:
    print(f"✗ synchronizer import failed: {e}")

try:
    from ai_video_editor.core.content_context import ContentContext
    print("✓ ContentContext imports OK")
except Exception as e:
    print(f"✗ ContentContext import failed: {e}")

try:
    from ai_video_editor.core.cache_manager import CacheManager
    print("✓ CacheManager imports OK")
except Exception as e:
    print(f"✗ CacheManager import failed: {e}")

try:
    from ai_video_editor.core.exceptions import ContentContextError, handle_errors
    print("✓ exceptions imports OK")
except Exception as e:
    print(f"✗ exceptions import failed: {e}")

try:
    from ai_video_editor.modules.intelligence.gemini_client import GeminiClient
    print("✓ GeminiClient imports OK")
except Exception as e:
    print(f"✗ GeminiClient import failed: {e}")

print("\nTesting generator module import...")

try:
    import ai_video_editor.modules.thumbnail_generation.generator as gen_module
    print("✓ generator module imported")
    
    # Check what's in the module
    attrs = [attr for attr in dir(gen_module) if not attr.startswith('_')]
    print(f"Module attributes: {attrs}")
    
    # Try to access the classes
    if hasattr(gen_module, 'ThumbnailGenerator'):
        print("✓ ThumbnailGenerator class found")
        ThumbnailGenerator = gen_module.ThumbnailGenerator
        print(f"ThumbnailGenerator type: {type(ThumbnailGenerator)}")
    else:
        print("✗ ThumbnailGenerator class NOT found")
        
    if hasattr(gen_module, 'ThumbnailGenerationError'):
        print("✓ ThumbnailGenerationError class found")
    else:
        print("✗ ThumbnailGenerationError class NOT found")
        
except Exception as e:
    print(f"✗ generator module import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting direct import...")
try:
    from ai_video_editor.modules.thumbnail_generation.generator import ThumbnailGenerator
    print("✓ Direct ThumbnailGenerator import successful")
except Exception as e:
    print(f"✗ Direct ThumbnailGenerator import failed: {e}")
    import traceback
    traceback.print_exc()