#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

print("Testing generator.py imports step by step...")

try:
    print("1. Testing basic imports...")
    import logging
    from typing import List, Dict, Any, Optional
    from datetime import datetime
    print("✓ Basic imports OK")
    
    print("2. Testing thumbnail_models import...")
    from ai_video_editor.modules.thumbnail_generation.thumbnail_models import ThumbnailPackage, ThumbnailVariation, ThumbnailConcept, ThumbnailGenerationStats
    print("✓ Thumbnail models OK")
    
    print("3. Testing concept_analyzer import...")
    from ai_video_editor.modules.thumbnail_generation.concept_analyzer import ThumbnailConceptAnalyzer
    print("✓ Concept analyzer OK")
    
    print("4. Testing image_generator import...")
    from ai_video_editor.modules.thumbnail_generation.image_generator import ThumbnailImageGenerator
    print("✓ Image generator OK")
    
    print("5. Testing synchronizer import...")
    from ai_video_editor.modules.thumbnail_generation.synchronizer import ThumbnailMetadataSynchronizer
    print("✓ Synchronizer OK")
    
    print("6. Testing core imports...")
    from ai_video_editor.core.content_context import ContentContext
    from ai_video_editor.core.cache_manager import CacheManager
    from ai_video_editor.core.exceptions import ContentContextError
    from ai_video_editor.modules.intelligence.gemini_client import GeminiClient
    print("✓ Core imports OK")
    
    print("7. Testing class definitions...")
    
    class ThumbnailGenerationError(ContentContextError):
        """Raised when thumbnail generation fails."""
        
        def __init__(self, message: str, **kwargs):
            super().__init__(message, error_code="THUMBNAIL_GENERATION_ERROR", **kwargs)
    
    print("✓ ThumbnailGenerationError defined")
    
    class ThumbnailGenerator:
        """Main thumbnail generation orchestrator."""
        
        def __init__(self, gemini_client: GeminiClient, cache_manager: CacheManager):
            """Initialize ThumbnailGenerator with required dependencies."""
            self.gemini_client = gemini_client
            self.cache_manager = cache_manager
            
            # Initialize sub-components
            self.concept_analyzer = ThumbnailConceptAnalyzer(gemini_client)
            self.image_generator = ThumbnailImageGenerator(cache_manager)
            self.synchronizer = ThumbnailMetadataSynchronizer()
            
            # Performance tracking
            self.stats = ThumbnailGenerationStats()
            
            print("ThumbnailGenerator initialized")
    
    print("✓ ThumbnailGenerator defined")
    
    print("8. Testing actual generator.py file...")
    with open('ai_video_editor/modules/thumbnail_generation/generator.py', 'r') as f:
        content = f.read()
    
    # Execute the file content
    exec(content)
    print("✓ Generator.py executed successfully")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()