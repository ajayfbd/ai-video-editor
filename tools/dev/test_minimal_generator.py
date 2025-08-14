#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

# Test minimal generator class
print("Testing minimal generator class...")

try:
    from ai_video_editor.core.exceptions import ContentContextError
    
    class ThumbnailGenerationError(ContentContextError):
        """Raised when thumbnail generation fails."""
        
        def __init__(self, message: str, **kwargs):
            super().__init__(message, error_code="THUMBNAIL_GENERATION_ERROR", **kwargs)
    
    print("✓ ThumbnailGenerationError class created")
    
    from ai_video_editor.modules.intelligence.gemini_client import GeminiClient
    from ai_video_editor.core.cache_manager import CacheManager
    
    class ThumbnailGenerator:
        """Minimal thumbnail generator for testing."""
        
        def __init__(self, gemini_client: GeminiClient, cache_manager: CacheManager):
            self.gemini_client = gemini_client
            self.cache_manager = cache_manager
            print("ThumbnailGenerator initialized")
    
    print("✓ ThumbnailGenerator class created")
    
    # Test instantiation
    mock_client = None  # We'll use None for testing
    mock_cache = None   # We'll use None for testing
    
    # This should work if the class is properly defined
    print("Classes defined successfully!")
    
except Exception as e:
    print(f"✗ Error creating classes: {e}")
    import traceback
    traceback.print_exc()