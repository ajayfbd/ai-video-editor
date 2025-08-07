print("Starting test...")

try:
    from ai_video_editor.modules.thumbnail_generation.thumbnail_models import ThumbnailPackage
    print("✓ ThumbnailPackage imported")
except Exception as e:
    print(f"✗ ThumbnailPackage failed: {e}")

try:
    from ai_video_editor.modules.thumbnail_generation.concept_analyzer import ThumbnailConceptAnalyzer
    print("✓ ThumbnailConceptAnalyzer imported")
except Exception as e:
    print(f"✗ ThumbnailConceptAnalyzer failed: {e}")

try:
    from ai_video_editor.modules.thumbnail_generation.generator import ThumbnailGenerator
    print("✓ ThumbnailGenerator imported")
except Exception as e:
    print(f"✗ ThumbnailGenerator failed: {e}")

print("Test complete.")