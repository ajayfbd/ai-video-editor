"""
Thumbnail Generator

Main class that orchestrates the complete thumbnail generation process.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .thumbnail_models import ThumbnailPackage, ThumbnailVariation, ThumbnailConcept, ThumbnailGenerationStats
from .concept_analyzer import ThumbnailConceptAnalyzer
from .image_generator import ThumbnailImageGenerator
from .synchronizer import ThumbnailMetadataSynchronizer

from ...core.content_context import ContentContext
from ...core.cache_manager import CacheManager
from ...core.exceptions import ContentContextError
from ...modules.intelligence.gemini_client import GeminiClient


logger = logging.getLogger(__name__)


class ThumbnailGenerationError(ContentContextError):
    """Raised when thumbnail generation fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="THUMBNAIL_GENERATION_ERROR", **kwargs)


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
        
        logger.info("ThumbnailGenerator initialized")
    
    async def generate_thumbnail_package(self, context: ContentContext) -> ThumbnailPackage:
        """Generate complete thumbnail package with A/B testing variations."""
        try:
            logger.info(f"Starting thumbnail package generation for project: {context.project_id}")
            
            # Step 1: Analyze thumbnail concepts
            concepts = await self.concept_analyzer.analyze_thumbnail_concepts(context)
            
            if not concepts:
                raise ThumbnailGenerationError("No thumbnail concepts could be generated", context)
            
            # Step 2: Generate thumbnail variations
            variations = []
            top_concepts = sorted(concepts, key=lambda c: c.thumbnail_potential, reverse=True)[:3]
            
            for concept in top_concepts:
                try:
                    image_path = await self.image_generator.generate_thumbnail_image(concept, context)
                    if image_path:
                        variation = ThumbnailVariation(
                            variation_id="",
                            concept=concept,
                            generated_image_path=image_path,
                            generation_method="procedural",
                            confidence_score=0.8,
                            estimated_ctr=0.12,
                            visual_appeal_score=0.8,
                            text_readability_score=0.8,
                            brand_consistency_score=0.7
                        )
                        variations.append(variation)
                except Exception as e:
                    logger.warning(f"Variation generation failed: {e}")
                    continue
            
            if not variations:
                raise ThumbnailGenerationError("No thumbnail variations could be generated", context)
            
            # Step 3: Create thumbnail package
            recommended_variation = max(variations, key=lambda v: v.confidence_score)
            
            package = ThumbnailPackage(
                package_id="",
                variations=variations,
                recommended_variation=recommended_variation.variation_id,
                generation_timestamp=datetime.now(),
                synchronized_metadata={},
                a_b_testing_config={},
                performance_predictions={}
            )
            
            # Step 4: Synchronize with metadata
            try:
                sync_data = self.synchronizer.synchronize_concepts(package, context)
                package.synchronized_metadata = sync_data
            except Exception as e:
                logger.warning(f"Synchronization failed: {e}")
            
            # Step 5: Update ContentContext
            context.thumbnail_concepts = [v.concept.to_dict() for v in variations]
            context.generated_thumbnails = [v.to_dict() for v in variations]
            
            logger.info(f"Thumbnail package generation completed with {len(variations)} variations")
            return package
            
        except Exception as e:
            logger.error(f"Thumbnail package generation failed: {str(e)}")
            raise ThumbnailGenerationError(f"Failed to generate thumbnail package: {str(e)}", context)