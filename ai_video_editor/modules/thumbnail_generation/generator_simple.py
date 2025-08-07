"""
Simple Thumbnail Generator (without decorators for testing)
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from .thumbnail_models import ThumbnailPackage, ThumbnailVariation, ThumbnailConcept, ThumbnailGenerationStats
from .concept_analyzer import ThumbnailConceptAnalyzer, ConceptAnalysisError
from .image_generator import ThumbnailImageGenerator, ImageGenerationError
from .synchronizer import ThumbnailMetadataSynchronizer, SynchronizationError

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
    """
    Main thumbnail generation orchestrator.
    """
    
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
        
        logger.info("ThumbnailGenerator initialized with all components")
    
    async def generate_thumbnail_package(self, context: ContentContext) -> ThumbnailPackage:
        """Generate complete thumbnail package with A/B testing variations."""
        start_time = time.time()
        
        try:
            logger.info(f"Starting thumbnail package generation for project: {context.project_id}")
            
            # Step 1: Analyze thumbnail concepts
            concepts = await self.analyze_thumbnail_concepts(context)
            
            if not concepts:
                raise ThumbnailGenerationError("No thumbnail concepts could be generated", context)
            
            # Step 2: Generate thumbnail variations
            variations = await self.generate_thumbnail_variations(concepts, context)
            
            if not variations:
                raise ThumbnailGenerationError("No thumbnail variations could be generated", context)
            
            # Step 3: Create thumbnail package
            thumbnail_package = self._create_thumbnail_package(variations, context)
            
            # Step 4: Synchronize with metadata
            synchronized_package = self.synchronize_with_metadata(thumbnail_package, context)
            
            # Step 5: Update ContentContext
            self._update_context_with_results(context, synchronized_package)
            
            total_time = time.time() - start_time
            logger.info(f"Thumbnail package generation completed in {total_time:.2f}s")
            
            return synchronized_package
            
        except Exception as e:
            logger.error(f"Thumbnail package generation failed: {str(e)}")
            raise ThumbnailGenerationError(f"Failed to generate thumbnail package: {str(e)}", context)
    
    async def analyze_thumbnail_concepts(self, context: ContentContext) -> List[ThumbnailConcept]:
        """Analyze visual highlights and emotional peaks to generate thumbnail concepts."""
        try:
            # Generate new concepts
            concepts = await self.concept_analyzer.analyze_thumbnail_concepts(context)
            
            # Update statistics
            self.stats.concepts_analyzed = len(concepts)
            
            return concepts
            
        except ConceptAnalysisError:
            raise  # Re-raise concept analysis errors
        except Exception as e:
            logger.error(f"Thumbnail concept analysis failed: {str(e)}")
            raise ThumbnailGenerationError(f"Concept analysis failed: {str(e)}", context)
    
    async def generate_thumbnail_variations(
        self, 
        concepts: List[ThumbnailConcept], 
        context: ContentContext
    ) -> List[ThumbnailVariation]:
        """Generate multiple thumbnail variations for A/B testing."""
        try:
            variations = []
            
            # Generate variations for top concepts
            top_concepts = sorted(concepts, key=lambda c: c.thumbnail_potential, reverse=True)[:5]
            
            # Generate images sequentially for simplicity
            for concept in top_concepts:
                variation = await self._generate_single_variation(concept, context)
                if variation:
                    variations.append(variation)
            
            # Sort variations by confidence score
            variations.sort(key=lambda v: v.confidence_score, reverse=True)
            
            return variations
            
        except Exception as e:
            logger.error(f"Thumbnail variation generation failed: {str(e)}")
            raise ThumbnailGenerationError(f"Variation generation failed: {str(e)}", context)
    
    def synchronize_with_metadata(
        self, 
        thumbnail_package: ThumbnailPackage, 
        context: ContentContext
    ) -> ThumbnailPackage:
        """Ensure thumbnail concepts align with metadata strategy."""
        try:
            # Perform synchronization analysis
            sync_data = self.synchronizer.synchronize_concepts(thumbnail_package, context)
            
            # Update package with synchronization data
            thumbnail_package.synchronized_metadata = sync_data
            
            return thumbnail_package
            
        except SynchronizationError:
            raise  # Re-raise synchronization errors
        except Exception as e:
            logger.error(f"Thumbnail-metadata synchronization failed: {str(e)}")
            raise ThumbnailGenerationError(f"Synchronization failed: {str(e)}", context)
    
    async def _generate_single_variation(
        self, 
        concept: ThumbnailConcept, 
        context: ContentContext
    ) -> Optional[ThumbnailVariation]:
        """Generate a single thumbnail variation from concept."""
        try:
            start_time = time.time()
            
            # Generate thumbnail image
            image_path = await self.image_generator.generate_thumbnail_image(concept, context)
            
            if not image_path:
                logger.warning(f"Image generation failed for concept {concept.concept_id}")
                return None
            
            # Calculate scores
            confidence_score = 0.8  # Simplified
            estimated_ctr = 0.12    # Simplified
            
            # Create variation
            variation = ThumbnailVariation(
                variation_id="",  # Will be auto-generated
                concept=concept,
                generated_image_path=image_path,
                generation_method="procedural",
                confidence_score=confidence_score,
                estimated_ctr=estimated_ctr,
                visual_appeal_score=0.8,
                text_readability_score=0.8,
                brand_consistency_score=0.7,
                generation_time=time.time() - start_time,
                generation_cost=0.0
            )
            
            return variation
            
        except Exception as e:
            logger.warning(f"Single variation generation failed: {str(e)}")
            return None
    
    def _create_thumbnail_package(
        self, 
        variations: List[ThumbnailVariation], 
        context: ContentContext
    ) -> ThumbnailPackage:
        """Create thumbnail package from variations."""
        if not variations:
            raise ThumbnailGenerationError("No variations available for package creation", context)
        
        # Find recommended variation (highest confidence score)
        recommended_variation = max(variations, key=lambda v: v.confidence_score)
        
        # Create package
        package = ThumbnailPackage(
            package_id="",  # Will be auto-generated
            variations=variations,
            recommended_variation=recommended_variation.variation_id,
            generation_timestamp=datetime.now(),
            synchronized_metadata={},  # Will be filled by synchronizer
            a_b_testing_config={},  # Will be filled later
            performance_predictions={},
            total_generation_time=sum(v.generation_time for v in variations),
            total_generation_cost=sum(v.generation_cost for v in variations)
        )
        
        return package
    
    def _update_context_with_results(self, context: ContentContext, thumbnail_package: ThumbnailPackage):
        """Update ContentContext with thumbnail generation results."""
        # Store thumbnail concepts
        context.thumbnail_concepts = [
            variation.concept.to_dict() for variation in thumbnail_package.variations
        ]
        
        # Store generated thumbnails
        context.generated_thumbnails = [
            variation.to_dict() for variation in thumbnail_package.variations
        ]