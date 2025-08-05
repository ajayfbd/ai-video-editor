"""
Metadata Package Integration - Complete metadata synchronization system.

This module provides comprehensive metadata package integration that ensures
synchronization between video content, thumbnails, and AI Director decisions.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ai_video_editor.core.content_context import ContentContext
from ai_video_editor.core.cache_manager import CacheManager
from ai_video_editor.core.exceptions import (
    ContentContextError,
    APIIntegrationError,
    handle_errors
)
from ai_video_editor.modules.intelligence.metadata_generator import (
    MetadataGenerator,
    MetadataPackage,
    MetadataVariation
)


logger = logging.getLogger(__name__)


@dataclass
class ThumbnailMetadataAlignment:
    """Represents alignment between thumbnail concepts and metadata."""
    thumbnail_concept: str
    aligned_title: str
    aligned_description: str
    hook_text_integration: str
    visual_consistency_score: float
    keyword_overlap: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'thumbnail_concept': self.thumbnail_concept,
            'aligned_title': self.aligned_title,
            'aligned_description': self.aligned_description,
            'hook_text_integration': self.hook_text_integration,
            'visual_consistency_score': self.visual_consistency_score,
            'keyword_overlap': self.keyword_overlap
        }


@dataclass
class MetadataValidationResult:
    """Results of metadata package validation."""
    is_complete: bool
    missing_components: List[str]
    quality_score: float
    synchronization_score: float
    ai_director_alignment: float
    validation_errors: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_complete': self.is_complete,
            'missing_components': self.missing_components,
            'quality_score': self.quality_score,
            'synchronization_score': self.synchronization_score,
            'ai_director_alignment': self.ai_director_alignment,
            'validation_errors': self.validation_errors,
            'recommendations': self.recommendations
        }


@dataclass
class IntegratedMetadataPackage:
    """Complete integrated metadata package ready for publishing."""
    primary_metadata: MetadataVariation
    alternative_variations: List[MetadataVariation]
    thumbnail_alignments: List[ThumbnailMetadataAlignment]
    ai_director_integration: Dict[str, Any]
    content_synchronization: Dict[str, Any]
    validation_result: MetadataValidationResult
    publish_readiness_score: float
    generation_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'primary_metadata': self.primary_metadata.to_dict(),
            'alternative_variations': [var.to_dict() for var in self.alternative_variations],
            'thumbnail_alignments': [align.to_dict() for align in self.thumbnail_alignments],
            'ai_director_integration': self.ai_director_integration,
            'content_synchronization': self.content_synchronization,
            'validation_result': self.validation_result.to_dict(),
            'publish_readiness_score': self.publish_readiness_score,
            'generation_timestamp': self.generation_timestamp.isoformat()
        }


class MetadataPackageIntegrator:
    """
    Complete metadata package integration system.
    
    Ensures synchronization between video content, thumbnails, and AI Director
    decisions to create publish-ready metadata packages.
    """
    
    def __init__(self, cache_manager: CacheManager, metadata_generator: MetadataGenerator):
        """
        Initialize MetadataPackageIntegrator.
        
        Args:
            cache_manager: CacheManager instance for caching
            metadata_generator: MetadataGenerator for base metadata creation
        """
        self.cache_manager = cache_manager
        self.metadata_generator = metadata_generator
        
        # Integration scoring weights
        self.scoring_weights = {
            'content_alignment': 0.25,
            'thumbnail_sync': 0.25,
            'ai_director_alignment': 0.25,
            'seo_optimization': 0.25
        }
        
        # Validation requirements
        self.validation_requirements = {
            'required_components': [
                'title', 'description', 'tags', 'thumbnail_concepts',
                'hook_text', 'target_keywords'
            ],
            'minimum_quality_score': 0.7,
            'minimum_sync_score': 0.8,
            'minimum_ai_alignment': 0.75
        }
        
        logger.info("MetadataPackageIntegrator initialized with comprehensive integration")
    
    @handle_errors(logger)
    async def create_integrated_package(self, context: ContentContext) -> ContentContext:
        """
        Create complete integrated metadata package.
        
        Args:
            context: ContentContext with analysis results and AI Director decisions
            
        Returns:
            Updated ContentContext with integrated metadata package
            
        Raises:
            ContentContextError: If integration fails
        """
        start_time = time.time()
        
        try:
            logger.info(f"Creating integrated metadata package for project {context.project_id}")
            
            # Validate prerequisites
            self._validate_prerequisites(context)
            
            # Check cache first
            cache_key = f"integrated_metadata:{context.project_id}"
            cached_package = self.cache_manager.get(cache_key)
            if cached_package:
                logger.info("Using cached integrated metadata package")
                context.metadata_variations = [cached_package]
                return context
            
            # Generate base metadata package if not exists
            if not context.metadata_variations:
                context = await self.metadata_generator.generate_metadata_package(context)
            
            # Extract AI Director decisions
            ai_director_decisions = self._extract_ai_director_decisions(context)
            
            # Create thumbnail-metadata alignments
            thumbnail_alignments = await self._create_thumbnail_alignments(context)
            
            # Synchronize with video content
            content_synchronization = self._synchronize_with_content(context)
            
            # Integrate with AI Director decisions
            ai_director_integration = self._integrate_ai_director_decisions(
                context, ai_director_decisions
            )
            
            # Select and optimize primary metadata
            primary_metadata = await self._select_primary_metadata(
                context, thumbnail_alignments, ai_director_integration
            )
            
            # Create alternative variations
            alternative_variations = await self._create_alternative_variations(
                context, primary_metadata, thumbnail_alignments
            )
            
            # Validate complete package
            validation_result = self._validate_metadata_package(
                context, primary_metadata, alternative_variations, thumbnail_alignments
            )
            
            # Calculate publish readiness score
            publish_readiness_score = self._calculate_publish_readiness(
                primary_metadata, validation_result, ai_director_integration
            )
            
            # Create integrated package
            integrated_package = IntegratedMetadataPackage(
                primary_metadata=primary_metadata,
                alternative_variations=alternative_variations,
                thumbnail_alignments=thumbnail_alignments,
                ai_director_integration=ai_director_integration,
                content_synchronization=content_synchronization,
                validation_result=validation_result,
                publish_readiness_score=publish_readiness_score,
                generation_timestamp=datetime.now()
            )
            
            # Store in context
            context.metadata_variations = [integrated_package.to_dict()]
            
            # Cache the integrated package
            self.cache_manager.put(cache_key, integrated_package.to_dict(), ttl=86400)
            
            # Update processing metrics
            processing_time = time.time() - start_time
            context.processing_metrics.add_module_metrics(
                "metadata_integration", processing_time, 0
            )
            
            logger.info(f"Integrated metadata package created in {processing_time:.2f}s")
            logger.info(f"Publish readiness score: {publish_readiness_score:.1%}")
            logger.info(f"Validation status: {'PASSED' if validation_result.is_complete else 'FAILED'}")
            
            return context
            
        except Exception as e:
            logger.error(f"Metadata integration failed: {str(e)}")
            raise ContentContextError(
                f"Metadata integration failed: {str(e)}",
                context_state=context
            )
    
    def _validate_prerequisites(self, context: ContentContext) -> None:
        """Validate that required data is available for integration."""
        missing_components = []
        
        if not context.trending_keywords:
            missing_components.append("trending_keywords")
        
        if not context.key_concepts:
            missing_components.append("key_concepts")
        
        if not context.content_themes:
            missing_components.append("content_themes")
        
        # Check for AI Director decisions (optional but recommended)
        if not hasattr(context, 'ai_director_plan') or not context.ai_director_plan:
            logger.warning("AI Director plan not available - integration will be limited")
        
        if missing_components:
            raise ContentContextError(
                f"Missing required components for metadata integration: {missing_components}",
                context_state=context
            )
    
    def _extract_ai_director_decisions(self, context: ContentContext) -> Dict[str, Any]:
        """Extract AI Director decisions for metadata integration."""
        ai_decisions = {
            'has_director_plan': False,
            'creative_strategy': 'default',
            'target_emotions': [],
            'key_messages': [],
            'content_strategy': {}
        }
        
        # Extract from AI Director plan if available
        if hasattr(context, 'ai_director_plan') and context.ai_director_plan:
            ai_decisions['has_director_plan'] = True
            
            # Extract creative strategy
            if 'creative_strategy' in context.ai_director_plan:
                ai_decisions['creative_strategy'] = context.ai_director_plan['creative_strategy']
            
            # Extract target emotions from emotional markers
            if context.emotional_markers:
                ai_decisions['target_emotions'] = [
                    marker.emotion for marker in context.emotional_markers[:3]
                ]
            
            # Extract key messages from content themes
            ai_decisions['key_messages'] = context.content_themes[:5]
            
            # Extract content strategy
            if 'content_strategy' in context.ai_director_plan:
                ai_decisions['content_strategy'] = context.ai_director_plan['content_strategy']
        
        # Fallback to content analysis
        else:
            # Use emotional markers as target emotions
            if context.emotional_markers:
                ai_decisions['target_emotions'] = [
                    marker.emotion for marker in context.emotional_markers[:3]
                ]
            
            # Use content themes as key messages
            ai_decisions['key_messages'] = context.content_themes[:5]
            
            # Determine strategy from content type
            ai_decisions['creative_strategy'] = f"{context.content_type.value}_focused"
        
        return ai_decisions
    
    async def _create_thumbnail_alignments(self, context: ContentContext) -> List[ThumbnailMetadataAlignment]:
        """Create alignments between thumbnail concepts and metadata."""
        alignments = []
        
        # Get thumbnail concepts from context or AI Director
        thumbnail_concepts = []
        
        # Extract from thumbnail_concepts field
        if context.thumbnail_concepts:
            for concept_data in context.thumbnail_concepts:
                if isinstance(concept_data, dict) and 'concept' in concept_data:
                    thumbnail_concepts.append(concept_data['concept'])
                elif isinstance(concept_data, str):
                    thumbnail_concepts.append(concept_data)
        
        # Extract from AI Director plan
        if hasattr(context, 'ai_director_plan') and context.ai_director_plan:
            if 'thumbnail_concepts' in context.ai_director_plan:
                thumbnail_concepts.extend(context.ai_director_plan['thumbnail_concepts'])
        
        # Fallback to visual highlights
        if not thumbnail_concepts and context.visual_highlights:
            thumbnail_concepts = [
                highlight.description for highlight in context.visual_highlights[:3]
            ]
        
        # Create alignments for each concept
        for concept in thumbnail_concepts[:5]:  # Limit to top 5 concepts
            # Generate aligned title
            aligned_title = await self._generate_aligned_title(concept, context)
            
            # Generate aligned description
            aligned_description = await self._generate_aligned_description(concept, context)
            
            # Create hook text integration
            hook_text = self._generate_hook_text(concept, context)
            
            # Calculate visual consistency score
            visual_score = self._calculate_visual_consistency(concept, context)
            
            # Find keyword overlap
            keyword_overlap = self._find_keyword_overlap(concept, context)
            
            alignment = ThumbnailMetadataAlignment(
                thumbnail_concept=concept,
                aligned_title=aligned_title,
                aligned_description=aligned_description,
                hook_text_integration=hook_text,
                visual_consistency_score=visual_score,
                keyword_overlap=keyword_overlap
            )
            
            alignments.append(alignment)
        
        return alignments
    
    async def _generate_aligned_title(self, concept: str, context: ContentContext) -> str:
        """Generate title aligned with thumbnail concept."""
        # Extract key elements from concept
        concept_words = concept.lower().split()
        primary_keyword = context.trending_keywords.primary_keywords[0] if context.trending_keywords.primary_keywords else "content"
        
        # Create concept-aligned title patterns
        title_patterns = [
            f"The {concept} That Will Change Your {primary_keyword.split()[0]} Forever",
            f"How {concept} Reveals the Secret to {primary_keyword}",
            f"Why This {concept} Is the Key to {primary_keyword} Success",
            f"{concept}: The Ultimate {primary_keyword} Strategy",
            f"This {concept} Shows Why {primary_keyword} Actually Works"
        ]
        
        # Select best pattern based on concept and keywords
        import random
        selected_title = random.choice(title_patterns)
        
        # Optimize length
        if len(selected_title) > 60:
            # Truncate while preserving key elements
            words = selected_title.split()
            while len(' '.join(words)) > 60 and len(words) > 5:
                words.pop()
            selected_title = ' '.join(words)
        
        return selected_title
    
    async def _generate_aligned_description(self, concept: str, context: ContentContext) -> str:
        """Generate description aligned with thumbnail concept."""
        primary_keyword = context.trending_keywords.primary_keywords[0] if context.trending_keywords.primary_keywords else "this topic"
        
        # Create concept-focused description
        description_parts = [
            f"🎯 Discover how {concept.lower()} reveals the truth about {primary_keyword}!",
            "",
            f"In this video, you'll see exactly how {concept.lower()} demonstrates:",
            f"• The real power of {primary_keyword}",
            f"• Why {concept.lower()} changes everything",
            f"• How to apply these insights immediately",
            "",
            f"💡 The {concept.lower()} in the thumbnail shows the exact moment when everything clicks!",
            "",
            "⏰ Timestamps:",
        ]
        
        # Add timestamps from emotional markers
        if context.emotional_markers:
            for marker in context.emotional_markers[:4]:
                minutes = int(marker.timestamp // 60)
                seconds = int(marker.timestamp % 60)
                timestamp_str = f"{minutes:02d}:{seconds:02d}"
                description_parts.append(f"{timestamp_str} - {marker.context}")
        
        # Add engagement elements
        description_parts.extend([
            "",
            f"🔥 Ready to master {primary_keyword}? Watch now!",
            "",
            f"#{primary_keyword.replace(' ', '').lower()} #tutorial #explained #tips"
        ])
        
        return '\n'.join(description_parts)
    
    def _generate_hook_text(self, concept: str, context: ContentContext) -> str:
        """Generate hook text that integrates thumbnail concept with title."""
        # Extract emotional trigger from concept
        emotional_triggers = {
            'chart': 'This chart reveals',
            'graph': 'This graph shows',
            'money': 'This money secret',
            'growth': 'This growth hack',
            'calculator': 'This calculation proves',
            'comparison': 'This comparison exposes',
            'formula': 'This formula unlocks',
            'strategy': 'This strategy delivers'
        }
        
        # Find matching trigger
        trigger = "This"
        for key, value in emotional_triggers.items():
            if key in concept.lower():
                trigger = value
                break
        
        # Create hook text variations
        hook_variations = [
            f"{trigger} everything!",
            f"{trigger} the truth!",
            f"{trigger} why it works!",
            f"{trigger} the secret!",
            f"{trigger} what they won't tell you!"
        ]
        
        # Select based on content emotion
        if context.emotional_markers:
            dominant_emotion = max(context.emotional_markers, key=lambda x: x.intensity).emotion
            if dominant_emotion in ['excitement', 'surprise']:
                return hook_variations[0]  # "everything!"
            elif dominant_emotion in ['curiosity']:
                return hook_variations[3]  # "the secret!"
        
        return hook_variations[1]  # Default: "the truth!"
    
    def _calculate_visual_consistency(self, concept: str, context: ContentContext) -> float:
        """Calculate visual consistency score between concept and content."""
        score = 0.0
        
        # Check alignment with visual highlights
        if context.visual_highlights:
            for highlight in context.visual_highlights:
                # Check for concept words in highlight description
                concept_words = set(concept.lower().split())
                highlight_words = set(highlight.description.lower().split())
                
                overlap = len(concept_words.intersection(highlight_words))
                if overlap > 0:
                    score += 0.2 * overlap
        
        # Check alignment with content themes
        if context.content_themes:
            concept_words = set(concept.lower().split())
            for theme in context.content_themes:
                theme_words = set(theme.lower().split())
                overlap = len(concept_words.intersection(theme_words))
                if overlap > 0:
                    score += 0.1 * overlap
        
        # Check alignment with key concepts
        if context.key_concepts:
            concept_words = set(concept.lower().split())
            for key_concept in context.key_concepts:
                key_words = set(key_concept.lower().split())
                overlap = len(concept_words.intersection(key_words))
                if overlap > 0:
                    score += 0.15 * overlap
        
        # Normalize to 0-1 range
        return min(score, 1.0)
    
    def _find_keyword_overlap(self, concept: str, context: ContentContext) -> List[str]:
        """Find keyword overlap between concept and trending keywords."""
        overlap = []
        concept_words = set(concept.lower().split())
        
        if context.trending_keywords:
            # Check primary keywords
            for keyword in context.trending_keywords.primary_keywords:
                keyword_words = set(keyword.lower().split())
                if concept_words.intersection(keyword_words):
                    overlap.append(keyword)
            
            # Check long-tail keywords
            for keyword in context.trending_keywords.long_tail_keywords:
                keyword_words = set(keyword.lower().split())
                if concept_words.intersection(keyword_words):
                    overlap.append(keyword)
        
        return overlap[:5]  # Limit to top 5 overlaps
    
    def _synchronize_with_content(self, context: ContentContext) -> Dict[str, Any]:
        """Synchronize metadata with video content analysis."""
        synchronization = {
            'content_alignment_score': 0.0,
            'emotional_alignment': {},
            'visual_alignment': {},
            'audio_alignment': {},
            'timing_synchronization': {}
        }
        
        # Calculate content alignment score
        alignment_factors = []
        
        # Emotional alignment
        if context.emotional_markers:
            emotional_coverage = len(context.emotional_markers) / 10.0  # Normalize to 0-1
            synchronization['emotional_alignment'] = {
                'peak_count': len(context.emotional_markers),
                'dominant_emotions': [marker.emotion for marker in context.emotional_markers[:3]],
                'coverage_score': min(emotional_coverage, 1.0)
            }
            alignment_factors.append(synchronization['emotional_alignment']['coverage_score'])
        
        # Visual alignment
        if context.visual_highlights:
            visual_coverage = len(context.visual_highlights) / 8.0  # Normalize to 0-1
            synchronization['visual_alignment'] = {
                'highlight_count': len(context.visual_highlights),
                'thumbnail_potential': sum(h.thumbnail_potential for h in context.visual_highlights) / len(context.visual_highlights),
                'coverage_score': min(visual_coverage, 1.0)
            }
            alignment_factors.append(synchronization['visual_alignment']['coverage_score'])
        
        # Audio alignment
        if context.audio_analysis:
            audio_quality = context.audio_analysis.overall_confidence
            synchronization['audio_alignment'] = {
                'transcript_quality': audio_quality,
                'concept_extraction': len(context.audio_analysis.financial_concepts) if hasattr(context.audio_analysis, 'financial_concepts') else 0,
                'coverage_score': audio_quality
            }
            alignment_factors.append(audio_quality)
        
        # Timing synchronization
        if context.emotional_markers and context.visual_highlights:
            # Calculate how well emotional peaks align with visual highlights
            timing_alignment = 0.0
            for emotion in context.emotional_markers:
                for visual in context.visual_highlights:
                    time_diff = abs(emotion.timestamp - visual.timestamp)
                    if time_diff <= 10.0:  # Within 10 seconds
                        timing_alignment += 1.0 / (time_diff + 1.0)
            
            timing_score = min(timing_alignment / len(context.emotional_markers), 1.0)
            synchronization['timing_synchronization'] = {
                'alignment_score': timing_score,
                'peak_visual_pairs': len(context.emotional_markers) + len(context.visual_highlights)
            }
            alignment_factors.append(timing_score)
        
        # Calculate overall content alignment score
        if alignment_factors:
            synchronization['content_alignment_score'] = sum(alignment_factors) / len(alignment_factors)
        
        return synchronization
    
    def _integrate_ai_director_decisions(self, context: ContentContext, 
                                       ai_decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate AI Director decisions into metadata."""
        integration = {
            'director_influence_score': 0.0,
            'creative_alignment': {},
            'strategic_alignment': {},
            'decision_integration': {}
        }
        
        if ai_decisions['has_director_plan']:
            # Creative alignment
            creative_score = 0.8  # High score when AI Director is available
            integration['creative_alignment'] = {
                'strategy': ai_decisions['creative_strategy'],
                'target_emotions': ai_decisions['target_emotions'],
                'alignment_score': creative_score
            }
            
            # Strategic alignment
            strategic_score = 0.85
            integration['strategic_alignment'] = {
                'key_messages': ai_decisions['key_messages'],
                'content_strategy': ai_decisions['content_strategy'],
                'alignment_score': strategic_score
            }
            
            # Decision integration
            integration['decision_integration'] = {
                'decisions_applied': len(ai_decisions['key_messages']) + len(ai_decisions['target_emotions']),
                'integration_completeness': 0.9
            }
            
            integration['director_influence_score'] = (creative_score + strategic_score) / 2
        
        else:
            # Fallback integration without AI Director
            integration['creative_alignment'] = {
                'strategy': 'content_driven',
                'target_emotions': ai_decisions['target_emotions'],
                'alignment_score': 0.6
            }
            
            integration['strategic_alignment'] = {
                'key_messages': ai_decisions['key_messages'],
                'content_strategy': {'type': 'basic'},
                'alignment_score': 0.5
            }
            
            integration['director_influence_score'] = 0.55
        
        return integration
    
    async def _select_primary_metadata(self, context: ContentContext,
                                     thumbnail_alignments: List[ThumbnailMetadataAlignment],
                                     ai_integration: Dict[str, Any]) -> MetadataVariation:
        """Select and optimize primary metadata variation."""
        # Get base metadata variations
        base_package = context.metadata_variations[0]
        variations = [MetadataVariation.from_dict(var) for var in base_package['variations']]
        
        # Score each variation for integration
        best_variation = None
        best_score = 0.0
        
        for variation in variations:
            # Calculate integration score
            score = self._calculate_integration_score(
                variation, thumbnail_alignments, ai_integration, context
            )
            
            if score > best_score:
                best_score = score
                best_variation = variation
        
        # Enhance the best variation with integration data
        if best_variation and thumbnail_alignments:
            # Use the best thumbnail alignment
            best_alignment = max(thumbnail_alignments, key=lambda x: x.visual_consistency_score)
            
            # Update title to integrate with thumbnail
            best_variation.title = best_alignment.aligned_title
            
            # Update description to include thumbnail reference
            best_variation.description = best_alignment.aligned_description
            
            # Add hook text as additional metadata
            if not hasattr(best_variation, 'hook_text'):
                best_variation.hook_text = best_alignment.hook_text_integration
        
        return best_variation or variations[0]  # Fallback to first variation
    
    def _calculate_integration_score(self, variation: MetadataVariation,
                                   thumbnail_alignments: List[ThumbnailMetadataAlignment],
                                   ai_integration: Dict[str, Any],
                                   context: ContentContext) -> float:
        """Calculate integration score for a metadata variation."""
        scores = []
        
        # Content alignment score
        content_score = variation.confidence_score
        scores.append(content_score * self.scoring_weights['content_alignment'])
        
        # Thumbnail synchronization score
        if thumbnail_alignments:
            thumbnail_score = max(align.visual_consistency_score for align in thumbnail_alignments)
            scores.append(thumbnail_score * self.scoring_weights['thumbnail_sync'])
        else:
            scores.append(0.5 * self.scoring_weights['thumbnail_sync'])
        
        # AI Director alignment score
        ai_score = ai_integration['director_influence_score']
        scores.append(ai_score * self.scoring_weights['ai_director_alignment'])
        
        # SEO optimization score
        seo_score = variation.seo_score
        scores.append(seo_score * self.scoring_weights['seo_optimization'])
        
        return sum(scores)
    
    async def _create_alternative_variations(self, context: ContentContext,
                                           primary_metadata: MetadataVariation,
                                           thumbnail_alignments: List[ThumbnailMetadataAlignment]) -> List[MetadataVariation]:
        """Create alternative metadata variations for A/B testing."""
        alternatives = []
        
        # Get base variations excluding the primary
        base_package = context.metadata_variations[0]
        base_variations = [MetadataVariation.from_dict(var) for var in base_package['variations']]
        
        # Filter out the primary variation
        for variation in base_variations:
            if variation.variation_id != primary_metadata.variation_id:
                # Enhance with thumbnail alignment if available
                if thumbnail_alignments:
                    # Find best alignment for this variation
                    best_alignment = max(thumbnail_alignments, key=lambda x: x.visual_consistency_score)
                    
                    # Create enhanced version
                    enhanced_variation = MetadataVariation(
                        title=variation.title,
                        description=variation.description,
                        tags=variation.tags,
                        variation_id=f"{variation.variation_id}_enhanced",
                        strategy=f"{variation.strategy}_integrated",
                        confidence_score=variation.confidence_score * 0.95,  # Slightly lower than primary
                        estimated_ctr=variation.estimated_ctr,
                        seo_score=variation.seo_score
                    )
                    
                    alternatives.append(enhanced_variation)
                else:
                    alternatives.append(variation)
        
        # Limit to top 3 alternatives
        return alternatives[:3]
    
    def _validate_metadata_package(self, context: ContentContext,
                                 primary_metadata: MetadataVariation,
                                 alternatives: List[MetadataVariation],
                                 thumbnail_alignments: List[ThumbnailMetadataAlignment]) -> MetadataValidationResult:
        """Validate complete metadata package."""
        missing_components = []
        validation_errors = []
        recommendations = []
        
        # Check required components
        required_components = self.validation_requirements['required_components']
        
        # Validate primary metadata
        if not primary_metadata.title or len(primary_metadata.title) < 10:
            missing_components.append('valid_title')
            validation_errors.append("Title is missing or too short")
        
        if not primary_metadata.description or len(primary_metadata.description) < 100:
            missing_components.append('adequate_description')
            validation_errors.append("Description is missing or too short")
        
        if not primary_metadata.tags or len(primary_metadata.tags) < 5:
            missing_components.append('sufficient_tags')
            validation_errors.append("Insufficient tags (minimum 5 required)")
        
        # Validate thumbnail alignments
        if not thumbnail_alignments:
            missing_components.append('thumbnail_alignments')
            validation_errors.append("No thumbnail alignments found")
        
        # Calculate quality scores
        quality_score = primary_metadata.confidence_score
        
        # Calculate synchronization score
        sync_factors = []
        if thumbnail_alignments:
            avg_visual_consistency = sum(align.visual_consistency_score for align in thumbnail_alignments) / len(thumbnail_alignments)
            sync_factors.append(avg_visual_consistency)
        
        if context.emotional_markers and context.visual_highlights:
            sync_factors.append(0.8)  # Good emotional-visual sync
        
        synchronization_score = sum(sync_factors) / len(sync_factors) if sync_factors else 0.5
        
        # Calculate AI Director alignment
        ai_alignment = 0.8 if hasattr(context, 'ai_director_plan') and context.ai_director_plan else 0.6
        
        # Generate recommendations
        if quality_score < self.validation_requirements['minimum_quality_score']:
            recommendations.append("Improve content analysis quality for better metadata generation")
        
        if synchronization_score < self.validation_requirements['minimum_sync_score']:
            recommendations.append("Enhance thumbnail-metadata synchronization")
        
        if ai_alignment < self.validation_requirements['minimum_ai_alignment']:
            recommendations.append("Integrate AI Director decisions for better alignment")
        
        if len(primary_metadata.tags) < 10:
            recommendations.append("Add more tags for better discoverability")
        
        # Determine completeness
        is_complete = (
            len(missing_components) == 0 and
            quality_score >= self.validation_requirements['minimum_quality_score'] and
            synchronization_score >= self.validation_requirements['minimum_sync_score']
        )
        
        return MetadataValidationResult(
            is_complete=is_complete,
            missing_components=missing_components,
            quality_score=quality_score,
            synchronization_score=synchronization_score,
            ai_director_alignment=ai_alignment,
            validation_errors=validation_errors,
            recommendations=recommendations
        )
    
    def _calculate_publish_readiness(self, primary_metadata: MetadataVariation,
                                   validation_result: MetadataValidationResult,
                                   ai_integration: Dict[str, Any]) -> float:
        """Calculate overall publish readiness score."""
        factors = [
            validation_result.quality_score * 0.3,
            validation_result.synchronization_score * 0.25,
            validation_result.ai_director_alignment * 0.2,
            primary_metadata.seo_score * 0.15,
            primary_metadata.estimated_ctr * 0.1
        ]
        
        base_score = sum(factors)
        
        # Apply penalties for missing components
        penalty = len(validation_result.missing_components) * 0.1
        
        # Apply bonus for completeness
        bonus = 0.1 if validation_result.is_complete else 0.0
        
        final_score = max(0.0, min(1.0, base_score - penalty + bonus))
        
        return final_score