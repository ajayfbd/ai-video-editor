"""
Thumbnail-Metadata Synchronizer

This module ensures thumbnail concepts are synchronized with metadata generation
for coordinated A/B testing and consistent messaging across all content assets.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .thumbnail_models import ThumbnailPackage, ThumbnailVariation, ThumbnailConcept
from ...core.content_context import ContentContext
from ...core.exceptions import ContentContextError, handle_errors


logger = logging.getLogger(__name__)


class SynchronizationError(ContentContextError):
    """Raised when thumbnail-metadata synchronization fails."""
    
    def __init__(self, sync_type: str, reason: Optional[str] = None, **kwargs):
        message = f"Synchronization failed: {sync_type}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, error_code="SYNCHRONIZATION_ERROR", **kwargs)
        self.sync_type = sync_type
        self.reason = reason


class ThumbnailMetadataSynchronizer:
    """
    Ensures thumbnail concepts align with metadata strategy for coordinated A/B testing.
    
    This class coordinates thumbnail generation with metadata creation to ensure
    consistent messaging, shared keyword usage, and synchronized A/B testing.
    """
    
    def __init__(self):
        """Initialize ThumbnailMetadataSynchronizer."""
        # Strategy mapping between thumbnails and metadata
        self.strategy_mapping = {
            "emotional": "emotional",
            "curiosity": "curiosity_driven", 
            "authority": "educational",
            "urgency": "urgency",
            "educational": "educational"
        }
        
        # Synchronization weights for different factors
        self.sync_weights = {
            "keyword_overlap": 0.3,
            "emotional_alignment": 0.25,
            "strategy_consistency": 0.25,
            "hook_title_similarity": 0.2
        }
        
        logger.info("ThumbnailMetadataSynchronizer initialized")
    
    @handle_errors(logger)
    def synchronize_concepts(
        self, 
        thumbnail_package: ThumbnailPackage, 
        context: ContentContext
    ) -> Dict[str, Any]:
        """
        Ensure thumbnail concepts align with metadata strategy.
        
        Args:
            thumbnail_package: ThumbnailPackage to synchronize
            context: ContentContext with metadata variations
            
        Returns:
            Synchronization analysis and recommendations
            
        Raises:
            SynchronizationError: If synchronization fails
        """
        try:
            start_time = time.time()
            
            # Get metadata variations from context
            metadata_variations = context.metadata_variations
            
            if not metadata_variations:
                logger.warning("No metadata variations found, creating basic synchronization")
                return self._create_basic_synchronization(thumbnail_package, context)
            
            # Analyze synchronization between thumbnails and metadata
            sync_analysis = self._analyze_synchronization(
                thumbnail_package.variations, 
                metadata_variations, 
                context
            )
            
            # Create synchronization mappings
            sync_mappings = self._create_synchronization_mappings(
                thumbnail_package.variations,
                metadata_variations,
                sync_analysis
            )
            
            # Generate recommendations for improvement
            recommendations = self._generate_sync_recommendations(
                sync_analysis, 
                thumbnail_package, 
                context
            )
            
            # Update thumbnail package with synchronization data
            thumbnail_package.synchronized_metadata = {
                "mappings": sync_mappings,
                "analysis": sync_analysis,
                "recommendations": recommendations,
                "sync_score": sync_analysis.get("overall_sync_score", 0.0),
                "synchronized_at": datetime.now().isoformat()
            }
            
            processing_time = time.time() - start_time
            logger.info(f"Synchronized thumbnails with metadata in {processing_time:.2f}s")
            
            return thumbnail_package.synchronized_metadata
            
        except Exception as e:
            logger.error(f"Thumbnail-metadata synchronization failed: {str(e)}")
            raise SynchronizationError("concept_synchronization", str(e), context)
    
    @handle_errors(logger)
    def create_ab_testing_config(
        self, 
        thumbnail_package: ThumbnailPackage, 
        metadata_variations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create coordinated A/B testing configuration.
        
        Args:
            thumbnail_package: ThumbnailPackage with variations
            metadata_variations: List of metadata variations
            
        Returns:
            A/B testing configuration dictionary
        """
        try:
            # Create test groups combining thumbnails and metadata
            test_groups = self._create_test_groups(
                thumbnail_package.variations,
                metadata_variations
            )
            
            # Calculate expected performance for each group
            performance_predictions = self._predict_group_performance(test_groups)
            
            # Create testing schedule and allocation
            testing_config = {
                "test_groups": test_groups,
                "performance_predictions": performance_predictions,
                "allocation_strategy": self._create_allocation_strategy(test_groups),
                "success_metrics": self._define_success_metrics(),
                "testing_duration": self._calculate_testing_duration(test_groups),
                "statistical_significance": {
                    "confidence_level": 0.95,
                    "minimum_sample_size": 1000,
                    "expected_effect_size": 0.1
                },
                "created_at": datetime.now().isoformat()
            }
            
            # Update thumbnail package
            thumbnail_package.a_b_testing_config = testing_config
            
            logger.info(f"Created A/B testing config with {len(test_groups)} test groups")
            
            return testing_config
            
        except Exception as e:
            logger.error(f"A/B testing config creation failed: {str(e)}")
            raise SynchronizationError("ab_testing_config", str(e))
    
    @handle_errors(logger)
    def validate_synchronization(
        self, 
        thumbnail_package: ThumbnailPackage, 
        context: ContentContext
    ) -> bool:
        """
        Validate that thumbnails and metadata are properly synchronized.
        
        Args:
            thumbnail_package: ThumbnailPackage to validate
            context: ContentContext with metadata
            
        Returns:
            True if synchronization is valid, False otherwise
        """
        try:
            validation_results = {
                "keyword_consistency": self._validate_keyword_consistency(thumbnail_package, context),
                "emotional_alignment": self._validate_emotional_alignment(thumbnail_package, context),
                "strategy_mapping": self._validate_strategy_mapping(thumbnail_package, context),
                "hook_title_coherence": self._validate_hook_title_coherence(thumbnail_package, context)
            }
            
            # Calculate overall validation score
            total_score = sum(validation_results.values())
            average_score = total_score / len(validation_results)
            
            # Log validation results
            logger.info(f"Synchronization validation score: {average_score:.2f}")
            for check, score in validation_results.items():
                logger.debug(f"  {check}: {score:.2f}")
            
            # Consider valid if average score > 0.7
            is_valid = average_score > 0.7
            
            # Store validation results in package
            if hasattr(thumbnail_package, 'synchronized_metadata'):
                thumbnail_package.synchronized_metadata["validation"] = {
                    "is_valid": is_valid,
                    "scores": validation_results,
                    "overall_score": average_score,
                    "validated_at": datetime.now().isoformat()
                }
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Synchronization validation failed: {str(e)}")
            return False
    
    def _analyze_synchronization(
        self, 
        thumbnail_variations: List[ThumbnailVariation],
        metadata_variations: List[Dict[str, Any]],
        context: ContentContext
    ) -> Dict[str, Any]:
        """Analyze synchronization between thumbnails and metadata."""
        analysis = {
            "keyword_overlap_scores": [],
            "emotional_alignment_scores": [],
            "strategy_consistency_scores": [],
            "hook_title_similarity_scores": [],
            "overall_sync_score": 0.0,
            "best_combinations": [],
            "sync_issues": []
        }
        
        # Analyze each thumbnail-metadata combination
        for thumb_var in thumbnail_variations:
            for meta_var in metadata_variations:
                combo_analysis = self._analyze_combination(thumb_var, meta_var, context)
                
                analysis["keyword_overlap_scores"].append(combo_analysis["keyword_overlap"])
                analysis["emotional_alignment_scores"].append(combo_analysis["emotional_alignment"])
                analysis["strategy_consistency_scores"].append(combo_analysis["strategy_consistency"])
                analysis["hook_title_similarity_scores"].append(combo_analysis["hook_title_similarity"])
                
                # Track best combinations
                if combo_analysis["overall_score"] > 0.8:
                    analysis["best_combinations"].append({
                        "thumbnail_id": thumb_var.variation_id,
                        "metadata_id": meta_var.get("variation_id", "unknown"),
                        "score": combo_analysis["overall_score"],
                        "strengths": combo_analysis["strengths"]
                    })
                
                # Track sync issues
                if combo_analysis["overall_score"] < 0.5:
                    analysis["sync_issues"].append({
                        "thumbnail_id": thumb_var.variation_id,
                        "metadata_id": meta_var.get("variation_id", "unknown"),
                        "issues": combo_analysis["issues"]
                    })
        
        # Calculate overall synchronization score
        if analysis["keyword_overlap_scores"]:
            analysis["overall_sync_score"] = (
                sum(analysis["keyword_overlap_scores"]) * self.sync_weights["keyword_overlap"] +
                sum(analysis["emotional_alignment_scores"]) * self.sync_weights["emotional_alignment"] +
                sum(analysis["strategy_consistency_scores"]) * self.sync_weights["strategy_consistency"] +
                sum(analysis["hook_title_similarity_scores"]) * self.sync_weights["hook_title_similarity"]
            ) / len(analysis["keyword_overlap_scores"])
        
        return analysis
    
    def _analyze_combination(
        self, 
        thumbnail_var: ThumbnailVariation, 
        metadata_var: Dict[str, Any], 
        context: ContentContext
    ) -> Dict[str, Any]:
        """Analyze synchronization for a specific thumbnail-metadata combination."""
        analysis = {
            "keyword_overlap": 0.0,
            "emotional_alignment": 0.0,
            "strategy_consistency": 0.0,
            "hook_title_similarity": 0.0,
            "overall_score": 0.0,
            "strengths": [],
            "issues": []
        }
        
        # Keyword overlap analysis
        thumb_keywords = self._extract_thumbnail_keywords(thumbnail_var, context)
        meta_keywords = self._extract_metadata_keywords(metadata_var)
        
        if thumb_keywords and meta_keywords:
            overlap = len(set(thumb_keywords) & set(meta_keywords))
            total_unique = len(set(thumb_keywords) | set(meta_keywords))
            analysis["keyword_overlap"] = overlap / total_unique if total_unique > 0 else 0.0
            
            if analysis["keyword_overlap"] > 0.5:
                analysis["strengths"].append("Strong keyword overlap")
            elif analysis["keyword_overlap"] < 0.2:
                analysis["issues"].append("Poor keyword overlap")
        
        # Emotional alignment analysis
        thumb_emotion = thumbnail_var.concept.emotional_peak.emotion
        meta_strategy = metadata_var.get("strategy", "unknown")
        
        analysis["emotional_alignment"] = self._calculate_emotional_alignment(thumb_emotion, meta_strategy)
        
        if analysis["emotional_alignment"] > 0.7:
            analysis["strengths"].append("Excellent emotional alignment")
        elif analysis["emotional_alignment"] < 0.4:
            analysis["issues"].append("Poor emotional alignment")
        
        # Strategy consistency analysis
        thumb_strategy = thumbnail_var.concept.strategy
        mapped_meta_strategy = self.strategy_mapping.get(thumb_strategy, "unknown")
        
        if mapped_meta_strategy == meta_strategy:
            analysis["strategy_consistency"] = 1.0
            analysis["strengths"].append("Perfect strategy consistency")
        elif self._are_compatible_strategies(thumb_strategy, meta_strategy):
            analysis["strategy_consistency"] = 0.7
            analysis["strengths"].append("Compatible strategies")
        else:
            analysis["strategy_consistency"] = 0.3
            analysis["issues"].append("Incompatible strategies")
        
        # Hook-title similarity analysis
        hook_text = thumbnail_var.concept.hook_text.lower()
        title_text = metadata_var.get("title", "").lower()
        
        analysis["hook_title_similarity"] = self._calculate_text_similarity(hook_text, title_text)
        
        if analysis["hook_title_similarity"] > 0.6:
            analysis["strengths"].append("Good hook-title coherence")
        elif analysis["hook_title_similarity"] < 0.3:
            analysis["issues"].append("Poor hook-title coherence")
        
        # Calculate overall score
        analysis["overall_score"] = (
            analysis["keyword_overlap"] * self.sync_weights["keyword_overlap"] +
            analysis["emotional_alignment"] * self.sync_weights["emotional_alignment"] +
            analysis["strategy_consistency"] * self.sync_weights["strategy_consistency"] +
            analysis["hook_title_similarity"] * self.sync_weights["hook_title_similarity"]
        )
        
        return analysis
    
    def _create_synchronization_mappings(
        self,
        thumbnail_variations: List[ThumbnailVariation],
        metadata_variations: List[Dict[str, Any]],
        sync_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create optimal mappings between thumbnails and metadata."""
        mappings = {
            "optimal_pairs": [],
            "strategy_groups": {},
            "fallback_pairs": []
        }
        
        # Find optimal pairs from best combinations
        best_combinations = sync_analysis.get("best_combinations", [])
        
        for combo in best_combinations:
            mappings["optimal_pairs"].append({
                "thumbnail_id": combo["thumbnail_id"],
                "metadata_id": combo["metadata_id"],
                "sync_score": combo["score"],
                "recommended": True
            })
        
        # Group by strategy
        for thumb_var in thumbnail_variations:
            strategy = thumb_var.concept.strategy
            if strategy not in mappings["strategy_groups"]:
                mappings["strategy_groups"][strategy] = {
                    "thumbnails": [],
                    "compatible_metadata": []
                }
            
            mappings["strategy_groups"][strategy]["thumbnails"].append(thumb_var.variation_id)
            
            # Find compatible metadata
            mapped_strategy = self.strategy_mapping.get(strategy, strategy)
            for meta_var in metadata_variations:
                if meta_var.get("strategy") == mapped_strategy:
                    mappings["strategy_groups"][strategy]["compatible_metadata"].append(
                        meta_var.get("variation_id", "unknown")
                    )
        
        # Create fallback pairs for unmatched items
        used_thumbnails = {pair["thumbnail_id"] for pair in mappings["optimal_pairs"]}
        used_metadata = {pair["metadata_id"] for pair in mappings["optimal_pairs"]}
        
        unused_thumbnails = [tv for tv in thumbnail_variations if tv.variation_id not in used_thumbnails]
        unused_metadata = [mv for mv in metadata_variations if mv.get("variation_id") not in used_metadata]
        
        for i, thumb_var in enumerate(unused_thumbnails):
            if i < len(unused_metadata):
                mappings["fallback_pairs"].append({
                    "thumbnail_id": thumb_var.variation_id,
                    "metadata_id": unused_metadata[i].get("variation_id", "unknown"),
                    "sync_score": 0.5,  # Default moderate score
                    "recommended": False
                })
        
        return mappings
    
    def _create_test_groups(
        self,
        thumbnail_variations: List[ThumbnailVariation],
        metadata_variations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create A/B test groups combining thumbnails and metadata."""
        test_groups = []
        
        # Create groups based on strategy alignment
        for i, thumb_var in enumerate(thumbnail_variations):
            # Find best matching metadata variation
            best_meta = None
            best_score = 0.0
            
            thumb_strategy = thumb_var.concept.strategy
            mapped_strategy = self.strategy_mapping.get(thumb_strategy, thumb_strategy)
            
            for meta_var in metadata_variations:
                if meta_var.get("strategy") == mapped_strategy:
                    score = meta_var.get("confidence_score", 0.5)
                    if score > best_score:
                        best_score = score
                        best_meta = meta_var
            
            # Use first metadata as fallback
            if not best_meta and metadata_variations:
                best_meta = metadata_variations[0]
            
            if best_meta:
                test_groups.append({
                    "group_id": f"group_{i+1}",
                    "thumbnail": {
                        "variation_id": thumb_var.variation_id,
                        "strategy": thumb_var.concept.strategy,
                        "confidence": thumb_var.confidence_score,
                        "estimated_ctr": thumb_var.estimated_ctr
                    },
                    "metadata": {
                        "variation_id": best_meta.get("variation_id", f"meta_{i+1}"),
                        "strategy": best_meta.get("strategy", "unknown"),
                        "title": best_meta.get("title", ""),
                        "confidence": best_meta.get("confidence_score", 0.5)
                    },
                    "expected_performance": {
                        "ctr_estimate": (thumb_var.estimated_ctr + best_meta.get("estimated_ctr", 0.1)) / 2,
                        "engagement_score": thumb_var.confidence_score * best_meta.get("confidence_score", 0.5),
                        "sync_quality": best_score
                    }
                })
        
        return test_groups
    
    def _predict_group_performance(self, test_groups: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict performance for each test group."""
        predictions = {
            "group_rankings": [],
            "expected_winner": None,
            "confidence_intervals": {},
            "risk_assessment": {}
        }
        
        # Rank groups by expected performance
        ranked_groups = sorted(
            test_groups, 
            key=lambda g: g["expected_performance"]["engagement_score"], 
            reverse=True
        )
        
        for i, group in enumerate(ranked_groups):
            group_id = group["group_id"]
            performance = group["expected_performance"]
            
            predictions["group_rankings"].append({
                "rank": i + 1,
                "group_id": group_id,
                "expected_ctr": performance["ctr_estimate"],
                "engagement_score": performance["engagement_score"],
                "win_probability": max(0.1, 0.9 - (i * 0.15))  # Decreasing probability
            })
            
            # Calculate confidence intervals (simplified)
            base_ctr = performance["ctr_estimate"]
            predictions["confidence_intervals"][group_id] = {
                "lower_bound": max(0.01, base_ctr - 0.02),
                "upper_bound": min(0.30, base_ctr + 0.02),
                "confidence_level": 0.95
            }
            
            # Risk assessment
            predictions["risk_assessment"][group_id] = {
                "performance_risk": "low" if performance["sync_quality"] > 0.7 else "medium",
                "strategy_risk": "low" if performance["engagement_score"] > 0.6 else "high",
                "overall_risk": "low" if i < 2 else "medium"
            }
        
        # Set expected winner
        if predictions["group_rankings"]:
            predictions["expected_winner"] = predictions["group_rankings"][0]["group_id"]
        
        return predictions
    
    def _create_allocation_strategy(self, test_groups: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create traffic allocation strategy for A/B testing."""
        num_groups = len(test_groups)
        
        if num_groups <= 2:
            # Simple 50/50 split
            allocation = {group["group_id"]: 50.0 for group in test_groups}
        elif num_groups <= 4:
            # Equal allocation
            allocation = {group["group_id"]: 100.0 / num_groups for group in test_groups}
        else:
            # Weighted allocation favoring top performers
            total_weight = sum(range(1, num_groups + 1))
            allocation = {}
            
            for i, group in enumerate(test_groups):
                weight = num_groups - i  # Higher weight for better performers
                percentage = (weight / total_weight) * 100
                allocation[group["group_id"]] = percentage
        
        return {
            "allocation_percentages": allocation,
            "allocation_method": "equal" if num_groups <= 4 else "weighted",
            "minimum_sample_per_group": max(100, 1000 // num_groups),
            "ramp_up_strategy": {
                "initial_percentage": 10.0,
                "ramp_up_duration_hours": 24,
                "full_allocation_after_hours": 48
            }
        }
    
    def _define_success_metrics(self) -> Dict[str, Any]:
        """Define success metrics for A/B testing."""
        return {
            "primary_metrics": [
                {
                    "name": "click_through_rate",
                    "description": "Thumbnail CTR",
                    "target_improvement": 0.05,
                    "minimum_significance": 0.95
                },
                {
                    "name": "watch_time",
                    "description": "Average watch time",
                    "target_improvement": 0.10,
                    "minimum_significance": 0.90
                }
            ],
            "secondary_metrics": [
                {
                    "name": "engagement_rate",
                    "description": "Likes, comments, shares",
                    "target_improvement": 0.15,
                    "minimum_significance": 0.85
                },
                {
                    "name": "subscriber_conversion",
                    "description": "New subscribers from video",
                    "target_improvement": 0.20,
                    "minimum_significance": 0.80
                }
            ],
            "guardrail_metrics": [
                {
                    "name": "bounce_rate",
                    "description": "Viewers leaving quickly",
                    "maximum_acceptable": 0.70,
                    "alert_threshold": 0.75
                }
            ]
        }
    
    def _calculate_testing_duration(self, test_groups: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate optimal testing duration."""
        num_groups = len(test_groups)
        
        # Base duration increases with number of groups
        base_days = 7 + (num_groups - 2) * 2
        
        return {
            "minimum_duration_days": base_days,
            "recommended_duration_days": base_days + 3,
            "maximum_duration_days": base_days + 7,
            "early_stopping_criteria": {
                "minimum_runtime_days": 3,
                "significance_threshold": 0.99,
                "minimum_sample_size": 1000
            },
            "extension_criteria": {
                "inconclusive_results": True,
                "close_performance": True,
                "insufficient_sample": True
            }
        }
    
    def _extract_thumbnail_keywords(self, thumbnail_var: ThumbnailVariation, context: ContentContext) -> List[str]:
        """Extract keywords from thumbnail concept."""
        keywords = []
        
        # Extract from hook text
        hook_words = thumbnail_var.concept.hook_text.lower().split()
        keywords.extend([word.strip("!?.,") for word in hook_words if len(word) > 2])
        
        # Extract from visual elements
        keywords.extend(thumbnail_var.concept.visual_elements)
        
        # Extract from emotional context
        emotion_context = thumbnail_var.concept.emotional_peak.context.lower()
        context_words = emotion_context.split()
        keywords.extend([word for word in context_words if len(word) > 3])
        
        # Add trending keywords if available
        if context.trending_keywords:
            keywords.extend(context.trending_keywords.primary_keywords[:3])
        
        return list(set(keywords))  # Remove duplicates
    
    def _extract_metadata_keywords(self, metadata_var: Dict[str, Any]) -> List[str]:
        """Extract keywords from metadata variation."""
        keywords = []
        
        # Extract from title
        title = metadata_var.get("title", "").lower()
        title_words = title.split()
        keywords.extend([word.strip("!?.,") for word in title_words if len(word) > 2])
        
        # Extract from tags
        tags = metadata_var.get("tags", [])
        keywords.extend([tag.lower() for tag in tags])
        
        # Extract from description (first few words)
        description = metadata_var.get("description", "").lower()
        desc_words = description.split()[:10]  # First 10 words
        keywords.extend([word for word in desc_words if len(word) > 3])
        
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_emotional_alignment(self, thumbnail_emotion: str, metadata_strategy: str) -> float:
        """Calculate emotional alignment between thumbnail and metadata."""
        # Emotion-strategy alignment matrix
        alignment_matrix = {
            "excitement": {"emotional": 1.0, "curiosity_driven": 0.7, "urgency": 0.8, "educational": 0.4},
            "curiosity": {"curiosity_driven": 1.0, "emotional": 0.6, "educational": 0.8, "urgency": 0.3},
            "confidence": {"educational": 1.0, "emotional": 0.5, "curiosity_driven": 0.4, "urgency": 0.3},
            "surprise": {"emotional": 1.0, "curiosity_driven": 0.9, "urgency": 0.6, "educational": 0.3},
            "urgency": {"urgency": 1.0, "emotional": 0.7, "curiosity_driven": 0.4, "educational": 0.2}
        }
        
        emotion_lower = thumbnail_emotion.lower()
        strategy_alignments = alignment_matrix.get(emotion_lower, {})
        
        return strategy_alignments.get(metadata_strategy, 0.5)  # Default moderate alignment
    
    def _are_compatible_strategies(self, thumb_strategy: str, meta_strategy: str) -> bool:
        """Check if thumbnail and metadata strategies are compatible."""
        compatibility_matrix = {
            "emotional": ["emotional", "curiosity_driven", "urgency"],
            "curiosity": ["curiosity_driven", "educational", "emotional"],
            "authority": ["educational", "emotional"],
            "urgency": ["urgency", "emotional"],
            "educational": ["educational", "curiosity_driven"]
        }
        
        compatible_strategies = compatibility_matrix.get(thumb_strategy, [])
        return meta_strategy in compatible_strategies
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _validate_keyword_consistency(self, thumbnail_package: ThumbnailPackage, context: ContentContext) -> float:
        """Validate keyword consistency between thumbnails and metadata."""
        if not context.metadata_variations:
            return 0.5
        
        consistency_scores = []
        
        for thumb_var in thumbnail_package.variations:
            thumb_keywords = self._extract_thumbnail_keywords(thumb_var, context)
            
            for meta_var in context.metadata_variations:
                meta_keywords = self._extract_metadata_keywords(meta_var)
                
                if thumb_keywords and meta_keywords:
                    overlap = len(set(thumb_keywords) & set(meta_keywords))
                    total = len(set(thumb_keywords) | set(meta_keywords))
                    consistency_scores.append(overlap / total if total > 0 else 0.0)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.5
    
    def _validate_emotional_alignment(self, thumbnail_package: ThumbnailPackage, context: ContentContext) -> float:
        """Validate emotional alignment between thumbnails and metadata."""
        if not context.metadata_variations:
            return 0.5
        
        alignment_scores = []
        
        for thumb_var in thumbnail_package.variations:
            thumb_emotion = thumb_var.concept.emotional_peak.emotion
            
            for meta_var in context.metadata_variations:
                meta_strategy = meta_var.get("strategy", "unknown")
                alignment = self._calculate_emotional_alignment(thumb_emotion, meta_strategy)
                alignment_scores.append(alignment)
        
        return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.5
    
    def _validate_strategy_mapping(self, thumbnail_package: ThumbnailPackage, context: ContentContext) -> float:
        """Validate strategy mapping consistency."""
        if not context.metadata_variations:
            return 0.5
        
        mapping_scores = []
        
        for thumb_var in thumbnail_package.variations:
            thumb_strategy = thumb_var.concept.strategy
            mapped_strategy = self.strategy_mapping.get(thumb_strategy, thumb_strategy)
            
            # Check if any metadata variation matches the mapped strategy
            has_match = any(
                meta_var.get("strategy") == mapped_strategy 
                for meta_var in context.metadata_variations
            )
            
            mapping_scores.append(1.0 if has_match else 0.0)
        
        return sum(mapping_scores) / len(mapping_scores) if mapping_scores else 0.5
    
    def _validate_hook_title_coherence(self, thumbnail_package: ThumbnailPackage, context: ContentContext) -> float:
        """Validate coherence between hook text and titles."""
        if not context.metadata_variations:
            return 0.5
        
        coherence_scores = []
        
        for thumb_var in thumbnail_package.variations:
            hook_text = thumb_var.concept.hook_text
            
            for meta_var in context.metadata_variations:
                title = meta_var.get("title", "")
                similarity = self._calculate_text_similarity(hook_text, title)
                coherence_scores.append(similarity)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5
    
    def _create_basic_synchronization(self, thumbnail_package: ThumbnailPackage, context: ContentContext) -> Dict[str, Any]:
        """Create basic synchronization when no metadata variations are available."""
        return {
            "mappings": {
                "optimal_pairs": [],
                "strategy_groups": {},
                "fallback_pairs": []
            },
            "analysis": {
                "overall_sync_score": 0.5,
                "best_combinations": [],
                "sync_issues": ["No metadata variations available for synchronization"]
            },
            "recommendations": [
                "Generate metadata variations before thumbnail synchronization",
                "Ensure ContentContext contains metadata_variations",
                "Consider running metadata generation first"
            ],
            "sync_score": 0.5,
            "synchronized_at": datetime.now().isoformat()
        }
    
    def _generate_sync_recommendations(
        self, 
        sync_analysis: Dict[str, Any], 
        thumbnail_package: ThumbnailPackage, 
        context: ContentContext
    ) -> List[str]:
        """Generate recommendations for improving synchronization."""
        recommendations = []
        
        overall_score = sync_analysis.get("overall_sync_score", 0.0)
        
        if overall_score < 0.5:
            recommendations.append("Consider regenerating thumbnails with better metadata alignment")
            recommendations.append("Review emotional peak selection for better strategy matching")
        
        if len(sync_analysis.get("sync_issues", [])) > 0:
            recommendations.append("Address identified synchronization issues")
            recommendations.append("Consider adjusting thumbnail strategies to match metadata")
        
        if len(sync_analysis.get("best_combinations", [])) < 2:
            recommendations.append("Generate additional thumbnail variations for better A/B testing")
            recommendations.append("Diversify thumbnail strategies to match metadata variations")
        
        keyword_scores = sync_analysis.get("keyword_overlap_scores", [])
        if keyword_scores and sum(keyword_scores) / len(keyword_scores) < 0.3:
            recommendations.append("Improve keyword consistency between thumbnails and metadata")
            recommendations.append("Use trending keywords in both thumbnail text and metadata")
        
        return recommendations