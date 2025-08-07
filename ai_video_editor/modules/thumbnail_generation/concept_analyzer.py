"""
Thumbnail Concept Analyzer

This module analyzes visual highlights and emotional peaks to generate
compelling thumbnail concepts with hook text and visual strategies.
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .thumbnail_models import ThumbnailConcept, ThumbnailGenerationStats
from ...core.content_context import ContentContext, EmotionalPeak, VisualHighlight
from ...core.exceptions import ContentContextError, handle_errors
from ...modules.intelligence.gemini_client import GeminiClient


logger = logging.getLogger(__name__)


class ConceptAnalysisError(ContentContextError):
    """Raised when thumbnail concept analysis fails."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="CONCEPT_ANALYSIS_ERROR", **kwargs)


class ThumbnailConceptAnalyzer:
    """
    Analyzes visual highlights and emotional peaks to generate thumbnail concepts.
    
    This class extracts the most compelling visual moments and emotional peaks
    from content analysis to create thumbnail concepts that maximize click-through rates.
    """
    
    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize ThumbnailConceptAnalyzer.
        
        Args:
            gemini_client: Gemini client for AI-powered concept analysis
        """
        self.gemini_client = gemini_client
        self.stats = ThumbnailGenerationStats()
        
        # Thumbnail strategies and their characteristics
        self.strategies = {
            "emotional": {
                "description": "High-emotion moments with expressive faces",
                "text_style": {"bold": True, "color": "#FF4444", "size": "large"},
                "background_style": "dynamic_gradient",
                "target_emotions": ["excitement", "surprise", "curiosity"]
            },
            "curiosity": {
                "description": "Intriguing moments that make viewers want to know more",
                "text_style": {"bold": True, "color": "#4444FF", "size": "medium"},
                "background_style": "question_mark_overlay",
                "target_emotions": ["curiosity", "confusion", "interest"]
            },
            "authority": {
                "description": "Professional moments showing expertise",
                "text_style": {"bold": False, "color": "#333333", "size": "medium"},
                "background_style": "clean_professional",
                "target_emotions": ["confidence", "trust", "authority"]
            },
            "urgency": {
                "description": "Time-sensitive or important moments",
                "text_style": {"bold": True, "color": "#FF8800", "size": "large"},
                "background_style": "urgent_arrows",
                "target_emotions": ["urgency", "excitement", "concern"]
            },
            "educational": {
                "description": "Learning moments with clear explanations",
                "text_style": {"bold": False, "color": "#2E7D32", "size": "medium"},
                "background_style": "educational_icons",
                "target_emotions": ["understanding", "clarity", "achievement"]
            }
        }
        
        logger.info("ThumbnailConceptAnalyzer initialized with 5 strategies")
    
    @handle_errors(logger)
    async def analyze_thumbnail_concepts(self, context: ContentContext) -> List[ThumbnailConcept]:
        """
        Analyze visual highlights and emotional peaks to generate thumbnail concepts.
        
        Args:
            context: ContentContext with visual highlights and emotional markers
            
        Returns:
            List of ThumbnailConcept objects
            
        Raises:
            ConceptAnalysisError: If concept analysis fails
        """
        start_time = time.time()
        
        try:
            # Get best visual highlights and emotional peaks
            visual_highlights = context.get_best_visual_highlights(count=10)
            emotional_peaks = context.get_top_emotional_peaks(count=8)
            
            if not visual_highlights:
                logger.warning("No visual highlights found, using fallback concept generation")
                return await self._generate_fallback_concepts(context)
            
            if not emotional_peaks:
                logger.warning("No emotional peaks found, using neutral emotional context")
                emotional_peaks = [self._create_neutral_emotional_peak()]
            
            # Generate concepts by combining visual highlights with emotional peaks
            concepts = []
            
            # Strategy 1: Direct pairing of best highlights with strongest emotions
            for i, highlight in enumerate(visual_highlights[:5]):
                emotion_idx = min(i, len(emotional_peaks) - 1)
                emotional_peak = emotional_peaks[emotion_idx]
                
                concept = await self._create_concept_from_highlight_and_emotion(
                    highlight, emotional_peak, context
                )
                if concept:
                    concepts.append(concept)
            
            # Strategy 2: Generate concepts for each thumbnail strategy
            for strategy_name in self.strategies.keys():
                concept = await self._generate_strategy_concept(
                    strategy_name, visual_highlights, emotional_peaks, context
                )
                if concept and not self._is_duplicate_concept(concept, concepts):
                    concepts.append(concept)
            
            # Filter and rank concepts
            concepts = self._filter_and_rank_concepts(concepts, context)
            
            # Update statistics
            self.stats.concepts_analyzed = len(concepts)
            processing_time = time.time() - start_time
            
            logger.info(f"Generated {len(concepts)} thumbnail concepts in {processing_time:.2f}s")
            
            return concepts[:8]  # Return top 8 concepts
            
        except Exception as e:
            logger.error(f"Thumbnail concept analysis failed: {str(e)}")
            raise ConceptAnalysisError(f"Failed to analyze thumbnail concepts: {str(e)}", context)
    
    @handle_errors(logger)
    async def generate_hook_text(self, emotional_peak: EmotionalPeak, context: ContentContext, strategy: str = "emotional") -> str:
        """
        Generate compelling hook text based on emotional analysis.
        
        Args:
            emotional_peak: Emotional peak to base hook text on
            context: ContentContext for additional insights
            strategy: Thumbnail strategy to align with
            
        Returns:
            Generated hook text string
        """
        try:
            # Create prompt for hook text generation
            prompt = self._create_hook_text_prompt(emotional_peak, context, strategy)
            
            # Get AI-generated hook text
            response = await self.gemini_client.generate_content(
                prompt=prompt,
                max_tokens=100,
                temperature=0.8
            )
            
            if response and response.content:
                hook_text = self._extract_hook_text(response.content)
                return self._validate_and_clean_hook_text(hook_text)
            else:
                logger.warning("AI hook text generation failed, using fallback")
                return self._generate_fallback_hook_text(emotional_peak, strategy)
                
        except Exception as e:
            logger.warning(f"Hook text generation failed: {str(e)}, using fallback")
            return self._generate_fallback_hook_text(emotional_peak, strategy)
    
    @handle_errors(logger)
    def score_thumbnail_potential(self, concept: ThumbnailConcept, context: ContentContext) -> float:
        """
        Score thumbnail potential based on multiple factors.
        
        Args:
            concept: ThumbnailConcept to score
            context: ContentContext for additional insights
            
        Returns:
            Thumbnail potential score (0.0 to 1.0)
        """
        try:
            score = 0.0
            
            # Visual highlight score (40% weight)
            visual_score = concept.visual_highlight.thumbnail_potential
            score += visual_score * 0.4
            
            # Emotional intensity score (30% weight)
            emotional_score = concept.emotional_peak.intensity * concept.emotional_peak.confidence
            score += emotional_score * 0.3
            
            # Hook text quality score (20% weight)
            hook_score = self._score_hook_text_quality(concept.hook_text)
            score += hook_score * 0.2
            
            # Strategy alignment score (10% weight)
            strategy_score = self._score_strategy_alignment(concept, context)
            score += strategy_score * 0.1
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.warning(f"Thumbnail potential scoring failed: {str(e)}")
            return 0.5  # Default moderate score
    
    async def _create_concept_from_highlight_and_emotion(
        self, 
        highlight: VisualHighlight, 
        emotional_peak: EmotionalPeak, 
        context: ContentContext
    ) -> Optional[ThumbnailConcept]:
        """Create thumbnail concept from visual highlight and emotional peak."""
        try:
            # Determine best strategy for this combination
            strategy = self._determine_best_strategy(highlight, emotional_peak)
            strategy_config = self.strategies[strategy]
            
            # Generate hook text
            hook_text = await self.generate_hook_text(emotional_peak, context, strategy)
            
            # Create concept
            concept = ThumbnailConcept(
                concept_id="",  # Will be auto-generated
                visual_highlight=highlight,
                emotional_peak=emotional_peak,
                hook_text=hook_text,
                background_style=strategy_config["background_style"],
                text_style=strategy_config["text_style"],
                visual_elements=highlight.visual_elements,
                thumbnail_potential=0.0,  # Will be calculated
                strategy=strategy
            )
            
            # Calculate thumbnail potential
            concept.thumbnail_potential = self.score_thumbnail_potential(concept, context)
            
            return concept
            
        except Exception as e:
            logger.warning(f"Failed to create concept from highlight and emotion: {str(e)}")
            return None
    
    async def _generate_strategy_concept(
        self, 
        strategy_name: str, 
        highlights: List[VisualHighlight], 
        emotions: List[EmotionalPeak], 
        context: ContentContext
    ) -> Optional[ThumbnailConcept]:
        """Generate concept optimized for specific strategy."""
        try:
            strategy_config = self.strategies[strategy_name]
            target_emotions = strategy_config["target_emotions"]
            
            # Find best emotional peak for this strategy
            best_emotion = None
            for emotion in emotions:
                if emotion.emotion in target_emotions:
                    best_emotion = emotion
                    break
            
            if not best_emotion:
                best_emotion = emotions[0] if emotions else self._create_neutral_emotional_peak()
            
            # Find best visual highlight for this strategy
            best_highlight = self._find_best_highlight_for_strategy(highlights, strategy_name)
            
            if not best_highlight:
                return None
            
            # Generate hook text for strategy
            hook_text = await self.generate_hook_text(best_emotion, context, strategy_name)
            
            # Create concept
            concept = ThumbnailConcept(
                concept_id="",
                visual_highlight=best_highlight,
                emotional_peak=best_emotion,
                hook_text=hook_text,
                background_style=strategy_config["background_style"],
                text_style=strategy_config["text_style"],
                visual_elements=best_highlight.visual_elements,
                thumbnail_potential=0.0,
                strategy=strategy_name
            )
            
            concept.thumbnail_potential = self.score_thumbnail_potential(concept, context)
            
            return concept
            
        except Exception as e:
            logger.warning(f"Failed to generate {strategy_name} strategy concept: {str(e)}")
            return None
    
    def _determine_best_strategy(self, highlight: VisualHighlight, emotional_peak: EmotionalPeak) -> str:
        """Determine best thumbnail strategy for highlight and emotion combination."""
        emotion = emotional_peak.emotion.lower()
        intensity = emotional_peak.intensity
        
        # High intensity emotions
        if intensity > 0.8:
            if emotion in ["excitement", "surprise", "shock"]:
                return "emotional"
            elif emotion in ["curiosity", "confusion"]:
                return "curiosity"
            elif emotion in ["urgency", "concern", "worry"]:
                return "urgency"
        
        # Medium intensity emotions
        elif intensity > 0.5:
            if emotion in ["confidence", "authority", "trust"]:
                return "authority"
            elif emotion in ["understanding", "clarity", "learning"]:
                return "educational"
            elif emotion in ["curiosity", "interest"]:
                return "curiosity"
        
        # Default to emotional strategy
        return "emotional"
    
    def _find_best_highlight_for_strategy(self, highlights: List[VisualHighlight], strategy: str) -> Optional[VisualHighlight]:
        """Find best visual highlight for specific strategy."""
        if not highlights:
            return None
        
        # Strategy-specific preferences
        if strategy == "authority":
            # Prefer highlights with professional visual elements
            for highlight in highlights:
                if any(element in ["professional_setting", "presentation", "charts"] 
                      for element in highlight.visual_elements):
                    return highlight
        
        elif strategy == "educational":
            # Prefer highlights with educational visual elements
            for highlight in highlights:
                if any(element in ["charts", "graphs", "diagrams", "text_overlay"] 
                      for element in highlight.visual_elements):
                    return highlight
        
        elif strategy == "emotional":
            # Prefer highlights with faces and high thumbnail potential
            for highlight in highlights:
                if highlight.faces and highlight.thumbnail_potential > 0.7:
                    return highlight
        
        # Return highest potential highlight as fallback
        return max(highlights, key=lambda h: h.thumbnail_potential)
    
    def _create_hook_text_prompt(self, emotional_peak: EmotionalPeak, context: ContentContext, strategy: str) -> str:
        """Create prompt for AI hook text generation."""
        key_concepts = ", ".join(context.key_concepts[:3]) if context.key_concepts else "general content"
        
        return f"""
Generate a compelling thumbnail hook text for a {context.content_type.value} video about {key_concepts}.

Context:
- Emotional peak: {emotional_peak.emotion} (intensity: {emotional_peak.intensity:.1f})
- Emotional context: {emotional_peak.context}
- Strategy: {strategy}
- Target: YouTube thumbnail text that maximizes click-through rate

Requirements:
- Maximum 6 words
- Create curiosity or emotional response
- Match the {strategy} strategy
- Be specific to the emotional context: {emotional_peak.context}

Generate only the hook text, no explanation:
"""
    
    def _extract_hook_text(self, ai_response: str) -> str:
        """Extract clean hook text from AI response."""
        # Remove quotes, extra whitespace, and common prefixes
        text = ai_response.strip().strip('"\'')
        
        # Remove common AI response prefixes
        prefixes_to_remove = [
            "Hook text:", "Thumbnail text:", "Here's the hook text:",
            "Generated text:", "Text:", "Hook:"
        ]
        
        for prefix in prefixes_to_remove:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        return text
    
    def _validate_and_clean_hook_text(self, hook_text: str) -> str:
        """Validate and clean hook text."""
        # Limit to 6 words maximum
        words = hook_text.split()
        if len(words) > 6:
            hook_text = " ".join(words[:6])
        
        # Ensure it's not empty
        if not hook_text.strip():
            return "MUST WATCH!"
        
        # Capitalize appropriately
        return hook_text.upper() if len(hook_text) < 20 else hook_text.title()
    
    def _generate_fallback_hook_text(self, emotional_peak: EmotionalPeak, strategy: str) -> str:
        """Generate fallback hook text when AI generation fails."""
        emotion = emotional_peak.emotion.lower()
        
        fallback_texts = {
            "emotional": {
                "excitement": "AMAZING RESULTS!",
                "surprise": "YOU WON'T BELIEVE",
                "shock": "SHOCKING TRUTH",
                "default": "INCREDIBLE!"
            },
            "curiosity": {
                "curiosity": "WHAT HAPPENS?",
                "confusion": "THE MYSTERY",
                "interest": "FIND OUT WHY",
                "default": "SECRET REVEALED"
            },
            "authority": {
                "confidence": "EXPERT EXPLAINS",
                "trust": "PROVEN METHOD",
                "authority": "PROFESSIONAL GUIDE",
                "default": "EXPERT ADVICE"
            },
            "urgency": {
                "urgency": "ACT NOW!",
                "concern": "DON'T MISS",
                "worry": "URGENT UPDATE",
                "default": "TIME SENSITIVE"
            },
            "educational": {
                "understanding": "LEARN HOW",
                "clarity": "EXPLAINED SIMPLY",
                "learning": "MASTER THIS",
                "default": "STEP BY STEP"
            }
        }
        
        strategy_texts = fallback_texts.get(strategy, fallback_texts["emotional"])
        return strategy_texts.get(emotion, strategy_texts["default"])
    
    def _score_hook_text_quality(self, hook_text: str) -> float:
        """Score hook text quality based on various factors."""
        if not hook_text:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length scoring (3-6 words is optimal)
        word_count = len(hook_text.split())
        if 3 <= word_count <= 6:
            score += 0.2
        elif word_count < 3:
            score -= 0.1
        elif word_count > 8:
            score -= 0.2
        
        # Emotional words boost
        emotional_words = [
            "amazing", "incredible", "shocking", "secret", "revealed", "must", "watch",
            "unbelievable", "proven", "expert", "urgent", "now", "don't", "miss"
        ]
        
        text_lower = hook_text.lower()
        emotional_word_count = sum(1 for word in emotional_words if word in text_lower)
        score += min(0.2, emotional_word_count * 0.1)
        
        # All caps bonus (for short text)
        if hook_text.isupper() and len(hook_text) < 20:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _score_strategy_alignment(self, concept: ThumbnailConcept, context: ContentContext) -> float:
        """Score how well concept aligns with its strategy."""
        strategy_config = self.strategies.get(concept.strategy, {})
        target_emotions = strategy_config.get("target_emotions", [])
        
        # Check if emotional peak matches strategy
        if concept.emotional_peak.emotion in target_emotions:
            return 0.8
        elif concept.emotional_peak.intensity > 0.7:
            return 0.6
        else:
            return 0.4
    
    def _filter_and_rank_concepts(self, concepts: List[ThumbnailConcept], context: ContentContext) -> List[ThumbnailConcept]:
        """Filter and rank concepts by quality and diversity."""
        if not concepts:
            return []
        
        # Remove concepts with very low potential
        filtered_concepts = [c for c in concepts if c.thumbnail_potential > 0.3]
        
        # Ensure strategy diversity
        strategy_counts = {}
        diverse_concepts = []
        
        # Sort by potential first
        sorted_concepts = sorted(filtered_concepts, key=lambda c: c.thumbnail_potential, reverse=True)
        
        for concept in sorted_concepts:
            strategy = concept.strategy
            if strategy_counts.get(strategy, 0) < 2:  # Max 2 per strategy
                diverse_concepts.append(concept)
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return diverse_concepts
    
    def _is_duplicate_concept(self, concept: ThumbnailConcept, existing_concepts: List[ThumbnailConcept]) -> bool:
        """Check if concept is too similar to existing concepts."""
        for existing in existing_concepts:
            # Same strategy and similar timestamp
            if (concept.strategy == existing.strategy and 
                abs(concept.visual_highlight.timestamp - existing.visual_highlight.timestamp) < 10.0):
                return True
            
            # Very similar hook text
            if concept.hook_text.lower() == existing.hook_text.lower():
                return True
        
        return False
    
    def _create_neutral_emotional_peak(self) -> EmotionalPeak:
        """Create neutral emotional peak as fallback."""
        return EmotionalPeak(
            timestamp=0.0,
            emotion="neutral",
            intensity=0.5,
            confidence=0.8,
            context="general content"
        )
    
    async def _generate_fallback_concepts(self, context: ContentContext) -> List[ThumbnailConcept]:
        """Generate fallback concepts when no visual highlights are available."""
        logger.info("Generating fallback thumbnail concepts")
        
        concepts = []
        neutral_emotion = self._create_neutral_emotional_peak()
        
        # Create basic visual highlight
        fallback_highlight = VisualHighlight(
            timestamp=0.0,
            description="General content frame",
            faces=[],
            visual_elements=["general_content"],
            thumbnail_potential=0.6
        )
        
        # Generate one concept per strategy
        for strategy_name in ["emotional", "curiosity", "authority"]:
            hook_text = self._generate_fallback_hook_text(neutral_emotion, strategy_name)
            strategy_config = self.strategies[strategy_name]
            
            concept = ThumbnailConcept(
                concept_id="",
                visual_highlight=fallback_highlight,
                emotional_peak=neutral_emotion,
                hook_text=hook_text,
                background_style=strategy_config["background_style"],
                text_style=strategy_config["text_style"],
                visual_elements=["general_content"],
                thumbnail_potential=0.5,
                strategy=strategy_name
            )
            
            concepts.append(concept)
        
        return concepts