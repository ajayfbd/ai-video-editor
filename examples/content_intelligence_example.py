"""
Content Intelligence Engine Example

This example demonstrates how to use the ContentIntelligenceEngine to analyze
content and generate intelligent editing recommendations that coordinate with
the AI Director for enhanced video editing plans.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required modules
from ai_video_editor.modules.intelligence.content_intelligence import (
    ContentIntelligenceEngine,
    EditingOpportunity,
    BRollPlacement,
    TransitionSuggestion,
    PacingPlan,
    EnhancedEditingPlan
)
from ai_video_editor.modules.intelligence.ai_director import (
    FinancialVideoEditor,
    AIDirectorPlan,
    EditingDecision,
    BRollPlan,
    MetadataStrategy
)
from ai_video_editor.modules.intelligence.gemini_client import GeminiClient, GeminiConfig
from ai_video_editor.core.content_context import (
    ContentContext,
    ContentType,
    EmotionalPeak,
    VisualHighlight
)


class MockTranscriptSegment:
    """Mock transcript segment for demonstration."""
    def __init__(self, text: str, start: float, end: float):
        self.text = text
        self.start = start
        self.end = end


class MockTranscript:
    """Mock transcript for demonstration."""
    def __init__(self, segments: List[MockTranscriptSegment]):
        self.segments = segments
        self.text = " ".join(segment.text for segment in segments)


def create_sample_content_context() -> ContentContext:
    """Create a sample ContentContext for demonstration."""
    
    # Create mock transcript segments
    segments = [
        MockTranscriptSegment(
            "Welcome to our financial education series. Today we'll explore compound interest.",
            0.0, 5.0
        ),
        MockTranscriptSegment(
            "Compound interest is the process where your investment grows exponentially over time.",
            5.0, 10.0
        ),
        MockTranscriptSegment(
            "For example, if you invest $1000 at a 7% annual return rate.",
            10.0, 15.0
        ),
        MockTranscriptSegment(
            "After 10 years, your investment would grow to approximately $1967.",
            15.0, 20.0
        ),
        MockTranscriptSegment(
            "This demonstrates the power of letting your money work for you through diversification.",
            20.0, 25.0
        ),
        MockTranscriptSegment(
            "Now let's look at a practical example of portfolio allocation strategies.",
            25.0, 30.0
        )
    ]
    
    # Create mock transcript
    transcript = MockTranscript(segments)
    
    # Create emotional peaks
    emotional_peaks = [
        EmotionalPeak(
            timestamp=12.5,
            emotion="excitement",
            intensity=0.8,
            confidence=0.9,
            context="Explaining compound interest benefits"
        ),
        EmotionalPeak(
            timestamp=22.0,
            emotion="curiosity",
            intensity=0.7,
            confidence=0.8,
            context="Introducing diversification concept"
        )
    ]
    
    # Create visual highlights
    visual_highlights = [
        VisualHighlight(
            timestamp=8.0,
            description="Speaker gesturing to explain growth",
            faces=[],
            visual_elements=["hand_gesture", "facial_expression"],
            thumbnail_potential=0.8
        )
    ]
    
    # Create mock user preferences
    from ai_video_editor.core.content_context import UserPreferences
    user_prefs = UserPreferences(
        quality_mode="high",
        thumbnail_resolution=(1920, 1080),
        batch_size=3,
        enable_aggressive_caching=False,
        parallel_processing=True,
        max_api_cost=2.0
    )
    
    # Create ContentContext
    context = ContentContext(
        project_id="demo_project_001",
        video_files=["demo_video.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=user_prefs
    )
    context.audio_transcript = transcript
    context.emotional_markers = emotional_peaks
    context.visual_highlights = visual_highlights
    context.key_concepts = ["compound interest", "investment", "diversification", "portfolio allocation"]
    context.total_duration = 30.0
    
    # Add processing metrics mock
    class MockProcessingMetrics:
        def add_module_processing_time(self, module: str, time: float):
            logger.info(f"Processing time for {module}: {time:.2f}s")
    
    context.processing_metrics = MockProcessingMetrics()
    
    return context


def create_sample_ai_director_plan() -> AIDirectorPlan:
    """Create a sample AI Director plan for demonstration."""
    
    # Create editing decisions
    editing_decisions = [
        EditingDecision(
            timestamp=5.0,
            decision_type="cut",
            parameters={"fade_duration": 0.5},
            rationale="Natural break between introduction and explanation",
            confidence=0.8,
            priority=7
        ),
        EditingDecision(
            timestamp=15.0,
            decision_type="emphasis",
            parameters={"zoom_factor": 1.2, "duration": 2.0},
            rationale="Emphasize key financial figure",
            confidence=0.9,
            priority=8
        )
    ]
    
    # Create B-roll plans
    broll_plans = [
        BRollPlan(
            timestamp=12.0,
            duration=6.0,
            content_type="chart",
            description="Compound interest growth visualization",
            visual_elements=["line_graph", "data_points", "growth_curve"],
            animation_style="fade_in",
            priority=9
        ),
        BRollPlan(
            timestamp=27.0,
            duration=5.0,
            content_type="animation",
            description="Portfolio diversification diagram",
            visual_elements=["pie_chart", "asset_icons"],
            animation_style="slide_up",
            priority=8
        )
    ]
    
    # Create metadata strategy
    metadata_strategy = MetadataStrategy(
        primary_title="The Power of Compound Interest: A Beginner's Guide",
        title_variations=[
            "How Compound Interest Can Make You Rich",
            "Compound Interest Explained Simply",
            "The Magic of Compound Growth"
        ],
        description="Learn how compound interest works and why it's called the eighth wonder of the world. Perfect for beginners looking to understand investment fundamentals.",
        tags=["compound interest", "investing", "financial education", "wealth building", "personal finance"],
        thumbnail_concepts=["growth chart", "money tree", "calculator with coins"],
        hook_text="This ONE concept can make you wealthy!",
        target_keywords=["compound interest", "investment growth", "financial education"]
    )
    
    # Create AI Director plan
    return AIDirectorPlan(
        editing_decisions=editing_decisions,
        broll_plans=broll_plans,
        metadata_strategy=metadata_strategy,
        quality_enhancements=["audio_noise_reduction", "color_correction"],
        pacing_adjustments=[
            {
                "timestamp": 10.0,
                "adjustment": "slow_down",
                "duration": 5.0,
                "reason": "Complex mathematical concept"
            }
        ],
        engagement_hooks=[
            {
                "timestamp": 18.0,
                "type": "question",
                "content": "Can you guess how much this grows to?",
                "visual_treatment": "text_overlay"
            }
        ],
        created_at=datetime.now(),
        confidence_score=0.85,
        processing_time=3.2,
        model_used="gemini-2.5-pro-latest"
    )


def demonstrate_editing_opportunities():
    """Demonstrate editing opportunity analysis."""
    logger.info("=== Editing Opportunities Analysis ===")
    
    # Create engine and context
    engine = ContentIntelligenceEngine(enable_advanced_analysis=True)
    context = create_sample_content_context()
    
    # Analyze editing opportunities
    opportunities = engine.analyze_editing_opportunities(context)
    
    logger.info(f"Found {len(opportunities)} editing opportunities:")
    for i, opportunity in enumerate(opportunities[:5], 1):  # Show top 5
        logger.info(f"{i}. {opportunity.opportunity_type.upper()} at {opportunity.timestamp:.1f}s")
        logger.info(f"   Priority: {opportunity.priority}/10, Confidence: {opportunity.confidence:.2f}")
        logger.info(f"   Rationale: {opportunity.rationale}")
        logger.info(f"   Trigger: {opportunity.content_trigger}")
        logger.info("")
    
    return opportunities


def demonstrate_broll_detection():
    """Demonstrate B-roll placement detection."""
    logger.info("=== B-roll Placement Detection ===")
    
    # Create engine and context
    engine = ContentIntelligenceEngine(enable_advanced_analysis=True)
    context = create_sample_content_context()
    
    # Detect B-roll placements
    placements = engine.detect_broll_placements(context)
    
    logger.info(f"Detected {len(placements)} B-roll placement opportunities:")
    for i, placement in enumerate(placements, 1):
        logger.info(f"{i}. {placement.content_type.upper()} at {placement.timestamp:.1f}s")
        logger.info(f"   Duration: {placement.duration:.1f}s, Priority: {placement.priority}/10")
        logger.info(f"   Description: {placement.description}")
        logger.info(f"   Educational Value: {placement.educational_value:.2f}")
        logger.info(f"   Trigger Keywords: {', '.join(placement.trigger_keywords)}")
        logger.info(f"   Visual Elements: {', '.join(placement.visual_elements)}")
        logger.info("")
    
    return placements


def demonstrate_transition_suggestions():
    """Demonstrate transition suggestions."""
    logger.info("=== Transition Suggestions ===")
    
    # Create engine and context
    engine = ContentIntelligenceEngine(enable_advanced_analysis=True)
    context = create_sample_content_context()
    
    # Generate transition suggestions
    suggestions = engine.suggest_transitions(context)
    
    logger.info(f"Generated {len(suggestions)} transition suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        logger.info(f"{i}. {suggestion.transition_type.upper()} from {suggestion.from_timestamp:.1f}s to {suggestion.to_timestamp:.1f}s")
        logger.info(f"   Reason: {suggestion.reason}")
        logger.info(f"   Context: {suggestion.content_context}")
        logger.info(f"   Parameters: {suggestion.parameters}")
        logger.info("")
    
    return suggestions


def demonstrate_pacing_optimization():
    """Demonstrate pacing optimization."""
    logger.info("=== Pacing Optimization ===")
    
    # Create engine and context
    engine = ContentIntelligenceEngine(enable_advanced_analysis=True)
    context = create_sample_content_context()
    
    # Optimize pacing
    pacing_plan = engine.optimize_pacing(context)
    
    logger.info(f"Pacing Strategy: {pacing_plan.overall_strategy}")
    logger.info(f"Number of segments: {len(pacing_plan.segments)}")
    logger.info("")
    
    for i, segment in enumerate(pacing_plan.segments, 1):
        logger.info(f"Segment {i}: {segment.start_timestamp:.1f}s - {segment.end_timestamp:.1f}s")
        logger.info(f"   Recommended Speed: {segment.recommended_speed:.2f}x")
        logger.info(f"   Content Complexity: {segment.content_complexity:.2f}")
        logger.info(f"   Reason: {segment.reason}")
        logger.info("")
    
    return pacing_plan


def demonstrate_ai_director_coordination():
    """Demonstrate coordination with AI Director."""
    logger.info("=== AI Director Coordination ===")
    
    # Create engine, context, and AI Director plan
    engine = ContentIntelligenceEngine(enable_advanced_analysis=True)
    context = create_sample_content_context()
    ai_director_plan = create_sample_ai_director_plan()
    
    # Coordinate with AI Director
    enhanced_plan = engine.coordinate_with_ai_director(context, ai_director_plan)
    
    logger.info("Enhanced Editing Plan Created:")
    logger.info(f"Overall Confidence Score: {enhanced_plan.confidence_score:.2f}")
    logger.info(f"Intelligence Recommendations: {len(enhanced_plan.intelligence_recommendations)}")
    logger.info(f"B-roll Enhancements: {len(enhanced_plan.broll_enhancements)}")
    logger.info(f"Transition Improvements: {len(enhanced_plan.transition_improvements)}")
    logger.info(f"Pacing Segments: {len(enhanced_plan.pacing_optimizations.segments)}")
    logger.info("")
    
    logger.info("Coordination Notes:")
    for note in enhanced_plan.coordination_notes:
        logger.info(f"  - {note}")
    logger.info("")
    
    # Show merged B-roll recommendations
    logger.info("Merged B-roll Recommendations:")
    for i, broll in enumerate(enhanced_plan.broll_enhancements, 1):
        logger.info(f"{i}. {broll.content_type} at {broll.timestamp:.1f}s (Priority: {broll.priority})")
        logger.info(f"   {broll.description}")
    
    return enhanced_plan


def demonstrate_performance_analysis():
    """Demonstrate performance analysis of the engine."""
    logger.info("=== Performance Analysis ===")
    
    import time
    
    # Create engine and context
    engine = ContentIntelligenceEngine(enable_advanced_analysis=True)
    context = create_sample_content_context()
    ai_director_plan = create_sample_ai_director_plan()
    
    # Measure performance of each method
    methods_to_test = [
        ("analyze_editing_opportunities", lambda: engine.analyze_editing_opportunities(context)),
        ("detect_broll_placements", lambda: engine.detect_broll_placements(context)),
        ("suggest_transitions", lambda: engine.suggest_transitions(context)),
        ("optimize_pacing", lambda: engine.optimize_pacing(context)),
        ("coordinate_with_ai_director", lambda: engine.coordinate_with_ai_director(context, ai_director_plan))
    ]
    
    performance_results = {}
    
    for method_name, method_func in methods_to_test:
        start_time = time.time()
        result = method_func()
        end_time = time.time()
        
        processing_time = end_time - start_time
        performance_results[method_name] = processing_time
        
        logger.info(f"{method_name}: {processing_time:.3f}s")
    
    total_time = sum(performance_results.values())
    logger.info(f"Total processing time: {total_time:.3f}s")
    
    return performance_results


def main():
    """Main demonstration function."""
    logger.info("Content Intelligence Engine Demonstration")
    logger.info("=" * 50)
    
    try:
        # Run all demonstrations
        opportunities = demonstrate_editing_opportunities()
        placements = demonstrate_broll_detection()
        suggestions = demonstrate_transition_suggestions()
        pacing_plan = demonstrate_pacing_optimization()
        enhanced_plan = demonstrate_ai_director_coordination()
        performance_results = demonstrate_performance_analysis()
        
        logger.info("=== Summary ===")
        logger.info(f"✅ Generated {len(opportunities)} editing opportunities")
        logger.info(f"✅ Detected {len(placements)} B-roll placements")
        logger.info(f"✅ Created {len(suggestions)} transition suggestions")
        logger.info(f"✅ Optimized pacing with {len(pacing_plan.segments)} segments")
        logger.info(f"✅ Enhanced AI Director plan with confidence {enhanced_plan.confidence_score:.2f}")
        logger.info(f"✅ Total processing time: {sum(performance_results.values()):.3f}s")
        
        logger.info("\nContent Intelligence Engine demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()