"""
Metadata Integration Example - Complete metadata package integration demonstration.

This example shows how to use the MetadataPackageIntegrator to create synchronized
metadata packages that align with video content, thumbnails, and AI Director decisions.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from ai_video_editor.core.content_context import (
    ContentContext,
    ContentType,
    TrendingKeywords,
    EmotionalPeak,
    VisualHighlight,
    FaceDetection,
    AudioAnalysisResult,
    AudioSegment,
    UserPreferences
)
from ai_video_editor.core.cache_manager import CacheManager
from ai_video_editor.modules.intelligence.metadata_generator import MetadataGenerator
from ai_video_editor.modules.intelligence.metadata_integration import MetadataPackageIntegrator


def create_comprehensive_content_context() -> ContentContext:
    """Create a comprehensive ContentContext for integration demonstration."""
    
    # Create trending keywords (simulating TrendAnalyzer output)
    trending_keywords = TrendingKeywords(
        primary_keywords=[
            "compound interest explained", "investment for beginners", "financial freedom",
            "passive income strategies", "wealth building tips"
        ],
        long_tail_keywords=[
            "how compound interest makes you rich",
            "beginner investment strategies that work",
            "passive income ideas for financial freedom"
        ],
        trending_hashtags=["#compoundinterest", "#investing", "#financialfreedom", "#wealth"],
        seasonal_keywords=["2024", "new year goals", "financial planning"],
        competitor_keywords=["money management", "investment education", "financial literacy"],
        search_volume_data={
            "compound interest explained": 15000,
            "investment for beginners": 22000,
            "financial freedom": 18000,
            "passive income strategies": 12000,
            "wealth building tips": 9500
        },
        research_timestamp=datetime.now(),
        keyword_difficulty={
            "compound interest explained": 0.68,
            "investment for beginners": 0.75,
            "financial freedom": 0.82,
            "passive income strategies": 0.71,
            "wealth building tips": 0.63
        },
        keyword_confidence={
            "compound interest explained": 0.94,
            "investment for beginners": 0.91,
            "financial freedom": 0.88,
            "passive income strategies": 0.92,
            "wealth building tips": 0.89
        },
        trending_topics=[
            "AI-powered investing", "sustainable wealth building", "crypto education",
            "retirement planning 2024", "emergency fund strategies"
        ],
        research_quality_score=0.91,
        cache_hit_rate=0.42
    )
    
    # Create emotional markers (simulating comprehensive audio analysis)
    emotional_markers = [
        EmotionalPeak(
            timestamp=12.5,
            emotion="excitement",
            intensity=0.89,
            confidence=0.94,
            context="Introducing the magic of compound interest"
        ),
        EmotionalPeak(
            timestamp=45.8,
            emotion="curiosity",
            intensity=0.82,
            confidence=0.91,
            context="Revealing the $1 million retirement secret"
        ),
        EmotionalPeak(
            timestamp=78.3,
            emotion="surprise",
            intensity=0.87,
            confidence=0.93,
            context="Showing real portfolio growth examples"
        ),
        EmotionalPeak(
            timestamp=125.7,
            emotion="confidence",
            intensity=0.91,
            confidence=0.96,
            context="Explaining how anyone can start today"
        ),
        EmotionalPeak(
            timestamp=180.2,
            emotion="inspiration",
            intensity=0.85,
            confidence=0.92,
            context="Motivating viewers to take action"
        )
    ]
    
    # Create visual highlights (simulating comprehensive video analysis)
    visual_highlights = [
        VisualHighlight(
            timestamp=25.0,
            description="Animated compound interest growth chart over 30 years",
            faces=[FaceDetection([160, 110, 320, 270], 0.97, "focused_explaining")],
            visual_elements=["animated_chart", "growth_curve", "dollar_signs", "timeline"],
            thumbnail_potential=0.96
        ),
        VisualHighlight(
            timestamp=52.5,
            description="Split-screen comparison: $100/month vs $500/month investing",
            faces=[FaceDetection([180, 120, 340, 280], 0.94, "confident_presenting")],
            visual_elements=["split_screen", "comparison_chart", "dollar_amounts", "arrows"],
            thumbnail_potential=0.92
        ),
        VisualHighlight(
            timestamp=89.2,
            description="Real brokerage account screenshot showing $847,000 balance",
            faces=[FaceDetection([200, 130, 360, 290], 0.96, "excited_revealing")],
            visual_elements=["screenshot", "account_balance", "green_numbers", "profit_chart"],
            thumbnail_potential=0.98
        ),
        VisualHighlight(
            timestamp=142.8,
            description="Calculator animation showing compound interest calculation",
            faces=[FaceDetection([170, 115, 330, 275], 0.95, "teaching_mode")],
            visual_elements=["calculator", "mathematical_formula", "step_by_step", "numbers"],
            thumbnail_potential=0.89
        ),
        VisualHighlight(
            timestamp=195.5,
            description="Before/after lifestyle comparison with financial freedom",
            faces=[FaceDetection([190, 125, 350, 285], 0.93, "inspiring_conclusion")],
            visual_elements=["lifestyle_images", "before_after", "freedom_symbols", "success"],
            thumbnail_potential=0.87
        )
    ]
    
    # Create comprehensive audio analysis
    audio_segments = [
        AudioSegment(
            text="Welcome to the most important financial lesson you'll ever learn about compound interest.",
            start=0.0,
            end=6.5,
            confidence=0.98,
            financial_concepts=["compound interest", "financial education"],
            cleaned_text="Welcome to the most important financial lesson you'll ever learn about compound interest."
        ),
        AudioSegment(
            text="Albert Einstein called compound interest the eighth wonder of the world, and today you'll see exactly why.",
            start=6.5,
            end=14.2,
            confidence=0.96,
            financial_concepts=["compound interest"],
            emotional_markers=["excitement", "authority"]
        ),
        AudioSegment(
            text="I'm going to show you how investing just $100 per month can turn into over $1.2 million by retirement.",
            start=14.2,
            end=22.8,
            confidence=0.97,
            financial_concepts=["investing", "retirement planning", "monthly investing"],
            emotional_markers=["surprise", "curiosity"]
        ),
        AudioSegment(
            text="But first, let me show you this chart that will blow your mind about the power of starting early.",
            start=22.8,
            end=29.5,
            confidence=0.95,
            financial_concepts=["early investing", "time value of money"],
            emotional_markers=["anticipation"]
        )
    ]
    
    audio_analysis = AudioAnalysisResult(
        transcript_text="Welcome to the most important financial lesson you'll ever learn about compound interest. Albert Einstein called compound interest the eighth wonder of the world, and today you'll see exactly why. I'm going to show you how investing just $100 per month can turn into over $1.2 million by retirement. But first, let me show you this chart that will blow your mind about the power of starting early.",
        segments=audio_segments,
        overall_confidence=0.96,
        language="en",
        processing_time=4.2,
        model_used="whisper-large-v3",
        filler_words_removed=18,
        segments_modified=4,
        quality_improvement_score=0.91,
        original_duration=420.0,
        enhanced_duration=398.5,
        financial_concepts=[
            "compound interest", "financial education", "investing", "retirement planning",
            "monthly investing", "early investing", "time value of money", "portfolio growth",
            "passive income", "wealth building", "financial freedom"
        ],
        explanation_segments=[
            {
                "start": 6.5,
                "end": 52.5,
                "concept": "compound interest basics",
                "complexity": "beginner",
                "visual_aid_potential": 0.95
            },
            {
                "start": 52.5,
                "end": 125.7,
                "concept": "investment comparison strategies",
                "complexity": "intermediate",
                "visual_aid_potential": 0.92
            },
            {
                "start": 125.7,
                "end": 195.5,
                "concept": "practical implementation",
                "complexity": "beginner",
                "visual_aid_potential": 0.88
            }
        ],
        data_references=[
            {
                "timestamp": 22.8,
                "data_type": "financial_projection",
                "values": ["$100", "$1.2 million"],
                "chart_potential": 0.98
            },
            {
                "timestamp": 89.2,
                "data_type": "real_account_balance",
                "values": ["$847,000"],
                "chart_potential": 0.96
            }
        ],
        complexity_level="beginner",
        detected_emotions=emotional_markers,
        engagement_points=[
            {"timestamp": 12.5, "type": "hook", "strength": 0.94},
            {"timestamp": 45.8, "type": "revelation", "strength": 0.96},
            {"timestamp": 89.2, "type": "proof", "strength": 0.98},
            {"timestamp": 180.2, "type": "call_to_action", "strength": 0.91}
        ]
    )
    
    # Create ContentContext with comprehensive data
    context = ContentContext(
        project_id="compound_interest_mastery",
        video_files=["compound_interest_explained.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences(
            quality_mode="high",
            thumbnail_resolution=(1920, 1080),
            max_api_cost=8.0
        ),
        key_concepts=[
            "compound interest", "investment strategies", "financial freedom",
            "passive income", "wealth building", "retirement planning",
            "early investing", "time value of money"
        ],
        content_themes=[
            "financial education", "beginner-friendly", "practical tips",
            "long-term wealth", "investment basics", "money management",
            "financial literacy", "retirement planning"
        ],
        emotional_markers=emotional_markers,
        visual_highlights=visual_highlights,
        trending_keywords=trending_keywords
    )
    
    # Set audio analysis
    context.set_audio_analysis(audio_analysis)
    
    # Add thumbnail concepts (simulating AI Director output)
    context.thumbnail_concepts = [
        {
            "concept": "compound interest growth chart",
            "confidence": 0.96,
            "visual_elements": ["chart", "growth_curve", "money"],
            "emotional_trigger": "excitement",
            "ctr_prediction": 0.18
        },
        {
            "concept": "shocked face with money calculator",
            "confidence": 0.92,
            "visual_elements": ["face", "calculator", "dollar_signs"],
            "emotional_trigger": "surprise",
            "ctr_prediction": 0.22
        },
        {
            "concept": "before after wealth comparison",
            "confidence": 0.89,
            "visual_elements": ["split_screen", "lifestyle", "transformation"],
            "emotional_trigger": "aspiration",
            "ctr_prediction": 0.16
        },
        {
            "concept": "million dollar account screenshot",
            "confidence": 0.94,
            "visual_elements": ["screenshot", "account_balance", "green_numbers"],
            "emotional_trigger": "proof",
            "ctr_prediction": 0.25
        }
    ]
    
    # Add AI Director plan (simulating comprehensive AI decisions)
    context.ai_director_plan = {
        'creative_strategy': 'educational_with_proof',
        'content_strategy': {
            'focus': 'beginner_friendly_with_advanced_insights',
            'emotional_arc': ['curiosity', 'excitement', 'surprise', 'confidence', 'inspiration'],
            'proof_points': ['real_account_screenshots', 'mathematical_demonstrations', 'visual_comparisons'],
            'call_to_action': 'start_investing_today'
        },
        'thumbnail_concepts': [
            'compound interest growth chart',
            'shocked face with money calculator',
            'million dollar account screenshot'
        ],
        'metadata_strategy': {
            'primary_hook': 'compound_interest_revelation',
            'target_emotions': ['curiosity', 'excitement', 'surprise'],
            'proof_integration': 'real_results_focus',
            'audience_targeting': 'beginner_investors'
        },
        'visual_storytelling': {
            'opening_hook': 'einstein_quote_with_chart',
            'proof_moment': 'real_account_reveal',
            'educational_flow': 'simple_to_complex',
            'closing_motivation': 'action_oriented'
        }
    }
    
    return context


async def demonstrate_complete_integration():
    """Demonstrate complete metadata package integration."""
    
    print("🎯 Complete Metadata Package Integration Demonstration")
    print("=" * 60)
    
    # Initialize components
    cache_manager = CacheManager("temp/integration_cache")
    metadata_generator = MetadataGenerator(cache_manager)
    integrator = MetadataPackageIntegrator(cache_manager, metadata_generator)
    
    # Create comprehensive content context
    print("\n📊 Creating comprehensive content context...")
    context = create_comprehensive_content_context()
    
    print(f"✅ Content Context created:")
    print(f"   - Project ID: {context.project_id}")
    print(f"   - Content Type: {context.content_type.value}")
    print(f"   - Key Concepts: {len(context.key_concepts)} concepts")
    print(f"   - Emotional Markers: {len(context.emotional_markers)} peaks")
    print(f"   - Visual Highlights: {len(context.visual_highlights)} highlights")
    print(f"   - Thumbnail Concepts: {len(context.thumbnail_concepts)} concepts")
    print(f"   - AI Director Plan: {'✅ Available' if context.ai_director_plan else '❌ Missing'}")
    print(f"   - Trending Keywords: {len(context.trending_keywords.primary_keywords)} primary keywords")
    
    # Create integrated metadata package
    print("\n🚀 Creating integrated metadata package...")
    start_time = datetime.now()
    
    try:
        result_context = await integrator.create_integrated_package(context)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Integration completed in {processing_time:.2f}s")
        
        # Display integrated package results
        if result_context.metadata_variations:
            package_data = result_context.metadata_variations[0]
            
            print(f"\n📦 Integrated Metadata Package:")
            print(f"   - Publish Readiness Score: {package_data['publish_readiness_score']:.1%}")
            print(f"   - Generation Time: {package_data['generation_timestamp']}")
            
            # Display validation results
            validation = package_data['validation_result']
            print(f"\n✅ Validation Results:")
            print(f"   - Package Complete: {'✅ YES' if validation['is_complete'] else '❌ NO'}")
            print(f"   - Quality Score: {validation['quality_score']:.1%}")
            print(f"   - Synchronization Score: {validation['synchronization_score']:.1%}")
            print(f"   - AI Director Alignment: {validation['ai_director_alignment']:.1%}")
            
            if validation['missing_components']:
                print(f"   - Missing Components: {', '.join(validation['missing_components'])}")
            
            if validation['recommendations']:
                print(f"   - Recommendations:")
                for rec in validation['recommendations']:
                    print(f"     • {rec}")
            
            # Display primary metadata
            primary = package_data['primary_metadata']
            print(f"\n📰 Primary Metadata:")
            print(f"   - Strategy: {primary['strategy']}")
            print(f"   - Confidence: {primary['confidence_score']:.1%}")
            
            print(f"\n   📝 Title ({len(primary['title'])} chars):")
            print(f"   {primary['title']}")
            
            print(f"\n   📄 Description ({len(primary['description'])} chars):")
            description_preview = primary['description'][:300] + "..." if len(primary['description']) > 300 else primary['description']
            print(f"   {description_preview}")
            
            print(f"\n   🏷️  Tags ({len(primary['tags'])} tags):")
            print(f"   {', '.join(primary['tags'])}")
            
            # Display thumbnail alignments
            if 'thumbnail_alignments' in package_data:
                alignments = package_data['thumbnail_alignments']
                print(f"\n🖼️  Thumbnail-Metadata Alignments ({len(alignments)} alignments):")
                
                for i, alignment in enumerate(alignments, 1):
                    print(f"\n   🎯 Alignment {i}: {alignment['thumbnail_concept']}")
                    print(f"      - Visual Consistency: {alignment['visual_consistency_score']:.1%}")
                    print(f"      - Hook Text: {alignment['hook_text_integration']}")
                    print(f"      - Keyword Overlap: {', '.join(alignment['keyword_overlap'][:3])}")
                    print(f"      - Aligned Title: {alignment['aligned_title'][:60]}...")
            
            # Display AI Director integration
            if 'ai_director_integration' in package_data:
                ai_integration = package_data['ai_director_integration']
                print(f"\n🤖 AI Director Integration:")
                print(f"   - Director Influence Score: {ai_integration['director_influence_score']:.1%}")
                
                if 'creative_alignment' in ai_integration:
                    creative = ai_integration['creative_alignment']
                    print(f"   - Creative Strategy: {creative['strategy']}")
                    print(f"   - Target Emotions: {', '.join(creative['target_emotions'][:3])}")
                    print(f"   - Creative Alignment: {creative['alignment_score']:.1%}")
                
                if 'strategic_alignment' in ai_integration:
                    strategic = ai_integration['strategic_alignment']
                    print(f"   - Key Messages: {len(strategic['key_messages'])} messages")
                    print(f"   - Strategic Alignment: {strategic['alignment_score']:.1%}")
            
            # Display content synchronization
            if 'content_synchronization' in package_data:
                sync = package_data['content_synchronization']
                print(f"\n🔄 Content Synchronization:")
                print(f"   - Overall Alignment Score: {sync['content_alignment_score']:.1%}")
                
                if 'emotional_alignment' in sync:
                    emotional = sync['emotional_alignment']
                    print(f"   - Emotional Peaks: {emotional['peak_count']} peaks")
                    print(f"   - Dominant Emotions: {', '.join(emotional['dominant_emotions'])}")
                    print(f"   - Emotional Coverage: {emotional['coverage_score']:.1%}")
                
                if 'visual_alignment' in sync:
                    visual = sync['visual_alignment']
                    print(f"   - Visual Highlights: {visual['highlight_count']} highlights")
                    print(f"   - Avg Thumbnail Potential: {visual['thumbnail_potential']:.1%}")
                    print(f"   - Visual Coverage: {visual['coverage_score']:.1%}")
            
            # Display alternative variations
            if 'alternative_variations' in package_data:
                alternatives = package_data['alternative_variations']
                print(f"\n🔄 Alternative Variations ({len(alternatives)} variations):")
                
                for i, alt in enumerate(alternatives, 1):
                    print(f"\n   📝 Alternative {i}: {alt['strategy'].upper()}")
                    print(f"      - Confidence: {alt['confidence_score']:.1%}")
                    print(f"      - Title: {alt['title'][:50]}...")
                    print(f"      - Tags: {', '.join(alt['tags'][:5])}")
        
        # Display processing metrics
        print(f"\n⚡ Processing Metrics:")
        metrics = result_context.processing_metrics
        print(f"   - Total Processing Time: {metrics.total_processing_time:.2f}s")
        print(f"   - Module Processing Times: {metrics.module_processing_times}")
        print(f"   - Memory Peak Usage: {metrics.memory_peak_usage} bytes")
        
        # Save integrated package
        output_dir = Path("output/integration_examples")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"integrated_package_{context.project_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(package_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Integrated package saved to: {output_file}")
        
        # Display cache statistics
        cache_stats = cache_manager.get_stats()
        print(f"\n📊 Cache Statistics:")
        print(f"   - Cache Hits: {cache_stats['hits']}")
        print(f"   - Cache Misses: {cache_stats['misses']}")
        print(f"   - Hit Rate: {cache_stats['hit_rate']:.1%}")
        print(f"   - Memory Cache Size: {cache_stats['memory_cache_size']}")
        
    except Exception as e:
        print(f"❌ Error during integration: {str(e)}")
        raise


async def demonstrate_publish_readiness_analysis():
    """Demonstrate publish readiness analysis and recommendations."""
    
    print("\n🎯 Publish Readiness Analysis")
    print("=" * 50)
    
    # Initialize components
    cache_manager = CacheManager("temp/integration_cache")
    metadata_generator = MetadataGenerator(cache_manager)
    integrator = MetadataPackageIntegrator(cache_manager, metadata_generator)
    
    # Create content context
    context = create_comprehensive_content_context()
    
    # Create integrated package
    result_context = await integrator.create_integrated_package(context)
    package_data = result_context.metadata_variations[0]
    
    print(f"\n📊 Publish Readiness Analysis:")
    
    # Overall readiness
    readiness_score = package_data['publish_readiness_score']
    print(f"   - Overall Readiness: {readiness_score:.1%}")
    
    if readiness_score >= 0.9:
        print("   - Status: 🟢 EXCELLENT - Ready to publish immediately")
    elif readiness_score >= 0.8:
        print("   - Status: 🟡 GOOD - Minor optimizations recommended")
    elif readiness_score >= 0.7:
        print("   - Status: 🟠 FAIR - Several improvements needed")
    else:
        print("   - Status: 🔴 POOR - Major improvements required")
    
    # Detailed breakdown
    validation = package_data['validation_result']
    print(f"\n📈 Readiness Breakdown:")
    print(f"   - Content Quality: {validation['quality_score']:.1%}")
    print(f"   - Synchronization: {validation['synchronization_score']:.1%}")
    print(f"   - AI Alignment: {validation['ai_director_alignment']:.1%}")
    
    # Recommendations
    if validation['recommendations']:
        print(f"\n💡 Optimization Recommendations:")
        for i, rec in enumerate(validation['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    # Success factors
    print(f"\n✅ Success Factors:")
    print(f"   - Comprehensive content analysis with {len(context.emotional_markers)} emotional peaks")
    print(f"   - Strong visual alignment with {len(context.visual_highlights)} highlights")
    print(f"   - AI Director integration with strategic planning")
    print(f"   - Thumbnail synchronization with {len(context.thumbnail_concepts)} concepts")
    print(f"   - SEO optimization with {len(context.trending_keywords.primary_keywords)} primary keywords")


if __name__ == "__main__":
    print("🎬 AI Video Editor - Complete Metadata Integration Example")
    print("=" * 70)
    
    # Run demonstrations
    asyncio.run(demonstrate_complete_integration())
    asyncio.run(demonstrate_publish_readiness_analysis())
    
    print("\n✅ Complete metadata integration demonstration completed!")
    print("\nKey Features Demonstrated:")
    print("• Complete metadata package integration with video content")
    print("• Thumbnail-metadata synchronization and alignment")
    print("• AI Director decision integration and strategic alignment")
    print("• Content synchronization with emotional and visual analysis")
    print("• Comprehensive validation and publish readiness scoring")
    print("• Alternative variations for A/B testing")
    print("• Performance optimization with intelligent caching")
    print("• Error handling and graceful degradation")
    print("• Detailed analytics and recommendations")