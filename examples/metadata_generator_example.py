"""
MetadataGenerator Example - Demonstrates SEO-optimized metadata generation.

This example shows how to use the MetadataGenerator to create comprehensive
metadata packages with multiple variations for A/B testing.
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


def create_sample_content_context() -> ContentContext:
    """Create a sample ContentContext for demonstration."""
    
    # Create trending keywords (simulating TrendAnalyzer output)
    trending_keywords = TrendingKeywords(
        primary_keywords=[
            "financial education", "investing for beginners", "compound interest",
            "passive income", "wealth building"
        ],
        long_tail_keywords=[
            "how to start investing with little money",
            "compound interest explained simply",
            "passive income strategies for beginners"
        ],
        trending_hashtags=["#finance", "#investing", "#wealth", "#money"],
        seasonal_keywords=["2024", "new year", "financial goals"],
        competitor_keywords=["financial literacy", "investment tips", "money management"],
        search_volume_data={
            "financial education": 12000,
            "investing for beginners": 8500,
            "compound interest": 6200,
            "passive income": 15000,
            "wealth building": 4800
        },
        research_timestamp=datetime.now(),
        keyword_difficulty={
            "financial education": 0.65,
            "investing for beginners": 0.72,
            "compound interest": 0.58,
            "passive income": 0.81,
            "wealth building": 0.69
        },
        keyword_confidence={
            "financial education": 0.92,
            "investing for beginners": 0.88,
            "compound interest": 0.95,
            "passive income": 0.85,
            "wealth building": 0.90
        },
        trending_topics=[
            "AI-powered investing", "sustainable investing", "crypto education",
            "retirement planning", "emergency fund strategies"
        ],
        research_quality_score=0.89,
        cache_hit_rate=0.35
    )
    
    # Create emotional markers (simulating audio analysis)
    emotional_markers = [
        EmotionalPeak(
            timestamp=15.5,
            emotion="excitement",
            intensity=0.85,
            confidence=0.92,
            context="Introducing the power of compound interest"
        ),
        EmotionalPeak(
            timestamp=45.2,
            emotion="curiosity",
            intensity=0.78,
            confidence=0.89,
            context="Revealing the 50/30/20 budgeting rule"
        ),
        EmotionalPeak(
            timestamp=120.8,
            emotion="surprise",
            intensity=0.82,
            confidence=0.91,
            context="Showing real investment growth examples"
        ),
        EmotionalPeak(
            timestamp=180.3,
            emotion="confidence",
            intensity=0.88,
            confidence=0.94,
            context="Explaining how anyone can start investing"
        )
    ]
    
    # Create visual highlights (simulating video analysis)
    visual_highlights = [
        VisualHighlight(
            timestamp=30.0,
            description="Animated chart showing compound interest growth over 30 years",
            faces=[FaceDetection([150, 100, 300, 250], 0.96, "focused")],
            visual_elements=["chart", "animation", "numbers", "growth_curve"],
            thumbnail_potential=0.94
        ),
        VisualHighlight(
            timestamp=75.5,
            description="Split-screen comparison of different investment strategies",
            faces=[FaceDetection([200, 120, 350, 280], 0.93, "explaining")],
            visual_elements=["split_screen", "comparison", "graphs", "percentages"],
            thumbnail_potential=0.87
        ),
        VisualHighlight(
            timestamp=135.2,
            description="Real portfolio screenshots showing actual returns",
            faces=[FaceDetection([180, 110, 320, 270], 0.95, "confident")],
            visual_elements=["screenshots", "portfolio", "green_numbers", "profits"],
            thumbnail_potential=0.91
        )
    ]
    
    # Create audio analysis results
    audio_segments = [
        AudioSegment(
            text="Welcome to Financial Education 101. Today we're going to learn about the incredible power of compound interest.",
            start=0.0,
            end=8.5,
            confidence=0.96,
            financial_concepts=["compound interest", "financial education"],
            cleaned_text="Welcome to Financial Education 101. Today we're going to learn about the incredible power of compound interest."
        ),
        AudioSegment(
            text="Albert Einstein supposedly called compound interest the eighth wonder of the world, and you're about to see why.",
            start=8.5,
            end=16.2,
            confidence=0.94,
            financial_concepts=["compound interest"],
            emotional_markers=["excitement"]
        ),
        AudioSegment(
            text="Let me show you how investing just $100 per month can turn into over $1 million by retirement.",
            start=16.2,
            end=24.8,
            confidence=0.97,
            financial_concepts=["investing", "retirement planning"]
        )
    ]
    
    audio_analysis = AudioAnalysisResult(
        transcript_text="Welcome to Financial Education 101. Today we're going to learn about the incredible power of compound interest. Albert Einstein supposedly called compound interest the eighth wonder of the world, and you're about to see why. Let me show you how investing just $100 per month can turn into over $1 million by retirement.",
        segments=audio_segments,
        overall_confidence=0.95,
        language="en",
        processing_time=3.2,
        model_used="whisper-large-v3",
        filler_words_removed=12,
        segments_modified=3,
        quality_improvement_score=0.88,
        original_duration=300.0,
        enhanced_duration=285.5,
        financial_concepts=[
            "compound interest", "financial education", "investing",
            "retirement planning", "portfolio diversification", "passive income"
        ],
        explanation_segments=[
            {
                "start": 8.5,
                "end": 45.2,
                "concept": "compound interest",
                "complexity": "beginner",
                "visual_aid_potential": 0.92
            },
            {
                "start": 45.2,
                "end": 85.7,
                "concept": "investment strategies",
                "complexity": "intermediate",
                "visual_aid_potential": 0.85
            }
        ],
        data_references=[
            {
                "timestamp": 24.8,
                "data_type": "financial_projection",
                "values": ["$100", "$1 million"],
                "chart_potential": 0.95
            }
        ],
        complexity_level="beginner",
        detected_emotions=emotional_markers,
        engagement_points=[
            {"timestamp": 15.5, "type": "hook", "strength": 0.89},
            {"timestamp": 120.8, "type": "revelation", "strength": 0.92},
            {"timestamp": 180.3, "type": "call_to_action", "strength": 0.87}
        ]
    )
    
    # Create ContentContext
    context = ContentContext(
        project_id="financial_education_demo",
        video_files=["financial_education_101.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences(
            quality_mode="high",
            thumbnail_resolution=(1920, 1080),
            max_api_cost=5.0
        ),
        key_concepts=[
            "compound interest", "financial education", "investing for beginners",
            "passive income", "wealth building", "retirement planning"
        ],
        content_themes=[
            "education", "finance", "beginner-friendly", "practical tips",
            "long-term wealth", "financial literacy"
        ],
        emotional_markers=emotional_markers,
        visual_highlights=visual_highlights,
        trending_keywords=trending_keywords
    )
    
    # Set audio analysis
    context.set_audio_analysis(audio_analysis)
    
    return context


async def demonstrate_metadata_generation():
    """Demonstrate comprehensive metadata generation."""
    
    print("🎯 MetadataGenerator Demonstration")
    print("=" * 50)
    
    # Initialize components
    cache_manager = CacheManager("temp/metadata_cache")
    metadata_generator = MetadataGenerator(cache_manager)
    
    # Create sample content context
    print("\n📊 Creating sample content context...")
    context = create_sample_content_context()
    
    print(f"✅ Content Context created:")
    print(f"   - Project ID: {context.project_id}")
    print(f"   - Content Type: {context.content_type.value}")
    print(f"   - Key Concepts: {len(context.key_concepts)} concepts")
    print(f"   - Emotional Markers: {len(context.emotional_markers)} peaks")
    print(f"   - Visual Highlights: {len(context.visual_highlights)} highlights")
    print(f"   - Trending Keywords: {len(context.trending_keywords.primary_keywords)} primary keywords")
    
    # Generate metadata package
    print("\n🚀 Generating metadata package...")
    start_time = datetime.now()
    
    try:
        result_context = await metadata_generator.generate_metadata_package(context)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"✅ Metadata generation completed in {processing_time:.2f}s")
        
        # Display results
        if result_context.metadata_variations:
            package_data = result_context.metadata_variations[0]
            
            print(f"\n📦 Generated Metadata Package:")
            print(f"   - Variations: {len(package_data['variations'])}")
            print(f"   - Recommended: {package_data['recommended_variation']}")
            print(f"   - Generation Time: {package_data['generation_timestamp']}")
            
            # Display each variation
            print(f"\n📝 Metadata Variations:")
            print("=" * 50)
            
            for i, variation_data in enumerate(package_data['variations'], 1):
                print(f"\n🎯 Variation {i}: {variation_data['strategy'].upper()}")
                print(f"   Strategy: {variation_data['strategy']}")
                print(f"   Confidence: {variation_data['confidence_score']:.1%}")
                print(f"   SEO Score: {variation_data['seo_score']:.1%}")
                print(f"   Est. CTR: {variation_data['estimated_ctr']:.1%}")
                
                print(f"\n   📰 Title ({len(variation_data['title'])} chars):")
                print(f"   {variation_data['title']}")
                
                print(f"\n   📄 Description ({len(variation_data['description'])} chars):")
                description_preview = variation_data['description'][:200] + "..." if len(variation_data['description']) > 200 else variation_data['description']
                print(f"   {description_preview}")
                
                print(f"\n   🏷️  Tags ({len(variation_data['tags'])} tags):")
                print(f"   {', '.join(variation_data['tags'])}")
                
                if variation_data['variation_id'] == package_data['recommended_variation']:
                    print("   ⭐ RECOMMENDED VARIATION")
                
                print("-" * 50)
            
            # Display SEO insights
            if 'seo_insights' in package_data:
                seo_insights = package_data['seo_insights']
                print(f"\n🔍 SEO Insights:")
                print(f"   - Trending Alignment: {seo_insights.get('trending_alignment', 0):.1%}")
                
                if 'optimization_suggestions' in seo_insights:
                    print(f"   - Optimization Suggestions:")
                    for suggestion in seo_insights['optimization_suggestions']:
                        print(f"     • {suggestion}")
            
            # Display performance predictions
            if 'performance_predictions' in package_data:
                predictions = package_data['performance_predictions']
                print(f"\n📈 Performance Predictions:")
                print(f"   - Best CTR Variation: {predictions.get('best_ctr_variation', 'N/A')}")
                print(f"   - Best SEO Variation: {predictions.get('best_seo_variation', 'N/A')}")
                print(f"   - Balanced Recommendation: {predictions.get('balanced_recommendation', 'N/A')}")
            
            # Display content analysis insights
            if 'content_analysis' in package_data:
                content_analysis = package_data['content_analysis']
                print(f"\n🎬 Content Analysis:")
                print(f"   - Primary Concept: {content_analysis.get('primary_concept', 'N/A')}")
                print(f"   - Target Audience: {content_analysis.get('target_audience', 'N/A')}")
                print(f"   - Emotional Peaks: {content_analysis.get('emotional_peaks', 0)}")
                print(f"   - Visual Highlights: {content_analysis.get('visual_highlights', 0)}")
                
                if 'dominant_emotion' in content_analysis:
                    print(f"   - Dominant Emotion: {content_analysis['dominant_emotion']}")
        
        # Display processing metrics
        print(f"\n⚡ Processing Metrics:")
        metrics = result_context.processing_metrics
        print(f"   - Total Processing Time: {metrics.total_processing_time:.2f}s")
        print(f"   - Module Processing Times: {metrics.module_processing_times}")
        print(f"   - Memory Peak Usage: {metrics.memory_peak_usage} bytes")
        
        # Save example output
        output_dir = Path("output/metadata_examples")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"metadata_package_{context.project_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(package_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Example output saved to: {output_file}")
        
        # Display cache statistics
        cache_stats = cache_manager.get_stats()
        print(f"\n📊 Cache Statistics:")
        print(f"   - Cache Hits: {cache_stats['hits']}")
        print(f"   - Cache Misses: {cache_stats['misses']}")
        print(f"   - Hit Rate: {cache_stats['hit_rate']:.1%}")
        print(f"   - Memory Cache Size: {cache_stats['memory_cache_size']}")
        
    except Exception as e:
        print(f"❌ Error during metadata generation: {str(e)}")
        raise


async def demonstrate_a_b_testing():
    """Demonstrate A/B testing capabilities."""
    
    print("\n🧪 A/B Testing Demonstration")
    print("=" * 50)
    
    # Initialize components
    cache_manager = CacheManager("temp/metadata_cache")
    metadata_generator = MetadataGenerator(cache_manager)
    
    # Create content context
    context = create_sample_content_context()
    
    # Generate metadata package
    result_context = await metadata_generator.generate_metadata_package(context)
    package_data = result_context.metadata_variations[0]
    
    print(f"\n📊 A/B Testing Analysis:")
    
    # Analyze variations by strategy
    strategy_performance = {}
    for variation in package_data['variations']:
        strategy = variation['strategy']
        if strategy not in strategy_performance:
            strategy_performance[strategy] = []
        
        strategy_performance[strategy].append({
            'confidence': variation['confidence_score'],
            'ctr': variation['estimated_ctr'],
            'seo': variation['seo_score'],
            'title': variation['title']
        })
    
    # Display strategy comparison
    print(f"\n🎯 Strategy Performance Comparison:")
    for strategy, variations in strategy_performance.items():
        avg_confidence = sum(v['confidence'] for v in variations) / len(variations)
        avg_ctr = sum(v['ctr'] for v in variations) / len(variations)
        avg_seo = sum(v['seo'] for v in variations) / len(variations)
        
        print(f"\n   📈 {strategy.upper()}:")
        print(f"      - Avg Confidence: {avg_confidence:.1%}")
        print(f"      - Avg Est. CTR: {avg_ctr:.1%}")
        print(f"      - Avg SEO Score: {avg_seo:.1%}")
        print(f"      - Sample Title: {variations[0]['title'][:60]}...")
    
    # Recommend testing strategy
    best_strategy = max(strategy_performance.items(), 
                       key=lambda x: sum(v['confidence'] for v in x[1]) / len(x[1]))
    
    print(f"\n🏆 Recommended A/B Testing Strategy:")
    print(f"   - Primary Strategy: {best_strategy[0]}")
    print(f"   - Test against: emotional and curiosity_driven variations")
    print(f"   - Expected performance lift: 15-25%")


if __name__ == "__main__":
    print("🎬 AI Video Editor - MetadataGenerator Example")
    print("=" * 60)
    
    # Run demonstrations
    asyncio.run(demonstrate_metadata_generation())
    asyncio.run(demonstrate_a_b_testing())
    
    print("\n✅ MetadataGenerator demonstration completed!")
    print("\nKey Features Demonstrated:")
    print("• Multiple metadata variations with different strategies")
    print("• SEO optimization with keyword integration")
    print("• A/B testing support with performance predictions")
    print("• Comprehensive descriptions with timestamps")
    print("• Optimized tag generation (10-15 tags)")
    print("• Content analysis integration")
    print("• Caching for performance optimization")
    print("• Memory tracking for pattern learning")