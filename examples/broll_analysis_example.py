"""
B-Roll Analysis and Graphics Generation Example.

This example demonstrates the complete Task 6 implementation:
- B-Roll Opportunity Analysis using FinancialBRollAnalyzer
- Graphics and Animation Planning using AIGraphicsDirector
- Integration with ContentContext and AI Director
"""

import sys
import os
from pathlib import Path
import asyncio

# Add the project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_video_editor.core.content_context import (
    ContentContext, ContentType, UserPreferences, 
    AudioSegment, AudioAnalysisResult
)
from ai_video_editor.modules.content_analysis import (
    FinancialBRollAnalyzer, AIGraphicsDirector, BRollOpportunity
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_context_with_audio() -> ContentContext:
    """Create sample ContentContext with financial education audio data."""
    
    # Create sample audio segments with financial content
    segments = [
        AudioSegment(
            text="Today we're going to talk about compound interest and how it can dramatically grow your investments over time. Let me show you some data on this.",
            start=10.0,
            end=18.0,
            confidence=0.95,
            financial_concepts=["compound interest", "investments", "growth"]
        ),
        AudioSegment(
            text="Diversification is a key strategy for managing risk in your portfolio. When you spread your investments across different asset classes, you reduce overall risk.",
            start=25.0,
            end=35.0,
            confidence=0.92,
            financial_concepts=["diversification", "risk management", "portfolio", "asset classes"]
        ),
        AudioSegment(
            text="Let's calculate how much your money would grow with a 7% annual return over 30 years. The formula for compound interest shows the power of time.",
            start=45.0,
            end=55.0,
            confidence=0.88,
            financial_concepts=["compound interest", "annual return", "formula"]
        ),
        AudioSegment(
            text="The process is simple: first, determine your risk tolerance, then allocate your assets accordingly, and finally, rebalance periodically.",
            start=70.0,
            end=80.0,
            confidence=0.91,
            financial_concepts=["risk tolerance", "asset allocation", "rebalancing"]
        )
    ]
    
    # Create comprehensive audio analysis result
    audio_analysis = AudioAnalysisResult(
        transcript_text=" ".join([seg.text for seg in segments]),
        segments=segments,
        overall_confidence=0.91,
        language="en",
        processing_time=2.5,
        model_used="whisper-base",
        financial_concepts=["compound interest", "diversification", "risk management", "portfolio"],
        complexity_level="intermediate"
    )
    
    # Create ContentContext
    context = ContentContext(
        project_id="financial_education_demo",
        video_files=["demo_video.mp4"],
        content_type=ContentType.EDUCATIONAL,
        user_preferences=UserPreferences(
            quality_mode="high",
            parallel_processing=True
        )
    )
    
    # Add audio analysis to context
    context.set_audio_analysis(audio_analysis)
    
    return context


async def demonstrate_broll_analysis():
    """Demonstrate B-roll opportunity detection and graphics generation."""
    
    print("🎬 B-Roll Analysis and Graphics Generation Demo")
    print("=" * 60)
    
    # Create sample context
    context = create_sample_context_with_audio()
    print(f"✅ Created ContentContext with {len(context.audio_analysis.segments)} audio segments")
    
    # Initialize B-Roll Analyzer
    broll_analyzer = FinancialBRollAnalyzer()
    print("✅ Initialized FinancialBRollAnalyzer")
    
    # Detect B-roll opportunities
    print("\n📊 Analyzing content for B-roll opportunities...")
    opportunities = broll_analyzer.detect_broll_opportunities(context)
    
    print(f"✅ Detected {len(opportunities)} B-roll opportunities:")
    for i, opp in enumerate(opportunities, 1):
        print(f"   {i}. {opp.opportunity_type} at {opp.timestamp}s ({opp.duration}s)")
        print(f"      Content: {opp.content[:80]}...")
        print(f"      Graphics: {opp.graphics_type} (confidence: {opp.confidence:.2f})")
        print(f"      Keywords: {', '.join(opp.keywords[:3])}...")
        print()
    
    # Get analysis statistics
    stats = broll_analyzer.get_analysis_stats()
    print(f"📈 Analysis Stats:")
    print(f"   - Analysis Time: {stats['analysis_time']:.2f}s")
    print(f"   - Opportunities Found: {stats['opportunities_detected']}")
    print(f"   - Trigger Categories: {stats['trigger_categories']}")
    
    # Initialize AI Graphics Director
    print("\n🎨 Initializing AI Graphics Director...")
    # Note: No API key provided, will use template-based generation
    graphics_director = AIGraphicsDirector(output_dir="temp/broll_demo")
    print("✅ AI Graphics Director initialized (template mode)")
    
    # Generate graphics for opportunities
    print("\n🎬 Generating graphics for B-roll opportunities...")
    generated_graphics = []
    
    for i, opportunity in enumerate(opportunities[:3]):  # Limit to first 3 for demo
        print(f"\n   Generating graphics for opportunity {i+1}...")
        try:
            graphics_files = await graphics_director.generate_contextual_graphics(
                opportunity, context
            )
            generated_graphics.extend(graphics_files)
            
            if graphics_files:
                print(f"   ✅ Generated {len(graphics_files)} graphics:")
                for file_path in graphics_files:
                    print(f"      - {Path(file_path).name}")
            else:
                print("   ⚠️  No graphics generated (may require matplotlib)")
        
        except Exception as e:
            print(f"   ❌ Error generating graphics: {e}")
    
    # Create motion graphics plans
    print("\n🎞️  Creating motion graphics plans...")
    for i, opportunity in enumerate(opportunities[:2]):
        motion_plan = graphics_director.create_movis_motion_graphics_plan(opportunity)
        print(f"   ✅ Motion plan {i+1}: {len(motion_plan['layers'])} layers, {len(motion_plan['animations'])} animations")
    
    # Create Blender animation scripts
    print("\n🎬 Creating Blender animation scripts...")
    for i, opportunity in enumerate(opportunities[:2]):
        script_path = graphics_director.create_blender_animation_script(opportunity)
        print(f"   ✅ Blender script {i+1}: {Path(script_path).name}")
    
    # Integration with AI Director format
    print("\n🤖 Converting to AI Director format...")
    ai_director_plans = broll_analyzer.integrate_with_ai_director(context, opportunities)
    print(f"✅ Created {len(ai_director_plans)} AI Director B-roll plans")
    
    # Store B-roll plans in ContentContext
    if not context.processed_video:
        context.processed_video = {}
    context.processed_video['broll_plans'] = ai_director_plans
    
    print(f"✅ B-roll plans stored in ContentContext")
    
    # Get graphics generation statistics
    graphics_stats = graphics_director.get_generation_stats()
    print(f"\n📊 Graphics Generation Stats:")
    print(f"   - Generation Time: {graphics_stats['generation_time']:.2f}s")
    print(f"   - Graphics Created: {graphics_stats['graphics_created']}")
    print(f"   - AI Enabled: {graphics_stats['ai_enabled']}")
    
    print("\n🎉 Task 6 Implementation Demo Completed Successfully!")
    print("\n📋 Summary:")
    print(f"   ✅ B-Roll Opportunities Detected: {len(opportunities)}")
    print(f"   ✅ Graphics Files Generated: {len(generated_graphics)}")
    print(f"   ✅ AI Director Plans Created: {len(ai_director_plans)}")
    print(f"   ✅ Motion Graphics Plans: {min(2, len(opportunities))}")
    print(f"   ✅ Blender Scripts: {min(2, len(opportunities))}")
    
    return {
        'opportunities': opportunities,
        'graphics_files': generated_graphics,
        'ai_director_plans': ai_director_plans,
        'context': context
    }


def demonstrate_specific_graphics():
    """Demonstrate specific graphics generation capabilities."""
    
    print("\n🎨 Specific Graphics Generation Demo")
    print("=" * 50)
    
    try:
        from ai_video_editor.modules.content_analysis.ai_graphics_director import (
            FinancialGraphicsGenerator, EducationalSlideGenerator
        )
        
        # Test financial graphics generation
        print("📊 Testing Financial Graphics Generator...")
        graphics_gen = FinancialGraphicsGenerator(output_dir="temp/demo_charts")
        
        try:
            # Test compound interest animation
            print("   Creating compound interest animation...")
            file_path = graphics_gen.create_compound_interest_animation(
                principal=10000, rate=0.07, years=20
            )
            print(f"   ✅ Created: {Path(file_path).name}")
        except Exception as e:
            print(f"   ⚠️  Compound interest animation: {e}")
        
        try:
            # Test diversification chart
            print("   Creating portfolio diversification chart...")
            portfolio = {
                'Stocks': 60.0,
                'Bonds': 25.0,
                'Real Estate': 10.0,
                'Cash': 5.0
            }
            file_path = graphics_gen.create_diversification_chart(portfolio)
            print(f"   ✅ Created: {Path(file_path).name}")
        except Exception as e:
            print(f"   ⚠️  Diversification chart: {e}")
        
        try:
            # Test risk-return scatter plot
            print("   Creating risk vs return analysis...")
            investments = {
                'Government Bonds': {'risk': 2, 'return': 3},
                'Index Funds': {'risk': 6, 'return': 7},
                'Individual Stocks': {'risk': 8, 'return': 9}
            }
            file_path = graphics_gen.create_risk_return_scatter(investments)
            print(f"   ✅ Created: {Path(file_path).name}")
        except Exception as e:
            print(f"   ⚠️  Risk-return chart: {e}")
        
        # Test educational slide generation
        print("\n📋 Testing Educational Slide Generator...")
        slide_gen = EducationalSlideGenerator(output_dir="temp/demo_slides")
        
        try:
            file_path = slide_gen.create_financial_concept_slide(
                concept="Compound Interest",
                explanation="The process where investment earnings generate additional earnings over time, creating exponential growth."
            )
            print(f"   ✅ Created concept slide: {Path(file_path).name}")
        except Exception as e:
            print(f"   ⚠️  Concept slide: {e}")
        
        print("✅ Graphics generation tests completed")
        
    except ImportError as e:
        print(f"⚠️  Graphics generation requires matplotlib: {e}")
        print("   Install with: pip install matplotlib")


async def main():
    """Main demonstration function."""
    
    try:
        # Run B-roll analysis demo
        results = await demonstrate_broll_analysis()
        
        # Run specific graphics demo
        demonstrate_specific_graphics()
        
        print("\n🎯 Task 6 Implementation Results:")
        print("✅ Task 6.1: B-Roll Opportunity Analysis - COMPLETED")
        print("✅ Task 6.2: Graphics and Animation Planning - COMPLETED")
        print("\n🚀 Ready for integration with Phase 3 Output Generation!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
