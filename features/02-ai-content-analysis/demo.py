#!/usr/bin/env python3
"""
🧠 AI Content Analysis Demo

This demo shows how to use Gemini AI to analyze video content and extract
key insights like concepts, emotions, themes, and keywords.

Since your Gemini API is working, this should run immediately!
"""

import os
import json
import google.generativeai as genai
from typing import Dict, List, Any


class AIContentAnalyzer:
    """Analyzes content using Gemini AI to extract insights."""
    
    def __init__(self):
        """Initialize with Gemini API."""
        # Load environment variables from .env file
        from pathlib import Path
        import sys
        
        # Add project root to path to find .env
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        # Try to load .env file
        try:
            from dotenv import load_dotenv
            load_dotenv(project_root / '.env')
        except ImportError:
            pass  # dotenv not available, rely on environment
        
        api_key = os.getenv('AI_VIDEO_EDITOR_GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Please set AI_VIDEO_EDITOR_GEMINI_API_KEY in your .env file")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✅ AI Content Analyzer initialized with Gemini API")
    
    def analyze_content(self, content_text: str) -> Dict[str, Any]:
        """Analyze content and extract key insights."""
        
        prompt = f"""
        Analyze this video content and extract key insights in JSON format:

        CONTENT TO ANALYZE:
        {content_text}

        Please provide a JSON response with these fields:
        {{
            "key_concepts": ["list of main topics and concepts"],
            "emotional_peaks": [
                {{"timestamp_estimate": 30, "emotion": "excitement", "intensity": 0.8, "reason": "why this emotion"}}
            ],
            "content_themes": ["main themes and categories"],
            "trending_keywords": ["SEO-friendly keywords"],
            "audience_insights": {{
                "target_audience": "who would be interested",
                "engagement_potential": "high/medium/low",
                "recommended_platforms": ["best social platforms"],
                "content_type": "educational/entertainment/business/etc"
            }},
            "summary": "brief summary of the content"
        }}

        Focus on actionable insights that would help with video editing, thumbnail creation, and metadata generation.
        """
        
        try:
            print("🤖 Analyzing content with Gemini AI...")
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            response_text = response.text
            
            # Find JSON in the response (sometimes it's wrapped in markdown)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text
            
            # Parse JSON
            analysis = json.loads(json_text)
            print("✅ Content analysis completed!")
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing AI response as JSON: {e}")
            print(f"Raw response: {response.text}")
            return {"error": "Failed to parse AI response", "raw_response": response.text}
        except Exception as e:
            print(f"❌ Error analyzing content: {e}")
            return {"error": str(e)}


def demo_with_sample_content():
    """Demo with sample video content."""
    
    # Sample content (like a video transcript)
    sample_content = """
    Welcome to this tutorial on artificial intelligence and video editing! 
    Today we're going to explore how AI can revolutionize content creation.
    
    First, let's talk about the challenges content creators face. Creating engaging videos 
    takes hours of editing, thumbnail design, and optimization. But what if AI could help?
    
    I'm excited to show you how machine learning can automatically detect the best moments 
    in your video, create stunning thumbnails, and even write compelling descriptions.
    
    This technology is game-changing for YouTubers, marketers, and anyone who creates video content.
    By the end of this video, you'll understand how to leverage AI tools to 10x your productivity.
    
    Let's dive in and see some amazing examples of AI-powered video editing in action!
    """
    
    print("🎬 AI Content Analysis Demo")
    print("=" * 50)
    print(f"📝 Sample Content:\n{sample_content[:200]}...\n")
    
    try:
        # Initialize analyzer
        analyzer = AIContentAnalyzer()
        
        # Analyze content
        results = analyzer.analyze_content(sample_content)
        
        # Display results
        print("📊 Analysis Results:")
        print("=" * 50)
        print(json.dumps(results, indent=2))
        
        # Highlight key insights
        if "key_concepts" in results:
            print(f"\n🎯 Key Concepts: {', '.join(results['key_concepts'][:3])}")
        
        if "audience_insights" in results:
            audience = results["audience_insights"]
            print(f"👥 Target Audience: {audience.get('target_audience', 'N/A')}")
            print(f"📈 Engagement Potential: {audience.get('engagement_potential', 'N/A')}")
        
        if "trending_keywords" in results:
            print(f"🔍 Top Keywords: {', '.join(results['trending_keywords'][:3])}")
        
        print("\n✅ Demo completed successfully!")
        print("\n💡 Next Steps:")
        print("   - Try with your own content")
        print("   - Use results for metadata generation")
        print("   - Combine with other features")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Check your Gemini API key in .env file")
        print("   - Ensure internet connection")
        print("   - Try running: python test.py")


def demo_with_custom_content():
    """Demo with user-provided content."""
    
    print("🎬 AI Content Analysis - Custom Content")
    print("=" * 50)
    
    content = input("📝 Enter your video content (transcript, description, etc.):\n")
    
    if not content.strip():
        print("❌ No content provided. Using sample content instead.")
        demo_with_sample_content()
        return
    
    try:
        analyzer = AIContentAnalyzer()
        results = analyzer.analyze_content(content)
        
        print("\n📊 Your Content Analysis:")
        print("=" * 50)
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


if __name__ == "__main__":
    print("🧠 AI Content Analysis Feature Demo")
    print("=" * 50)
    print("Choose an option:")
    print("1. Demo with sample content (recommended)")
    print("2. Demo with your own content")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        demo_with_custom_content()
    else:
        demo_with_sample_content()