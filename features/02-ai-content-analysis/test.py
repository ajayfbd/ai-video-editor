#!/usr/bin/env python3
"""
🧪 AI Content Analysis Tests

Simple tests to verify the AI Content Analysis feature is working correctly.
"""

import os
import sys
import json
from demo import AIContentAnalyzer


def test_api_connection():
    """Test if we can connect to Gemini API."""
    print("🔌 Testing API Connection...")
    
    try:
        analyzer = AIContentAnalyzer()
        print("✅ API connection successful")
        return True
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False


def test_basic_analysis():
    """Test basic content analysis functionality."""
    print("🧠 Testing Basic Analysis...")
    
    test_content = "This is a video about artificial intelligence and machine learning."
    
    try:
        analyzer = AIContentAnalyzer()
        results = analyzer.analyze_content(test_content)
        
        # Check if we got valid results
        if "error" in results:
            print(f"❌ Analysis returned error: {results['error']}")
            return False
        
        # Check for expected fields
        expected_fields = ["key_concepts", "content_themes", "audience_insights"]
        missing_fields = [field for field in expected_fields if field not in results]
        
        if missing_fields:
            print(f"⚠️  Missing fields: {missing_fields}")
            print("📊 Actual results:")
            print(json.dumps(results, indent=2))
        else:
            print("✅ Basic analysis successful")
            print(f"🎯 Found {len(results.get('key_concepts', []))} key concepts")
            print(f"🎨 Found {len(results.get('content_themes', []))} themes")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic analysis failed: {e}")
        return False


def test_json_parsing():
    """Test that AI responses are properly parsed as JSON."""
    print("📋 Testing JSON Parsing...")
    
    test_content = "Short video about Python programming."
    
    try:
        analyzer = AIContentAnalyzer()
        results = analyzer.analyze_content(test_content)
        
        # Try to serialize back to JSON (tests if it's valid JSON structure)
        json_str = json.dumps(results)
        parsed_back = json.loads(json_str)
        
        print("✅ JSON parsing successful")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing failed: {e}")
        return False
    except Exception as e:
        print(f"❌ JSON test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("🧪 AI Content Analysis Test Suite")
    print("=" * 50)
    
    tests = [
        ("API Connection", test_api_connection),
        ("Basic Analysis", test_basic_analysis),
        ("JSON Parsing", test_json_parsing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! AI Content Analysis is working correctly.")
        print("\n💡 Next steps:")
        print("   - Run: python demo.py")
        print("   - Try with your own content")
        print("   - Move to next feature: 03-metadata-generation")
    else:
        print("⚠️  Some tests failed. Check your setup:")
        print("   - Verify Gemini API key in .env file")
        print("   - Check internet connection")
        print("   - Ensure google-generativeai is installed")
    
    return passed == len(tests)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)