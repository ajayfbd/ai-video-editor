# tests/validation/validate_mock_consistency.py
"""
Mock data consistency validation for pre-commit hooks.
Ensures mock data remains consistent and realistic across all test modules.
"""

import sys
import json
import traceback
from typing import Dict, List, Any, Set
from pathlib import Path

from tests.data.sample_data import get_sample_data
from tests.mocks.api_mocks import GeminiAPIMock, ImagenAPIMock, WhisperAPIMock


class MockConsistencyValidator:
    """Validates consistency of mock data across the test suite."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.test_data_types = [
            "video_properties", "transcript", "gemini_response", 
            "imagen_response", "expected_thumbnails", "expected_metadata"
        ]
        self.content_types = ["educational", "music", "general"]
    
    def validate_all(self) -> bool:
        """Run all mock consistency validations."""
        print("🔍 Validating mock data consistency...")
        
        try:
            self.validate_sample_data_completeness()
            self.validate_data_structure_consistency()
            self.validate_api_mock_responses()
            self.validate_cross_reference_integrity()
            self.validate_performance_benchmarks()
            self.validate_error_scenarios()
            
            return self.report_results()
            
        except Exception as e:
            self.errors.append(f"Mock validation failed with exception: {str(e)}")
            print(f"❌ Validation failed: {e}")
            traceback.print_exc()
            return False
    
    def validate_sample_data_completeness(self):
        """Validate that all required sample data exists for all content types."""
        print("Checking sample data completeness...")
        
        for data_type in self.test_data_types:
            if data_type in ["performance_benchmarks", "error_scenarios"]:
                continue  # These don't vary by content type
                
            for content_type in self.content_types:
                try:
                    data = get_sample_data(data_type, content_type)
                    if not data:
                        self.errors.append(f"Empty sample data for {data_type}/{content_type}")
                except Exception as e:
                    self.errors.append(f"Missing sample data for {data_type}/{content_type}: {str(e)}")
        
        print("✅ Sample data completeness checked")
    
    def validate_data_structure_consistency(self):
        """Validate that data structures are consistent across content types."""
        print("Checking data structure consistency...")
        
        # Check video properties structure
        video_structures = {}
        for content_type in self.content_types:
            try:
                props = get_sample_data("video_properties", content_type)
                video_structures[content_type] = set(props.keys())
            except Exception as e:
                self.errors.append(f"Failed to get video properties for {content_type}: {str(e)}")
                continue
        
        # All video properties should have the same structure
        if len(set(frozenset(s) for s in video_structures.values())) > 1:
            self.warnings.append("Video properties have inconsistent structures across content types")
            for content_type, structure in video_structures.items():
                print(f"  {content_type}: {sorted(structure)}")
        
        # Check transcript structure consistency
        transcript_structures = {}
        for content_type in self.content_types:
            try:
                transcript = get_sample_data("transcript", content_type)
                transcript_structures[content_type] = set(transcript.keys())
            except Exception as e:
                self.errors.append(f"Failed to get transcript for {content_type}: {str(e)}")
                continue
        
        if len(set(frozenset(s) for s in transcript_structures.values())) > 1:
            self.warnings.append("Transcript structures are inconsistent across content types")
        
        print("✅ Data structure consistency checked")
    
    def validate_api_mock_responses(self):
        """Validate that API mocks produce consistent and realistic responses."""
        print("Checking API mock responses...")
        
        # Test Gemini API mock
        try:
            gemini_mock = GeminiAPIMock()
            
            # Test content analysis
            response = gemini_mock.analyze_content("Test financial education content", "educational")
            required_keys = ["content_analysis", "trending_keywords", "emotional_analysis", "competitor_insights"]
            
            for key in required_keys:
                if key not in response:
                    self.errors.append(f"Gemini mock missing required key: {key}")
            
            # Validate response structure
            if "content_analysis" in response:
                content_analysis = response["content_analysis"]
                if "key_concepts" not in content_analysis or not isinstance(content_analysis["key_concepts"], list):
                    self.errors.append("Gemini mock content_analysis.key_concepts should be a list")
            
        except Exception as e:
            self.errors.append(f"Gemini API mock validation failed: {str(e)}")
        
        # Test Imagen API mock
        try:
            imagen_mock = ImagenAPIMock()
            response = imagen_mock.generate_background("Test prompt", "modern_professional")
            
            required_keys = ["image_url", "image_data", "generation_metadata", "quality_metrics"]
            for key in required_keys:
                if key not in response:
                    self.errors.append(f"Imagen mock missing required key: {key}")
            
            # Validate cost tracking
            if imagen_mock.get_total_cost() <= 0:
                self.errors.append("Imagen mock should track costs")
            
        except Exception as e:
            self.errors.append(f"Imagen API mock validation failed: {str(e)}")
        
        # Test Whisper API mock
        try:
            whisper_mock = WhisperAPIMock()
            response = whisper_mock.transcribe("test_audio.wav", "en")
            
            required_keys = ["text", "segments", "language", "confidence"]
            for key in required_keys:
                if key not in response:
                    self.errors.append(f"Whisper mock missing required key: {key}")
            
            # Validate segments structure
            if "segments" in response and response["segments"]:
                segment = response["segments"][0]
                segment_keys = ["text", "start_time", "end_time", "confidence"]
                for key in segment_keys:
                    if key not in segment:
                        self.errors.append(f"Whisper mock segment missing key: {key}")
            
        except Exception as e:
            self.errors.append(f"Whisper API mock validation failed: {str(e)}")
        
        print("✅ API mock responses checked")
    
    def validate_cross_reference_integrity(self):
        """Validate that cross-references between mock data are consistent."""
        print("Checking cross-reference integrity...")
        
        for content_type in self.content_types:
            try:
                # Get related data
                transcript = get_sample_data("transcript", content_type)
                gemini_response = get_sample_data("gemini_response", content_type)
                expected_thumbnails = get_sample_data("expected_thumbnails", content_type)
                expected_metadata = get_sample_data("expected_metadata", content_type)
                
                # Check that emotional markers in transcript align with Gemini response
                transcript_emotions = set()
                for segment in transcript.get("segments", []):
                    for marker in segment.get("emotional_markers", []):
                        transcript_emotions.add(marker["emotion"])
                
                gemini_emotions = set()
                for peak in gemini_response.get("emotional_analysis", {}).get("peaks", []):
                    gemini_emotions.add(peak["emotion"])
                
                # Some overlap expected but not required to be identical
                if transcript_emotions and gemini_emotions and not transcript_emotions & gemini_emotions:
                    self.warnings.append(f"No emotional overlap between transcript and Gemini response for {content_type}")
                
                # Check that thumbnail concepts align with metadata themes
                thumbnail_themes = set()
                for concept in expected_thumbnails:
                    thumbnail_themes.add(concept.get("emotional_theme", ""))
                
                # Check keyword consistency
                gemini_keywords = set(gemini_response.get("trending_keywords", {}).get("primary_keywords", []))
                
                for metadata in expected_metadata:
                    metadata_keywords = set(metadata.get("tags", []))
                    if gemini_keywords and metadata_keywords and not gemini_keywords & metadata_keywords:
                        self.warnings.append(f"No keyword overlap between Gemini and metadata for {content_type}")
                
            except Exception as e:
                self.errors.append(f"Cross-reference validation failed for {content_type}: {str(e)}")
        
        print("✅ Cross-reference integrity checked")
    
    def validate_performance_benchmarks(self):
        """Validate performance benchmark data."""
        print("Checking performance benchmarks...")
        
        try:
            benchmarks = get_sample_data("performance_benchmarks")
            
            required_content_types = ["educational_content", "music_content", "general_content"]
            for content_type in required_content_types:
                if content_type not in benchmarks:
                    self.errors.append(f"Missing performance benchmark for {content_type}")
                    continue
                
                benchmark = benchmarks[content_type]
                required_metrics = ["max_processing_time", "max_memory_usage", "target_api_cost", "expected_quality_score"]
                
                for metric in required_metrics:
                    if metric not in benchmark:
                        self.errors.append(f"Missing benchmark metric {metric} for {content_type}")
                    elif not isinstance(benchmark[metric], (int, float)):
                        self.errors.append(f"Benchmark metric {metric} should be numeric for {content_type}")
            
            # Validate that benchmarks are realistic
            if "educational_content" in benchmarks:
                edu_benchmark = benchmarks["educational_content"]
                if edu_benchmark.get("max_processing_time", 0) > 3600:  # 1 hour
                    self.warnings.append("Educational content processing time benchmark seems too high")
                if edu_benchmark.get("target_api_cost", 0) > 10:  # $10
                    self.warnings.append("Educational content API cost benchmark seems too high")
        
        except Exception as e:
            self.errors.append(f"Performance benchmark validation failed: {str(e)}")
        
        print("✅ Performance benchmarks checked")
    
    def validate_error_scenarios(self):
        """Validate error scenario definitions."""
        print("Checking error scenarios...")
        
        try:
            error_scenarios = get_sample_data("error_scenarios")
            
            required_categories = ["api_failures", "resource_constraints", "data_corruption"]
            for category in required_categories:
                if category not in error_scenarios:
                    self.errors.append(f"Missing error scenario category: {category}")
                    continue
                
                scenarios = error_scenarios[category]
                if not isinstance(scenarios, dict) or not scenarios:
                    self.errors.append(f"Error scenario category {category} should be non-empty dict")
                    continue
                
                # Validate scenario structure
                for scenario_name, scenario_data in scenarios.items():
                    if "error_type" not in scenario_data:
                        self.errors.append(f"Error scenario {category}.{scenario_name} missing error_type")
                    
                    if "expected_behavior" not in scenario_data and "expected_fallback" not in scenario_data:
                        self.errors.append(f"Error scenario {category}.{scenario_name} missing expected behavior")
        
        except Exception as e:
            self.errors.append(f"Error scenario validation failed: {str(e)}")
        
        print("✅ Error scenarios checked")
    
    def report_results(self) -> bool:
        """Report validation results."""
        print("\n" + "="*60)
        print("MOCK CONSISTENCY VALIDATION RESULTS")
        print("="*60)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ All mock consistency validations passed!")
        elif not self.errors:
            print(f"\n✅ Mock consistency validation passed with {len(self.warnings)} warnings")
        
        print("="*60)
        
        return len(self.errors) == 0


def main():
    """Main validation function for pre-commit hook."""
    validator = MockConsistencyValidator()
    success = validator.validate_all()
    
    if not success:
        print("\n💥 Mock consistency validation failed!")
        print("Please fix the errors above before committing.")
        sys.exit(1)
    else:
        print("\n🎉 Mock consistency validation passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()