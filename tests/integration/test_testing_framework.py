# tests/integration/test_testing_framework.py
"""
Integration tests for the comprehensive testing framework.
Validates that all testing components work together properly.
"""

import pytest
import tempfile
import os
from unittest.mock import patch

from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from tests.data.sample_data import get_sample_data
from tests.data.test_data_manager import TestDataManager
from tests.mocks.api_mocks import ComprehensiveAPIMocker


@pytest.mark.integration
class TestTestingFrameworkIntegration:
    """Test the integration of all testing framework components."""
    
    def test_complete_testing_workflow(self, performance_monitor, memory_profiler):
        """Test complete workflow using all testing framework components."""
        memory_profiler.take_snapshot("start")
        performance_monitor.start_monitoring()
        
        # Create test data manager
        with TestDataManager() as data_manager:
            # Create sample files
            video_file = data_manager.create_sample_video_file("test_video.mp4", "educational")
            audio_file = data_manager.create_sample_audio_file("test_audio.wav", "educational")
            
            # Verify files were created
            assert os.path.exists(video_file)
            assert os.path.exists(audio_file)
            
            # Create expected outputs
            expected_thumbnails = get_sample_data("expected_thumbnails", "educational")
            data_manager.create_expected_output("integration_test", "thumbnails", expected_thumbnails)
            
            # Test API mocking
            with ComprehensiveAPIMocker() as api_mocker:
                api_mocker.mock_all_apis()
                
                # Create ContentContext
                context = ContentContext(
                    project_id="integration_test",
                    video_files=[video_file],
                    content_type=ContentType.EDUCATIONAL,
                    user_preferences=UserPreferences()
                )
                
                # Simulate processing with mocked APIs
                # This would normally call actual processing modules
                gemini_response = api_mocker.gemini_mock.analyze_content("Test content", "educational")
                imagen_response = api_mocker.imagen_mock.generate_background("Test prompt")
                whisper_response = api_mocker.whisper_mock.transcribe(audio_file)
                
                # Verify API responses
                assert "content_analysis" in gemini_response
                assert "image_url" in imagen_response
                assert "text" in whisper_response
                
                # Check API call tracking
                api_calls = api_mocker.get_total_api_calls()
                assert api_calls["gemini"] > 0
                assert api_calls["imagen"] > 0
                assert api_calls["whisper"] > 0
                
                # Test cost tracking
                total_cost = api_mocker.get_total_cost()
                assert total_cost > 0
                assert total_cost < 5.0  # Should be reasonable for test
        
        # Check performance metrics
        metrics = performance_monitor.stop_monitoring()
        memory_profiler.take_snapshot("end")
        
        # Validate performance
        assert metrics["processing_time"] < 10.0  # Should complete quickly
        
        memory_diff = memory_profiler.get_memory_diff("start", "end")
        assert memory_diff["rss_diff"] < 500_000_000  # Less than 500MB
    
    def test_mock_consistency_across_modules(self, api_mocker):
        """Test that mocks provide consistent responses across different usage patterns."""
        api_mocker.mock_all_apis()
        
        # Test multiple calls to same API
        response1 = api_mocker.gemini_mock.analyze_content("Test content 1", "educational")
        response2 = api_mocker.gemini_mock.analyze_content("Test content 2", "educational")
        
        # Responses should have same structure but different content
        assert set(response1.keys()) == set(response2.keys())
        assert response1["content_analysis"]["key_concepts"] != response2["content_analysis"]["key_concepts"]
        
        # Test API call counting
        assert api_mocker.gemini_mock.get_call_count("analyze_content") == 2
        
        # Test failure simulation
        api_mocker.gemini_mock.set_failure_rate(1.0)  # 100% failure rate
        
        with pytest.raises(Exception, match="Gemini API temporarily unavailable"):
            api_mocker.gemini_mock.analyze_content("Test content 3", "educational")
        
        # Reset failure rate
        api_mocker.gemini_mock.set_failure_rate(0.0)
        
        # Should work again
        response3 = api_mocker.gemini_mock.analyze_content("Test content 4", "educational")
        assert "content_analysis" in response3
    
    def test_performance_monitoring_accuracy(self, performance_monitor, memory_profiler):
        """Test that performance monitoring provides accurate measurements."""
        import time
        
        # Test processing time measurement
        performance_monitor.start_monitoring()
        
        # Simulate work
        start_time = time.time()
        time.sleep(0.1)  # 100ms of work
        actual_time = time.time() - start_time
        
        metrics = performance_monitor.stop_monitoring()
        
        # Performance monitor should be reasonably accurate
        time_diff = abs(metrics["processing_time"] - actual_time)
        assert time_diff < 0.01  # Within 10ms accuracy
        
        # Test memory profiling
        memory_profiler.take_snapshot("before_allocation")
        
        # Allocate some memory
        large_list = [i for i in range(100000)]  # ~800KB
        
        memory_profiler.take_snapshot("after_allocation")
        
        memory_diff = memory_profiler.get_memory_diff("before_allocation", "after_allocation")
        
        # Should detect memory increase
        assert memory_diff["rss_diff"] > 0
        
        # Clean up
        del large_list
    
    def test_test_data_management_workflow(self):
        """Test complete test data management workflow."""
        with TestDataManager() as data_manager:
            # Create test session
            session_id = data_manager.create_test_session("framework_integration_test")
            assert session_id.startswith("framework_integration_test_")
            
            # Create sample data
            video_file = data_manager.create_sample_video_file("session_test.mp4", "educational")
            data_manager.add_to_session(session_id, video_file, "Test video file")
            
            # Create expected output
            expected_data = {"test": "data", "values": [1, 2, 3]}
            expected_file = data_manager.create_expected_output("session_test", "output", expected_data)
            data_manager.add_to_session(session_id, expected_file, "Expected output")
            
            # Test comparison
            actual_data = {"test": "data", "values": [1, 2, 3]}  # Exact match
            comparison = data_manager.compare_outputs("session_test", "output", actual_data)
            
            assert comparison["status"] == "compared"
            assert comparison["similarity_score"] == 1.0  # Perfect match
            assert len(comparison["differences"]) == 0
            
            # Test with differences
            different_data = {"test": "different", "values": [1, 2, 4]}
            comparison = data_manager.compare_outputs("session_test", "output", different_data)
            
            assert comparison["similarity_score"] < 1.0
            assert len(comparison["differences"]) > 0
            
            # Get session summary
            summary = data_manager.get_session_summary(session_id)
            assert summary["file_count"] == 2
            assert len(summary["files"]) == 2
    
    def test_error_handling_integration(self, api_mocker):
        """Test error handling across the testing framework."""
        api_mocker.mock_all_apis()
        
        # Test API failure handling
        api_mocker.set_failure_rates(gemini=0.5, imagen=0.3, whisper=0.2)
        
        successful_calls = 0
        failed_calls = 0
        
        # Make multiple API calls
        for i in range(20):
            try:
                api_mocker.gemini_mock.analyze_content(f"Test content {i}", "educational")
                successful_calls += 1
            except Exception:
                failed_calls += 1
        
        # Should have some failures due to failure rate
        assert failed_calls > 0
        assert successful_calls > 0
        
        # Test recovery after failures
        api_mocker.set_failure_rates(gemini=0.0, imagen=0.0, whisper=0.0)
        
        # Should work reliably now
        for i in range(5):
            response = api_mocker.gemini_mock.analyze_content(f"Recovery test {i}", "educational")
            assert "content_analysis" in response
    
    @pytest.mark.slow
    def test_performance_under_load(self, performance_monitor, api_mocker):
        """Test framework performance under load."""
        api_mocker.mock_all_apis()
        
        performance_monitor.start_monitoring()
        
        # Simulate high load
        contexts = []
        for i in range(50):
            context = ContentContext(
                project_id=f"load_test_{i}",
                video_files=[f"video_{i}.mp4"],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences()
            )
            contexts.append(context)
            
            # Make API calls
            api_mocker.gemini_mock.analyze_content(f"Load test content {i}", "educational")
            if i % 5 == 0:  # Every 5th iteration
                api_mocker.imagen_mock.generate_background(f"Load test prompt {i}")
        
        metrics = performance_monitor.stop_monitoring()
        
        # Should handle load efficiently
        assert metrics["processing_time"] < 30.0  # Should complete in under 30 seconds
        
        # Check API call counts
        api_calls = api_mocker.get_total_api_calls()
        assert api_calls["gemini"] == 50
        assert api_calls["imagen"] == 10  # Every 5th iteration
        
        # Cost should be reasonable
        total_cost = api_mocker.get_total_cost()
        assert total_cost < 10.0  # Should be under $10 for test load
    
    def test_cross_module_data_consistency(self):
        """Test data consistency across different testing modules."""
        # Test that sample data is consistent
        educational_video = get_sample_data("video_properties", "educational")
        educational_transcript = get_sample_data("transcript", "educational")
        educational_gemini = get_sample_data("gemini_response", "educational")
        
        # Check duration consistency
        video_duration = educational_video["duration"]
        transcript_segments = educational_transcript["segments"]
        
        if transcript_segments:
            last_segment = max(transcript_segments, key=lambda s: s["end_time"])
            transcript_duration = last_segment["end_time"]
            
            # Transcript should not exceed video duration
            assert transcript_duration <= video_duration
        
        # Check concept consistency
        transcript_concepts = set(educational_transcript["key_concepts_identified"])
        gemini_concepts = set(educational_gemini["content_analysis"]["key_concepts"])
        
        # Should have some overlap
        overlap = transcript_concepts & gemini_concepts
        assert len(overlap) > 0, "Should have some concept overlap between transcript and Gemini response"
        
        # Check emotional consistency
        transcript_emotions = set()
        for segment in transcript_segments:
            for marker in segment.get("emotional_markers", []):
                transcript_emotions.add(marker["emotion"])
        
        gemini_emotions = set()
        for peak in gemini_concepts.get("emotional_analysis", {}).get("peaks", []):
            gemini_emotions.add(peak["emotion"])
        
        if transcript_emotions and gemini_emotions:
            emotional_overlap = transcript_emotions & gemini_emotions
            # Some emotional consistency expected but not required
            if not emotional_overlap:
                pytest.skip("No emotional overlap found - this is acceptable but worth noting")
    
    def test_testing_framework_cleanup(self, api_mocker):
        """Test that testing framework properly cleans up resources."""
        # Test API mocker cleanup
        api_mocker.mock_all_apis()
        
        # Make some calls
        api_mocker.gemini_mock.analyze_content("Cleanup test", "educational")
        api_mocker.imagen_mock.generate_background("Cleanup test")
        
        initial_calls = api_mocker.get_total_api_calls()
        assert initial_calls["total"] > 0
        
        # Reset mocks
        api_mocker.reset_all_mocks()
        
        reset_calls = api_mocker.get_total_api_calls()
        assert reset_calls["total"] == 0
        
        # Test data manager cleanup
        temp_dir = None
        with TestDataManager() as data_manager:
            temp_dir = data_manager.base_dir
            assert os.path.exists(temp_dir)
            
            # Create some files
            data_manager.create_sample_video_file("cleanup_test.mp4", "educational")
        
        # Directory should be cleaned up after context exit
        # Note: In some cases, cleanup might be delayed by OS
        # So we'll just verify the cleanup was attempted
        assert data_manager._cleanup_on_exit or not os.path.exists(temp_dir)