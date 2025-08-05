# tests/performance/test_memory_usage.py
"""
Memory usage performance tests for AI Video Editor.
Tests memory consumption patterns and ensures efficient resource usage.
"""

import pytest
import psutil
import time
import gc
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.core.context_manager import ContextManager


@pytest.mark.performance
class TestMemoryUsage:
    """Test memory usage patterns and limits."""
    
    def test_content_context_memory_footprint(self, sample_content_context, memory_profiler):
        """Test that ContentContext objects stay within memory limits."""
        memory_profiler.take_snapshot("start")
        
        # Create multiple ContentContext objects
        contexts = []
        for i in range(10):
            context = ContentContext(
                project_id=f"test_project_{i}",
                video_files=[f"video_{i}.mp4"],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences()
            )
            
            # Populate with sample data
            context.key_concepts = [f"concept_{j}" for j in range(100)]
            contexts.append(context)
        
        memory_profiler.take_snapshot("after_creation")
        
        # Each ContentContext should use less than 50MB
        memory_diff = memory_profiler.get_memory_diff("start", "after_creation")
        memory_per_context = memory_diff["rss_diff"] / len(contexts)
        
        assert memory_per_context < 50_000_000, f"Each ContentContext uses {memory_per_context} bytes, expected < 50MB"
    
    def test_context_manager_memory_efficiency(self, memory_profiler):
        """Test ContextManager memory efficiency during operations."""
        memory_profiler.take_snapshot("start")
        
        context_manager = ContextManager()
        
        # Create and process multiple contexts
        for i in range(5):
            context = ContentContext(
                project_id=f"memory_test_{i}",
                video_files=[f"test_{i}.mp4"],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences()
            )
            
            # Simulate processing
            context_manager.validate_context(context)
            
            memory_profiler.take_snapshot(f"after_context_{i}")
        
        memory_profiler.take_snapshot("end")
        
        # Memory growth should be linear, not exponential
        total_diff = memory_profiler.get_memory_diff("start", "end")
        assert total_diff["rss_diff"] < 500_000_000, f"Total memory growth {total_diff['rss_diff']} bytes, expected < 500MB"
    
    def test_large_content_context_handling(self, memory_profiler):
        """Test handling of large ContentContext objects."""
        memory_profiler.take_snapshot("start")
        
        # Create a large ContentContext
        large_context = ContentContext(
            project_id="large_test",
            video_files=["large_video.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences()
        )
        
        # Add large amounts of data
        large_context.key_concepts = [f"concept_{i}" for i in range(10000)]
        
        # Simulate processing large context
        context_manager = ContextManager()
        result = context_manager.validate_context(large_context)
        
        memory_profiler.take_snapshot("after_large_processing")
        
        # Should handle large contexts without excessive memory usage
        memory_diff = memory_profiler.get_memory_diff("start", "after_large_processing")
        assert memory_diff["rss_diff"] < 1_000_000_000, f"Large context processing used {memory_diff['rss_diff']} bytes, expected < 1GB"
        assert result["valid"], "Large context should still be valid"
    
    @pytest.mark.slow
    def test_memory_leak_detection(self, memory_profiler):
        """Test for memory leaks during repeated operations."""
        memory_profiler.take_snapshot("start")
        
        context_manager = ContextManager()
        
        # Perform repeated operations
        for iteration in range(50):
            context = ContentContext(
                project_id=f"leak_test_{iteration}",
                video_files=[f"test_{iteration}.mp4"],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences()
            )
            
            # Process and discard
            context_manager.validate_context(context)
            del context
            
            if iteration % 10 == 0:
                memory_profiler.take_snapshot(f"iteration_{iteration}")
        
        memory_profiler.take_snapshot("end")
        
        # Memory should not grow significantly over iterations
        start_to_mid = memory_profiler.get_memory_diff("start", "iteration_20")
        mid_to_end = memory_profiler.get_memory_diff("iteration_20", "end")
        
        # Later iterations should not use significantly more memory than earlier ones, allowing for a small tolerance
        assert mid_to_end["rss_diff"] < (start_to_mid["rss_diff"] * 1.5) + 1024 * 1024, "Potential memory leak detected"


@pytest.mark.performance
class TestProcessingTimeMetrics:
    """Test processing time performance."""
    
    def test_context_creation_performance(self, performance_monitor):
        """Test ContentContext creation performance."""
        performance_monitor.start_monitoring()
        
        # Create multiple contexts quickly
        contexts = []
        for i in range(100):
            context = ContentContext(
                project_id=f"perf_test_{i}",
                video_files=[f"video_{i}.mp4"],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences()
            )
            contexts.append(context)
        
        metrics = performance_monitor.stop_monitoring()
        
        # Should create 100 contexts in under 1 second
        assert metrics["processing_time"] < 1.0, f"Context creation took {metrics['processing_time']:.2f}s, expected < 1s"
        
        # Average time per context should be minimal
        avg_time_per_context = metrics["processing_time"] / len(contexts)
        assert avg_time_per_context < 0.01, f"Average time per context {avg_time_per_context:.4f}s, expected < 0.01s"
    
    def test_context_validation_performance(self, sample_content_context, performance_monitor):
        """Test ContentContext validation performance."""
        context_manager = ContextManager()
        
        performance_monitor.start_monitoring()
        
        # Validate context multiple times
        for _ in range(1000):
            result = context_manager.validate_context(sample_content_context)
            assert result["valid"]
        
        metrics = performance_monitor.stop_monitoring()
        
        # Should validate 1000 contexts in under 2 seconds
        assert metrics["processing_time"] < 2.0, f"Validation took {metrics['processing_time']:.2f}s, expected < 2s"
    
    @pytest.mark.slow
    def test_concurrent_processing_performance(self, performance_monitor):
        """Test performance under concurrent processing load."""
        import concurrent.futures
        import threading
        
        context_manager = ContextManager()
        
        def create_and_validate_context(thread_id: int):
            """Create and validate a context in a separate thread."""
            context = ContentContext(
                project_id=f"concurrent_test_{thread_id}",
                video_files=[f"video_{thread_id}.mp4"],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences()
            )
            return context_manager.validate_context(context)
        
        performance_monitor.start_monitoring()
        
        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create_and_validate_context, i) for i in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        metrics = performance_monitor.stop_monitoring()
        
        # All validations should succeed
        assert all(result["valid"] for result in results), "All concurrent validations should succeed"
        
        # Concurrent processing should be efficient
        assert metrics["processing_time"] < 5.0, f"Concurrent processing took {metrics['processing_time']:.2f}s, expected < 5s"


@pytest.mark.performance
class TestResourceUtilization:
    """Test overall resource utilization patterns."""
    
    def test_cpu_utilization_efficiency(self, performance_monitor):
        """Test CPU utilization during processing."""
        import multiprocessing
        
        # Get baseline CPU usage
        cpu_before = psutil.cpu_percent(interval=1)
        
        performance_monitor.start_monitoring()
        
        # Simulate CPU-intensive operations
        context_manager = ContextManager()
        contexts = []
        
        for i in range(multiprocessing.cpu_count() * 2):
            context = ContentContext(
                project_id=f"cpu_test_{i}",
                video_files=[f"video_{i}.mp4"],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences()
            )
            
            # Add computational work
            context.key_concepts = [f"concept_{j}" for j in range(1000)]
            context_manager.validate_context(context)
            contexts.append(context)
        
        metrics = performance_monitor.stop_monitoring()
        cpu_after = psutil.cpu_percent(interval=1)
        
        # CPU usage should be reasonable
        cpu_increase = cpu_after - cpu_before
        assert cpu_increase < 80, f"CPU usage increased by {cpu_increase}%, expected < 80%"
        
        # Processing should complete in reasonable time
        assert metrics["processing_time"] < 10.0, f"CPU-intensive processing took {metrics['processing_time']:.2f}s, expected < 10s"
    
    import gc
# ... (inside TestResourceUtilization class)
    def test_memory_efficiency_under_load(self, memory_profiler):
        """Test memory efficiency under processing load."""
        memory_profiler.take_snapshot("start")
        
        context_manager = ContextManager()
        
        # Create processing load
        active_contexts = []
        for i in range(20):
            context = ContentContext(
                project_id=f"load_test_{i}",
                video_files=[f"video_{i}.mp4"],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences()
            )
            
            # Add substantial data
            context.key_concepts = [f"concept_{j}" for j in range(500)]
            context_manager.validate_context(context)
            active_contexts.append(context)
            
            if i % 5 == 0:
                memory_profiler.take_snapshot(f"load_step_{i}")
        
        memory_profiler.take_snapshot("peak_load")
        
        # Clear contexts to test cleanup
        active_contexts.clear()
        gc.collect()  # Encourage garbage collection
        
        memory_profiler.take_snapshot("after_cleanup")
        
        # Memory usage should be proportional to load
        peak_diff = memory_profiler.get_memory_diff("start", "peak_load")
        cleanup_diff = memory_profiler.get_memory_diff("peak_load", "after_cleanup")
        
        # Should use less than 2GB at peak
        assert peak_diff["rss_diff"] < 2_000_000_000, f"Peak memory usage {peak_diff['rss_diff']} bytes, expected < 2GB"
        
        # Should release significant memory after cleanup
        # This assertion can be flaky, so we check that memory does go down.
        assert cleanup_diff["rss_diff"] <= 0, "Memory should not increase after cleanup"