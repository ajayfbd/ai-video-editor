"""
Performance benchmarking tests for AI Video Editor.

These tests measure and track performance metrics to detect regressions
and ensure the system meets performance targets.
"""

import pytest
import time
import psutil
import json
import os
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch
import numpy as np

from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.modules.enhancement.audio_enhancement import AudioEnhancementEngine, AudioEnhancementSettings
from ai_video_editor.modules.enhancement.audio_synchronizer import AudioSynchronizer
from ai_video_editor.modules.video_processing.composer import VideoComposer


class PerformanceBenchmark:
    """Performance benchmark tracking and comparison."""
    
    def __init__(self, benchmark_name: str):
        self.benchmark_name = benchmark_name
        self.results: Dict[str, Any] = {}
        self.baseline_file = Path("performance_benchmarks.json")
        
    def start_measurement(self):
        """Start performance measurement."""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss
        
    def end_measurement(self):
        """End performance measurement and record results."""
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        self.results = {
            'processing_time': end_time - self.start_time,
            'memory_used': end_memory - self.start_memory,
            'peak_memory': end_memory,
            'timestamp': time.time()
        }
        
    def compare_with_baseline(self, tolerance: float = 0.1) -> Dict[str, Any]:
        """Compare current results with baseline performance."""
        if not self.baseline_file.exists():
            # First run, create baseline
            self.save_as_baseline()
            return {'status': 'baseline_created', 'results': self.results}
            
        with open(self.baseline_file, 'r') as f:
            baselines = json.load(f)
            
        if self.benchmark_name not in baselines:
            baselines[self.benchmark_name] = self.results
            with open(self.baseline_file, 'w') as f:
                json.dump(baselines, f, indent=2)
            return {'status': 'baseline_created', 'results': self.results}
            
        baseline = baselines[self.benchmark_name]
        comparison = {
            'status': 'compared',
            'current': self.results,
            'baseline': baseline,
            'changes': {}
        }
        
        # Compare processing time
        time_change = (self.results['processing_time'] - baseline['processing_time']) / baseline['processing_time']
        comparison['changes']['processing_time'] = {
            'change_percent': time_change * 100,
            'regression': time_change > tolerance
        }
        
        # Compare memory usage
        memory_change = (self.results['memory_used'] - baseline['memory_used']) / max(baseline['memory_used'], 1)
        comparison['changes']['memory_used'] = {
            'change_percent': memory_change * 100,
            'regression': memory_change > tolerance
        }
        
        return comparison
        
    def save_as_baseline(self):
        """Save current results as new baseline."""
        baselines = {}
        if self.baseline_file.exists():
            with open(self.baseline_file, 'r') as f:
                baselines = json.load(f)
                
        baselines[self.benchmark_name] = self.results
        
        with open(self.baseline_file, 'w') as f:
            json.dump(baselines, f, indent=2)


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.fixture
    def mock_audio_data(self):
        """Create mock audio data for testing."""
        sample_rate = 48000
        duration = 15.0  # 15 seconds
        return np.random.normal(0, 0.1, int(sample_rate * duration)), sample_rate
    
    @pytest.fixture
    def performance_context(self, tmp_path):
        """Create ContentContext for performance testing."""
        from ai_video_editor.core.content_context import AudioSegment, AudioAnalysisResult, EmotionalPeak
        
        context = ContentContext(
            project_id="performance_test",
            video_files=[str(tmp_path / "test_video.mp4")],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences(quality_mode="balanced")
        )
        
        # Add audio analysis data (required for enhancement)
        segments = [
            AudioSegment(
                text="Performance test audio segment",
                start=0.0,
                end=15.0,
                confidence=0.95,
                filler_words=[],
                cleaned_text="Performance test audio segment"
            )
        ]
        
        emotional_peaks = [
            EmotionalPeak(
                timestamp=5.0,
                emotion="neutral",
                intensity=0.5,
                confidence=0.9,
                context="test content"
            )
        ]
        
        context.audio_analysis = AudioAnalysisResult(
            transcript_text="Performance test audio segment",
            segments=segments,
            overall_confidence=0.95,
            language="en",
            processing_time=1.0,
            model_used="test",
            filler_words_removed=0,
            segments_modified=0,
            quality_improvement_score=0.8,
            original_duration=15.0,
            enhanced_duration=15.0,
            financial_concepts=["performance", "test"]
        )
        
        # Add emotional markers separately
        context.emotional_markers = emotional_peaks
        
        # Create mock video file
        video_path = Path(context.video_files[0])
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.touch()
        
        return context
    
    @patch('librosa.load')
    @patch('soundfile.write')
    def test_audio_enhancement_performance(self, mock_write, mock_load, 
                                         performance_context, mock_audio_data):
        """Benchmark audio enhancement performance."""
        benchmark = PerformanceBenchmark("audio_enhancement")
        
        # Setup mocks
        mock_load.return_value = mock_audio_data
        def mock_write_side_effect(filepath, data, samplerate):
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            Path(filepath).touch()
        mock_write.side_effect = mock_write_side_effect
        
        # Create enhancement engine
        settings = AudioEnhancementSettings(
            noise_reduction_strength=0.5,
            enable_dynamic_levels=True,
            emotional_boost_factor=1.2,
            explanation_boost_factor=1.1,
            filler_reduction_factor=0.7
        )
        engine = AudioEnhancementEngine(output_dir="temp/performance_test", settings=settings)
        
        # Benchmark audio enhancement
        benchmark.start_measurement()
        result = engine.enhance_audio(performance_context)
        benchmark.end_measurement()
        
        # Verify functionality
        assert result.processing_time > 0
        assert result.enhanced_duration > 0
        
        # Compare with baseline
        comparison = benchmark.compare_with_baseline(tolerance=0.2)  # 20% tolerance
        
        # Performance assertions
        assert benchmark.results['processing_time'] < 5.0, "Audio enhancement should complete in under 5 seconds"
        assert benchmark.results['memory_used'] < 500_000_000, "Memory usage should be under 500MB"
        
        # Check for regressions
        if comparison['status'] == 'compared':
            time_regression = comparison['changes']['processing_time']['regression']
            memory_regression = comparison['changes']['memory_used']['regression']
            
            if time_regression:
                pytest.fail(f"Performance regression detected: processing time increased by "
                          f"{comparison['changes']['processing_time']['change_percent']:.1f}%")
            
            if memory_regression:
                pytest.fail(f"Memory regression detected: memory usage increased by "
                          f"{comparison['changes']['memory_used']['change_percent']:.1f}%")
    
    @patch('ai_video_editor.modules.enhancement.audio_synchronizer.MOVIS_AVAILABLE', True)
    @patch('librosa.load')
    @patch('soundfile.write')
    def test_audio_synchronization_performance(self, mock_write, mock_load,
                                             performance_context, mock_audio_data):
        """Benchmark audio synchronization performance."""
        benchmark = PerformanceBenchmark("audio_synchronization")
        
        # Setup mocks
        mock_load.return_value = mock_audio_data
        def mock_write_side_effect(filepath, data, samplerate):
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            Path(filepath).touch()
        mock_write.side_effect = mock_write_side_effect
        
        # Create enhancement result
        from ai_video_editor.modules.enhancement.audio_enhancement import AudioEnhancementResult
        enhancement_result = AudioEnhancementResult(
            processing_time=0.5,
            original_duration=15.0,
            enhanced_duration=14.8,
            noise_reduction_applied=True,
            dynamic_adjustments_made=5,
            peak_reductions=2,
            level_boosts=3,
            snr_improvement=2.5,
            dynamic_range_improvement=1.2,
            loudness_consistency_score=0.85,
            enhanced_audio_path=str(performance_context.video_files[0]).replace('.mp4', '_enhanced.wav'),
            sync_points=[],
            level_adjustments=[]
        )
        
        # Create the enhanced audio file
        enhanced_path = Path(enhancement_result.enhanced_audio_path)
        enhanced_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced_path.touch()
        
        # Create synchronizer
        synchronizer = AudioSynchronizer()
        
        # Benchmark synchronization
        benchmark.start_measurement()
        result = synchronizer.synchronize_audio_video(performance_context, enhancement_result)
        benchmark.end_measurement()
        
        # Verify functionality
        assert result.sync_points_processed > 0
        
        # Compare with baseline
        comparison = benchmark.compare_with_baseline(tolerance=0.2)
        
        # Performance assertions
        assert benchmark.results['processing_time'] < 2.0, "Audio sync should complete in under 2 seconds"
        assert benchmark.results['memory_used'] < 200_000_000, "Memory usage should be under 200MB"
        
        # Check for regressions
        if comparison['status'] == 'compared':
            time_regression = comparison['changes']['processing_time']['regression']
            memory_regression = comparison['changes']['memory_used']['regression']
            
            if time_regression:
                pytest.fail(f"Performance regression detected: processing time increased by "
                          f"{comparison['changes']['processing_time']['change_percent']:.1f}%")
            
            if memory_regression:
                pytest.fail(f"Memory regression detected: memory usage increased by "
                          f"{comparison['changes']['memory_used']['change_percent']:.1f}%")
    
    @patch('ai_video_editor.modules.video_processing.broll_generation.BLENDER_AVAILABLE', False)
    def test_video_composition_performance(self, performance_context):
        """Benchmark video composition performance."""
        benchmark = PerformanceBenchmark("video_composition")
        
        # Create video composer
        composer = VideoComposer()
        
        # Benchmark composition plan creation
        benchmark.start_measurement()
        plan = composer.create_composition_plan(performance_context)
        benchmark.end_measurement()
        
        # Verify functionality
        assert plan is not None
        assert len(plan.layers) > 0
        
        # Compare with baseline
        comparison = benchmark.compare_with_baseline(tolerance=0.3)
        
        # Performance assertions
        assert benchmark.results['processing_time'] < 3.0, "Composition planning should complete in under 3 seconds"
        assert benchmark.results['memory_used'] < 300_000_000, "Memory usage should be under 300MB"
        
        # Check for regressions
        if comparison['status'] == 'compared':
            time_regression = comparison['changes']['processing_time']['regression']
            memory_regression = comparison['changes']['memory_used']['regression']
            
            if time_regression:
                pytest.fail(f"Performance regression detected: processing time increased by "
                          f"{comparison['changes']['processing_time']['change_percent']:.1f}%")
            
            if memory_regression:
                pytest.fail(f"Memory regression detected: memory usage increased by "
                          f"{comparison['changes']['memory_used']['change_percent']:.1f}%")
    
    def test_memory_leak_detection(self, performance_context):
        """Test for memory leaks during repeated operations."""
        initial_memory = psutil.Process().memory_info().rss
        memory_samples = []
        
        # Perform repeated operations
        for i in range(10):
            # Create and destroy objects
            context = ContentContext(
                project_id=f"leak_test_{i}",
                video_files=["/tmp/test.mp4"],
                content_type=ContentType.EDUCATIONAL
            )
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Sample memory
            current_memory = psutil.Process().memory_info().rss
            memory_samples.append(current_memory - initial_memory)
        
        # Check for consistent memory growth (potential leak)
        if len(memory_samples) >= 5:
            # Calculate trend
            x = list(range(len(memory_samples)))
            y = memory_samples
            
            # Simple linear regression to detect trend
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            # If slope is positive and significant, we might have a leak
            memory_growth_per_iteration = slope
            total_growth = memory_samples[-1] - memory_samples[0]
            
            assert total_growth < 50_000_000, f"Potential memory leak detected: {total_growth / 1_000_000:.1f}MB growth"
            assert memory_growth_per_iteration < 5_000_000, f"Memory growth per iteration too high: {memory_growth_per_iteration / 1_000_000:.1f}MB"


@pytest.mark.performance
class TestPerformanceTargets:
    """Test specific performance targets from requirements."""
    
    def test_educational_content_processing_target(self):
        """Test that educational content (15+ min) processes in under 10 minutes."""
        # This would be an end-to-end test with actual video processing
        # For now, we'll create a placeholder that documents the requirement
        target_processing_time = 600  # 10 minutes in seconds
        content_duration = 900  # 15 minutes in seconds
        
        # TODO: Implement full end-to-end processing test
        # This test should:
        # 1. Load a 15-minute educational video
        # 2. Run complete AI Video Editor pipeline
        # 3. Measure total processing time
        # 4. Assert processing_time < target_processing_time
        
        pytest.skip("End-to-end performance test requires full pipeline implementation")
    
    def test_memory_usage_target(self):
        """Test that peak memory usage stays under 16GB."""
        target_memory_limit = 16 * 1024 * 1024 * 1024  # 16GB in bytes
        current_memory = psutil.Process().memory_info().rss
        
        # This is a basic check - real test would monitor throughout processing
        assert current_memory < target_memory_limit, f"Memory usage {current_memory / 1024**3:.1f}GB exceeds 16GB limit"
    
    def test_api_cost_tracking(self):
        """Test that API costs are tracked and stay within budget."""
        # TODO: Implement API cost tracking test
        # This test should:
        # 1. Mock API calls with cost tracking
        # 2. Run typical processing workflow
        # 3. Verify total costs are under $2 per project
        
        pytest.skip("API cost tracking test requires cost monitoring implementation")


if __name__ == "__main__":
    # Run performance benchmarks
    pytest.main([__file__, "-v", "--tb=short"])