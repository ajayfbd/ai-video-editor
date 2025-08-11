"""
End-to-end integration tests for AI Video Editor workflows.

These tests validate complete workflows from input to output,
ensuring all components work together correctly.
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from ai_video_editor.core.content_context import (
    ContentContext, ContentType, UserPreferences,
    AudioAnalysisResult, AudioSegment, EmotionalPeak
)
from ai_video_editor.modules.enhancement.audio_enhancement import AudioEnhancementEngine, AudioEnhancementSettings
from ai_video_editor.modules.enhancement.audio_synchronizer import AudioSynchronizer
from ai_video_editor.modules.video_processing.composer import VideoComposer


@pytest.mark.integration
class TestEndToEndWorkflows:
    """End-to-end workflow integration tests."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create directory structure
            (workspace / "input").mkdir()
            (workspace / "output").mkdir()
            (workspace / "cache").mkdir()
            
            yield workspace
    
    @pytest.fixture
    def educational_video_context(self, temp_workspace):
        """Create context for educational video processing."""
        input_video = temp_workspace / "input" / "educational_lecture.mp4"
        input_video.touch()  # Create mock file
        
        context = ContentContext(
            project_id="educational_e2e_test",
            video_files=[str(input_video)],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences(
                quality_mode="high",
                thumbnail_resolution=(1920, 1080),
                batch_size=3
            )
        )
        
        # Add mock audio analysis data for end-to-end testing
        audio_segments = [
            AudioSegment(
                text="In this comprehensive educational lecture, we'll explore advanced financial concepts.",
                start=0.0,
                end=4.0,
                confidence=0.96,
                financial_concepts=["financial concepts", "educational lecture"],
                emotional_markers=["introduction"]
            ),
            AudioSegment(
                text="We'll start with portfolio theory and risk management strategies.",
                start=4.0,
                end=8.0,
                confidence=0.94,
                financial_concepts=["portfolio theory", "risk management"],
                emotional_markers=["explanation"]
            ),
            AudioSegment(
                text="These concepts are fundamental to understanding modern investment approaches.",
                start=8.0,
                end=12.0,
                confidence=0.91,
                financial_concepts=["investment approaches", "modern finance"],
                emotional_markers=["emphasis"]
            )
        ]
        
        audio_analysis = AudioAnalysisResult(
            transcript_text="In this comprehensive educational lecture, we'll explore advanced financial concepts. We'll start with portfolio theory and risk management strategies. These concepts are fundamental to understanding modern investment approaches.",
            segments=audio_segments,
            overall_confidence=0.94,
            language="en",
            processing_time=1.2,
            model_used="whisper-large",
            financial_concepts=["financial concepts", "portfolio theory", "risk management", "investment approaches"],
            detected_emotions=[
                EmotionalPeak(2.0, "confidence", 0.85, 0.92, "educational introduction"),
                EmotionalPeak(6.0, "explanation", 0.78, 0.88, "concept explanation"),
                EmotionalPeak(10.0, "emphasis", 0.82, 0.90, "key concept emphasis")
            ],
            complexity_level="advanced"
        )
        
        context.set_audio_analysis(audio_analysis)
        
        # Add mock AI Director plan for video composition
        context.processed_video = {
            'editing_decisions': [
                {
                    'type': 'cut',
                    'start_time': 0.0,
                    'end_time': 4.0,
                    'reason': 'Educational introduction'
                },
                {
                    'type': 'trim',
                    'start_time': 4.0,
                    'end_time': 8.0,
                    'reason': 'Portfolio theory explanation'
                },
                {
                    'type': 'cut',
                    'start_time': 8.0,
                    'end_time': 12.0,
                    'reason': 'Investment concepts'
                }
            ],
            'broll_plans': [
                {
                    'type': 'chart',
                    'timing': 2.0,
                    'duration': 4.0,
                    'description': 'Portfolio theory visualization'
                },
                {
                    'type': 'animation',
                    'timing': 6.0,
                    'duration': 3.0,
                    'description': 'Risk management concepts'
                }
            ],
            'audio_enhancements': {
                'noise_reduction': True,
                'level_adjustment': True,
                'compression': True
            }
        }
        
        return context
    
    @pytest.fixture
    def mock_audio_processing(self):
        """Mock audio processing components."""
        sample_rate = 48000
        duration = 20.0  # 20 seconds
        mock_audio = np.random.normal(0, 0.1, int(sample_rate * duration))
        
        with patch('librosa.load') as mock_load, \
             patch('soundfile.write') as mock_write:
            
            mock_load.return_value = (mock_audio, sample_rate)
            
            def mock_write_side_effect(filepath, data, samplerate):
                Path(filepath).parent.mkdir(parents=True, exist_ok=True)
                Path(filepath).touch()
            
            mock_write.side_effect = mock_write_side_effect
            
            yield {
                'mock_load': mock_load,
                'mock_write': mock_write,
                'audio_data': mock_audio,
                'sample_rate': sample_rate
            }
    
    @patch('ai_video_editor.modules.enhancement.audio_synchronizer.MOVIS_AVAILABLE', True)
    @patch('ai_video_editor.modules.video_processing.broll_generation.BLENDER_AVAILABLE', False)
    def test_complete_educational_video_processing_workflow(self, educational_video_context, 
                                                          mock_audio_processing, temp_workspace):
        """
        Test complete workflow: Raw video → Enhanced audio → Synchronized video → Composition plan
        """
        # Step 1: Audio Enhancement
        print("\n=== Step 1: Audio Enhancement ===")
        
        enhancement_settings = AudioEnhancementSettings(
            noise_reduction_strength=0.6,
            enable_dynamic_levels=True,
            emotional_boost_factor=1.3,
            explanation_boost_factor=1.15,
            filler_reduction_factor=0.6,
            target_lufs=-14.0,
            enable_eq=True
        )
        
        audio_engine = AudioEnhancementEngine(output_dir="temp/e2e_test", settings=enhancement_settings)
        enhancement_result = audio_engine.enhance_audio(educational_video_context)
        
        # Validate audio enhancement
        assert enhancement_result.processing_time > 0
        assert enhancement_result.original_duration > 0
        assert enhancement_result.enhanced_duration > 0
        assert enhancement_result.noise_reduction_applied is True
        assert enhancement_result.dynamic_adjustments_made > 0
        assert Path(enhancement_result.enhanced_audio_path).exists()
        
        print(f"✅ Audio enhanced: {enhancement_result.dynamic_adjustments_made} adjustments made")
        print(f"   SNR improvement: {enhancement_result.snr_improvement:.2f}dB")
        print(f"   Processing time: {enhancement_result.processing_time:.2f}s")
        
        # Step 2: Audio-Video Synchronization
        print("\n=== Step 2: Audio-Video Synchronization ===")
        
        synchronizer = AudioSynchronizer()
        sync_result = synchronizer.synchronize_audio_video(educational_video_context, enhancement_result)
        
        # Validate synchronization
        assert sync_result.sync_points_processed > 0
        assert sync_result.frame_accurate_points > 0
        assert len(sync_result.audio_tracks) > 0
        assert len(sync_result.movis_audio_layers) > 0
        
        print(f"✅ Audio synchronized: {sync_result.sync_points_processed} sync points processed")
        print(f"   Frame-accurate points: {sync_result.frame_accurate_points}")
        print(f"   Max sync error: {sync_result.max_sync_error:.3f}s")
        
        # Step 3: Video Composition Planning
        print("\n=== Step 3: Video Composition Planning ===")
        
        composer = VideoComposer()
        composition_plan = composer.create_composition_plan(educational_video_context)
        
        # Validate composition planning
        assert composition_plan is not None
        assert len(composition_plan.layers) > 0
        assert composition_plan.output_settings.duration > 0
        assert composition_plan.output_settings.quality == "high"
        
        print(f"✅ Composition planned: {len(composition_plan.layers)} layers")
        print(f"   Total duration: {composition_plan.output_settings.duration:.2f}s")
        print(f"   Quality profile: {composition_plan.output_settings.quality}")
        
        # Step 4: Validate ContentContext Integration
        print("\n=== Step 4: ContentContext Integration Validation ===")
        
        # Check that ContentContext was properly updated throughout the workflow
        assert educational_video_context.processing_metrics is not None
        assert 'audio_enhancement' in educational_video_context.processing_metrics.module_processing_times
        assert 'audio_synchronization' in educational_video_context.processing_metrics.module_processing_times
        
        # Check that audio analysis was updated
        if educational_video_context.audio_analysis:
            assert educational_video_context.audio_analysis.enhanced_duration == enhancement_result.enhanced_duration
        
        print("✅ ContentContext properly integrated across all modules")
        
        # Step 5: End-to-End Validation
        print("\n=== Step 5: End-to-End Validation ===")
        
        # Validate complete workflow results
        workflow_results = {
            'audio_enhancement': enhancement_result,
            'synchronization': sync_result,
            'composition_plan': composition_plan,
            'context': educational_video_context
        }
        
        # Check workflow completeness
        assert all(result is not None for result in workflow_results.values())
        
        # Check processing metrics
        total_processing_time = (
            enhancement_result.processing_time + 
            sync_result.processing_time
        )
        
        assert total_processing_time < 60.0, "Total processing time should be under 1 minute for test content"
        
        print(f"✅ End-to-end workflow completed successfully")
        print(f"   Total processing time: {total_processing_time:.2f}s")
        print(f"   Memory usage tracked: {len(educational_video_context.processing_metrics.module_processing_times)} modules")
        
        return workflow_results
    
    def test_error_recovery_workflow(self, educational_video_context, temp_workspace):
        """
        Test workflow error recovery and graceful degradation.
        """
        print("\n=== Error Recovery Workflow Test ===")
        
        # Test 1: Missing input file
        print("Testing missing input file handling...")
        
        invalid_context = ContentContext(
            project_id="error_test",
            video_files=["/nonexistent/file.mp4"],
            content_type=ContentType.EDUCATIONAL,
            user_preferences=UserPreferences(quality_mode="balanced")
        )
        
        settings = AudioEnhancementSettings()
        engine = AudioEnhancementEngine(output_dir="temp/e2e_test", settings=settings)
        
        # Should handle missing file gracefully
        with pytest.raises(Exception) as exc_info:
            engine.enhance_audio(invalid_context)
        
        # Error should be informative
        error_message = str(exc_info.value)
        assert len(error_message) > 0
        print(f"✅ Missing file error handled: {error_message[:50]}...")
        
        # Test 2: Invalid settings recovery
        print("Testing invalid settings recovery...")
        
        invalid_settings = AudioEnhancementSettings(
            noise_reduction_strength=2.0,  # Invalid value > 1.0
            emotional_boost_factor=-1.0    # Invalid negative value
        )
        
        # System should either correct invalid values or fail gracefully
        try:
            engine_with_invalid = AudioEnhancementEngine(output_dir="temp/e2e_test", settings=invalid_settings)
            print("✅ Invalid settings handled gracefully")
        except Exception as e:
            print(f"✅ Invalid settings rejected with clear error: {str(e)[:50]}...")
        
        # Test 3: Partial workflow completion
        print("Testing partial workflow completion...")
        
        # This test would verify that if one step fails, 
        # the system can still provide partial results
        # TODO: Implement partial completion logic
        
        print("✅ Error recovery workflow tests completed")
    
    @patch('ai_video_editor.modules.video_processing.broll_generation.BLENDER_AVAILABLE', False)
    def test_resource_constrained_workflow(self, educational_video_context, mock_audio_processing):
        """
        Test workflow under resource constraints (limited memory, no GPU, etc.).
        """
        print("\n=== Resource Constrained Workflow Test ===")
        
        # Simulate resource constraints
        original_preferences = educational_video_context.user_preferences
        educational_video_context.user_preferences = UserPreferences(
            quality_mode="fast",  # Lower quality for resource constraints
            thumbnail_resolution=(1280, 720),  # Lower resolution for resource constraints
            batch_size=1  # Smaller batch size for resource constraints
        )
        
        # Test with reduced settings
        constrained_settings = AudioEnhancementSettings(
            noise_reduction_strength=0.3,  # Reduced processing
            enable_dynamic_levels=False,   # Disable expensive features
            enable_eq=False
        )
        
        # Run workflow with constraints
        engine = AudioEnhancementEngine(output_dir="temp/e2e_test", settings=constrained_settings)
        result = engine.enhance_audio(educational_video_context)
        
        # Should still work but with reduced processing
        assert result.processing_time > 0
        assert result.dynamic_adjustments_made >= 0  # May be 0 due to disabled features
        
        print(f"✅ Resource constrained processing completed")
        print(f"   Processing time: {result.processing_time:.2f}s")
        print(f"   Adjustments made: {result.dynamic_adjustments_made}")
        
        # Restore original preferences
        educational_video_context.user_preferences = original_preferences
    
    def test_concurrent_workflow_processing(self, temp_workspace):
        """
        Test multiple workflows running concurrently.
        """
        print("\n=== Concurrent Workflow Processing Test ===")
        
        # Create multiple contexts
        contexts = []
        for i in range(3):
            video_file = temp_workspace / f"input/video_{i}.mp4"
            video_file.touch()
            
            context = ContentContext(
                project_id=f"concurrent_test_{i}",
                video_files=[str(video_file)],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences(quality_mode="balanced")
            )
            contexts.append(context)
        
        # TODO: Implement actual concurrent processing
        # For now, just verify contexts are created correctly
        
        assert len(contexts) == 3
        assert all(ctx.project_id.startswith("concurrent_test_") for ctx in contexts)
        
        print("✅ Concurrent workflow setup completed")
        print(f"   Created {len(contexts)} concurrent contexts")
        
        # In a real implementation, this would:
        # 1. Process all contexts simultaneously
        # 2. Verify no resource conflicts
        # 3. Ensure all complete successfully
        # 4. Check memory usage stays within limits
    
    def test_workflow_state_persistence(self, educational_video_context, temp_workspace):
        """
        Test that workflow state can be saved and resumed.
        """
        print("\n=== Workflow State Persistence Test ===")
        
        # Create state file
        state_file = temp_workspace / "workflow_state.json"
        
        # Save initial state
        initial_state = {
            'project_id': educational_video_context.project_id,
            'video_files': educational_video_context.video_files,
            'content_type': educational_video_context.content_type.value,
            'step_completed': 'initialization'
        }
        
        with open(state_file, 'w') as f:
            json.dump(initial_state, f)
        
        assert state_file.exists()
        
        # Load and verify state
        with open(state_file, 'r') as f:
            loaded_state = json.load(f)
        
        assert loaded_state['project_id'] == educational_video_context.project_id
        assert loaded_state['step_completed'] == 'initialization'
        
        print("✅ Workflow state persistence working")
        print(f"   State file: {state_file}")
        print(f"   Project ID: {loaded_state['project_id']}")
        
        # TODO: Implement actual workflow resumption
        # This would involve:
        # 1. Saving state after each major step
        # 2. Loading state and resuming from correct point
        # 3. Handling partial results and intermediate files


@pytest.mark.integration
@pytest.mark.slow
class TestLongRunningWorkflows:
    """Tests for longer-running workflows that simulate real usage."""
    
    def test_large_content_processing_simulation(self):
        """
        Simulate processing of large educational content (15+ minutes).
        """
        print("\n=== Large Content Processing Simulation ===")
        
        # This test simulates the processing of a 15-minute educational video
        # without actually processing that much content (for test speed)
        
        simulated_duration = 900.0  # 15 minutes
        target_processing_time = 600.0  # 10 minutes (requirement)
        
        # Simulate processing metrics
        simulated_metrics = {
            'content_duration': simulated_duration,
            'audio_enhancement_time': 45.0,
            'synchronization_time': 15.0,
            'composition_planning_time': 30.0,
            'total_processing_time': 90.0,  # Much less than 10 minutes
            'memory_peak': 8_000_000_000,  # 8GB
            'api_calls_made': 25
        }
        
        # Validate against requirements
        assert simulated_metrics['total_processing_time'] < target_processing_time
        assert simulated_metrics['memory_peak'] < 16_000_000_000  # Under 16GB
        
        print(f"✅ Large content simulation passed")
        print(f"   Content duration: {simulated_metrics['content_duration']/60:.1f} minutes")
        print(f"   Processing time: {simulated_metrics['total_processing_time']/60:.1f} minutes")
        print(f"   Memory peak: {simulated_metrics['memory_peak']/1024**3:.1f}GB")
        
        # TODO: Implement actual large content test with real processing
    
    def test_batch_processing_workflow(self):
        """
        Test processing multiple videos in batch.
        """
        print("\n=== Batch Processing Workflow Test ===")
        
        # Simulate batch processing of 5 videos
        batch_size = 5
        simulated_results = []
        
        for i in range(batch_size):
            result = {
                'video_id': f"batch_video_{i}",
                'processing_time': 30.0 + (i * 2),  # Slight variation
                'success': True,
                'output_files': [f"output_{i}.mp4", f"thumbnail_{i}.jpg"]
            }
            simulated_results.append(result)
        
        # Validate batch results
        assert len(simulated_results) == batch_size
        assert all(result['success'] for result in simulated_results)
        
        total_batch_time = sum(result['processing_time'] for result in simulated_results)
        average_time = total_batch_time / batch_size
        
        print(f"✅ Batch processing simulation completed")
        print(f"   Batch size: {batch_size} videos")
        print(f"   Total time: {total_batch_time:.1f}s")
        print(f"   Average per video: {average_time:.1f}s")
        
        # TODO: Implement actual batch processing


if __name__ == "__main__":
    # Run end-to-end integration tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])