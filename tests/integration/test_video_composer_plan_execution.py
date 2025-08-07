"""
Integration tests for VideoComposer with AI Director Plan Execution Engine.

Tests the complete integration between AI Director plans, plan execution engine,
and video composition using movis.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ai_video_editor.modules.video_processing.composer import VideoComposer
from ai_video_editor.modules.video_processing.plan_execution import ExecutionTimeline, TrackOperation
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.core.exceptions import ProcessingError, InputValidationError


class TestVideoComposerPlanExecution:
    """Test VideoComposer integration with Plan Execution Engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir()
        
        # Mock movis availability
        self.movis_patch = patch('ai_video_editor.modules.video_processing.composer.MOVIS_AVAILABLE', True)
        self.movis_patch.start()
        
        # Mock movis module
        self.mock_movis = MagicMock()
        self.movis_module_patch = patch('ai_video_editor.modules.video_processing.composer.mv', self.mock_movis)
        self.movis_module_patch.start()
        
        # Create VideoComposer instance
        self.composer = VideoComposer(output_dir=str(self.output_dir), temp_dir=self.temp_dir)
        
        # Mock the plan execution engine
        self.mock_execution_engine = Mock()
        self.composer.plan_execution_engine = self.mock_execution_engine
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.movis_patch.stop()
        self.movis_module_patch.stop()
        shutil.rmtree(self.temp_dir)
    
    def _create_mock_context(self, with_ai_plan=True):
        """Create mock ContentContext with AI Director plan."""
        context = Mock(spec=ContentContext)
        context.project_id = "test_integration"
        context.video_files = [str(Path(self.temp_dir) / "test_video.mp4")]
        context.video_metadata = {"duration": 180.0}
        context.user_preferences = UserPreferences(quality_mode="high")
        context.audio_analysis = Mock()
        context.audio_analysis.segments = []
        
        if with_ai_plan:
            context.processed_video = {
                'editing_decisions': [
                    {
                        'timestamp': 30.0,
                        'decision_type': 'cut',
                        'parameters': {'duration': 0.5, 'cut_type': 'hard'},
                        'rationale': 'Natural pause',
                        'confidence': 0.9,
                        'priority': 8
                    },
                    {
                        'timestamp': 90.0,
                        'decision_type': 'transition',
                        'parameters': {'duration': 1.5, 'type': 'crossfade'},
                        'rationale': 'Topic change',
                        'confidence': 0.8,
                        'priority': 7
                    }
                ],
                'broll_plans': [
                    {
                        'timestamp': 60.0,
                        'duration': 8.0,
                        'content_type': 'chart',
                        'description': 'Investment growth visualization',
                        'visual_elements': ['line_chart', 'data_points'],
                        'animation_style': 'fade_in',
                        'priority': 8
                    }
                ],
                'metadata_strategy': {
                    'primary_title': 'Test Video',
                    'description': 'Test description',
                    'tags': ['test', 'video']
                }
            }
        else:
            context.processed_video = None
        
        # Create mock video file
        video_file = Path(context.video_files[0])
        video_file.touch()
        
        return context
    
    def _create_mock_execution_timeline(self):
        """Create mock ExecutionTimeline for testing."""
        operations = [
            TrackOperation(
                operation_id="cut_video_1",
                track_id="main_video",
                track_type="video",
                start_time=30.0,
                end_time=30.5,
                operation_type="cut",
                parameters={'cut_type': 'hard', 'fade_duration': 0.1},
                priority=8,
                source_decision="cut_30.0"
            ),
            TrackOperation(
                operation_id="cut_audio_1",
                track_id="main_audio",
                track_type="audio",
                start_time=30.0,
                end_time=30.5,
                operation_type="cut",
                parameters={'fade_duration': 0.05},
                priority=8,
                source_decision="cut_30.0"
            ),
            TrackOperation(
                operation_id="broll_content_1",
                track_id="broll_track_60.0",
                track_type="broll",
                start_time=60.0,
                end_time=68.0,
                operation_type="insert",
                parameters={
                    'content_type': 'chart',
                    'description': 'Investment growth visualization',
                    'visual_elements': ['line_chart', 'data_points'],
                    'animation_style': 'fade_in',
                    'opacity': 0.9,
                    'z_index': 10
                },
                priority=8,
                source_decision="broll_plan_60.0"
            ),
            TrackOperation(
                operation_id="transition_1",
                track_id="main_video",
                track_type="effect",
                start_time=90.0,
                end_time=91.5,
                operation_type="transition",
                parameters={'transition_type': 'crossfade', 'duration': 1.5},
                priority=8,
                source_decision="transition_90.0"
            )
        ]
        
        return ExecutionTimeline(
            total_duration=180.0,
            operations=operations,
            sync_points=[],
            track_mapping={
                "main_video": "video",
                "main_audio": "audio",
                "broll_track_60.0": "broll"
            },
            conflicts_resolved=0,
            optimization_applied=True
        )
    
    def test_compose_video_with_ai_plan_success(self):
        """Test successful video composition with AI Director plan."""
        context = self._create_mock_context()
        execution_timeline = self._create_mock_execution_timeline()
        
        # Mock plan execution engine
        self.mock_execution_engine.execute_plan.return_value = execution_timeline
        
        # Mock movis composition and rendering
        mock_composition = MagicMock()
        self.mock_movis.Composition.return_value = mock_composition
        self.mock_movis.layer.VideoFile.return_value = MagicMock()
        self.mock_movis.layer.AudioFile.return_value = MagicMock()
        
        # Execute composition
        result = self.composer.compose_video_with_ai_plan(context)
        
        # Verify plan execution was called
        self.mock_execution_engine.execute_plan.assert_called_once_with(context)
        
        # Verify result structure
        assert 'output_path' in result
        assert 'composition_plan' in result
        assert 'execution_timeline' in result
        assert 'performance_metrics' in result
        assert 'ai_director_integration' in result
        
        # Verify performance metrics
        metrics = result['performance_metrics']
        assert 'plan_execution_time' in metrics
        assert 'composition_time' in metrics
        assert 'render_time' in metrics
        assert 'total_time' in metrics
        assert 'operations_executed' in metrics
        assert metrics['operations_executed'] == 4
        
        # Verify AI Director integration info
        ai_integration = result['ai_director_integration']
        assert ai_integration['plan_execution_successful'] is True
        assert ai_integration['operations_count'] == 4
        assert ai_integration['timeline_optimized'] is True
        
        # Verify movis composition was created and rendered
        self.mock_movis.Composition.assert_called()
        mock_composition.write_video.assert_called_once()
    
    def test_compose_video_without_ai_plan(self):
        """Test composition failure when AI Director plan is missing."""
        context = self._create_mock_context(with_ai_plan=False)
        
        with pytest.raises(InputValidationError, match="Invalid AI Director plan"):
            self.composer.compose_video_with_ai_plan(context)
    
    def test_create_composition_plan_from_timeline(self):
        """Test creation of composition plan from execution timeline."""
        context = self._create_mock_context()
        execution_timeline = self._create_mock_execution_timeline()
        
        composition_plan = self.composer.create_composition_plan_from_timeline(execution_timeline, context)
        
        # Verify composition plan structure
        assert composition_plan.output_settings.duration == 180.0
        assert composition_plan.output_settings.quality == "high"
        
        # Verify layers were created from operations
        assert len(composition_plan.layers) >= 2  # At least video and B-roll layers
        
        # Check for video layer
        video_layers = [layer for layer in composition_plan.layers if layer.layer_type == "video"]
        assert len(video_layers) >= 1
        
        # Check for B-roll layer
        broll_layers = [layer for layer in composition_plan.layers if layer.layer_type == "broll"]
        assert len(broll_layers) >= 1
        
        broll_layer = broll_layers[0]
        assert broll_layer.start_time == 60.0
        assert broll_layer.end_time == 68.0
        assert broll_layer.properties['content_type'] == 'chart'
        assert broll_layer.opacity == 0.9
        
        # Verify transitions were extracted
        assert len(composition_plan.transitions) >= 1
        transition = composition_plan.transitions[0]
        assert transition['timestamp'] == 90.0
        assert transition['type'] == 'crossfade'
        assert transition['duration'] == 1.5
        
        # Verify audio adjustments were created
        assert len(composition_plan.audio_adjustments) >= 1
        audio_adjustment = composition_plan.audio_adjustments[0]
        assert audio_adjustment['timestamp'] == 30.0
        assert audio_adjustment['type'] == 'cut'
    
    def test_create_layer_from_video_operation(self):
        """Test creation of video layer from track operation."""
        context = self._create_mock_context()
        from ai_video_editor.modules.video_processing.composer import CompositionSettings
        settings = CompositionSettings()
        
        # Test cut operation
        cut_operation = TrackOperation(
            operation_id="test_cut",
            track_id="main_video",
            track_type="video",
            start_time=45.0,
            end_time=45.5,
            operation_type="cut",
            parameters={'cut_type': 'hard', 'fade_duration': 0.1},
            priority=8
        )
        
        layer = self.composer._create_layer_from_operation(cut_operation, context, settings)
        
        assert layer is not None
        assert layer.layer_type == "video"
        assert layer.start_time == 45.0
        assert layer.end_time == 45.5
        assert layer.source_path == context.video_files[0]
        assert layer.properties['operation_type'] == 'cut'
        assert layer.properties['fade_duration'] == 0.1
        assert layer.opacity == 1.0
        
        # Test trim operation
        trim_operation = TrackOperation(
            operation_id="test_trim",
            track_id="main_video",
            track_type="video",
            start_time=75.0,
            end_time=78.0,
            operation_type="trim",
            parameters={'trim_type': 'remove', 'smooth_transition': True},
            priority=7
        )
        
        layer = self.composer._create_layer_from_operation(trim_operation, context, settings)
        
        assert layer is not None
        assert layer.layer_type == "video"
        assert layer.start_time == 75.0
        assert layer.end_time == 78.0
        assert layer.properties['operation_type'] == 'trim'
        assert layer.properties['smooth_transition'] is True
    
    def test_create_broll_layer_from_operation(self):
        """Test creation of B-roll layer from track operation."""
        context = self._create_mock_context()
        from ai_video_editor.modules.video_processing.composer import CompositionSettings
        settings = CompositionSettings()
        
        broll_operation = TrackOperation(
            operation_id="test_broll",
            track_id="broll_track_120",
            track_type="broll",
            start_time=120.0,
            end_time=126.0,
            operation_type="insert",
            parameters={
                'content_type': 'animation',
                'description': 'Process visualization',
                'visual_elements': ['flowchart', 'arrows'],
                'animation_style': 'slide_up',
                'opacity': 0.85,
                'z_index': 15
            },
            priority=7
        )
        
        layer = self.composer._create_broll_layer_from_operation(broll_operation, context, settings)
        
        assert layer is not None
        assert layer.layer_type == "broll"
        assert layer.start_time == 120.0
        assert layer.end_time == 126.0
        assert layer.properties['content_type'] == 'animation'
        assert layer.properties['description'] == 'Process visualization'
        assert layer.properties['visual_elements'] == ['flowchart', 'arrows']
        assert layer.properties['animation_style'] == 'slide_up'
        assert layer.properties['z_index'] == 15
        assert layer.opacity == 0.85
    
    def test_movis_composition_creation_with_timeline_layers(self):
        """Test movis composition creation with layers from timeline."""
        context = self._create_mock_context()
        execution_timeline = self._create_mock_execution_timeline()
        
        # Create composition plan from timeline
        composition_plan = self.composer.create_composition_plan_from_timeline(execution_timeline, context)
        
        # Mock movis components
        mock_composition = MagicMock()
        mock_video_layer = MagicMock()
        mock_broll_layer = MagicMock()
        
        self.mock_movis.Composition.return_value = mock_composition
        self.mock_movis.layer.VideoFile.return_value = mock_video_layer
        
        # Create movis composition
        composition = self.composer.create_movis_composition(composition_plan)
        
        # Verify composition was created with correct parameters
        self.mock_movis.Composition.assert_called_with(
            size=(1920, 1080),
            duration=180.0,
            fps=30.0
        )
        
        # Verify layers were added to composition
        assert mock_composition.add_layer.call_count >= len(composition_plan.layers)
        
        # Verify video layers were created
        video_layers = [layer for layer in composition_plan.layers if layer.layer_type == "video"]
        if video_layers:
            self.mock_movis.layer.VideoFile.assert_called()
    
    def test_performance_tracking_integration(self):
        """Test performance tracking throughout the integration."""
        context = self._create_mock_context()
        execution_timeline = self._create_mock_execution_timeline()
        
        # Mock plan execution with timing
        self.mock_execution_engine.execute_plan.return_value = execution_timeline
        
        # Mock movis operations
        mock_composition = MagicMock()
        self.mock_movis.Composition.return_value = mock_composition
        
        # Execute composition
        result = self.composer.compose_video_with_ai_plan(context)
        
        # Verify performance metrics are comprehensive
        metrics = result['performance_metrics']
        
        # Check all timing metrics are present and reasonable
        assert metrics['plan_execution_time'] >= 0
        assert metrics['composition_time'] >= 0
        assert metrics['render_time'] >= 0
        assert metrics['total_time'] >= 0
        
        # Check operation counts
        assert metrics['operations_executed'] == 4
        assert metrics['sync_points_processed'] == 0  # No sync points in mock timeline
        assert metrics['conflicts_resolved'] == 0
        
        # Check layer and effect counts
        assert metrics['layer_count'] >= 2
        assert metrics['transition_count'] >= 1
    
    def test_error_handling_in_plan_execution(self):
        """Test error handling when plan execution fails."""
        context = self._create_mock_context()
        
        # Mock plan execution failure
        self.mock_execution_engine.execute_plan.side_effect = Exception("Plan execution failed")
        
        with pytest.raises(ProcessingError, match="AI-directed video composition failed"):
            self.composer.compose_video_with_ai_plan(context)
    
    def test_legacy_compose_video_method(self):
        """Test that legacy compose_video method uses new AI Director plan execution."""
        context = self._create_mock_context()
        execution_timeline = self._create_mock_execution_timeline()
        
        # Mock plan execution engine
        self.mock_execution_engine.execute_plan.return_value = execution_timeline
        
        # Mock movis composition
        mock_composition = MagicMock()
        self.mock_movis.Composition.return_value = mock_composition
        
        # Call legacy method
        result = self.composer.compose_video(context)
        
        # Should use the new AI Director plan-based composition
        self.mock_execution_engine.execute_plan.assert_called_once_with(context)
        
        # Should have AI Director integration info
        assert 'ai_director_integration' in result
        assert result['ai_director_integration']['plan_execution_successful'] is True
    
    def test_context_update_with_composition_result(self):
        """Test that ContentContext is updated with composition results."""
        context = self._create_mock_context()
        execution_timeline = self._create_mock_execution_timeline()
        
        # Mock plan execution engine
        self.mock_execution_engine.execute_plan.return_value = execution_timeline
        
        # Mock movis composition
        mock_composition = MagicMock()
        self.mock_movis.Composition.return_value = mock_composition
        
        # Execute composition
        result = self.composer.compose_video_with_ai_plan(context)
        
        # Verify context was updated
        assert context.processed_video is not None
        assert 'composition_result' in context.processed_video
        
        composition_result = context.processed_video['composition_result']
        assert composition_result['output_path'] == result['output_path']
        assert 'execution_timeline' in composition_result
        assert 'ai_director_integration' in composition_result


class TestVideoComposerPlanExecutionEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir()
        
        # Mock movis availability
        self.movis_patch = patch('ai_video_editor.modules.video_processing.composer.MOVIS_AVAILABLE', True)
        self.movis_patch.start()
        
        # Mock movis module
        self.mock_movis = MagicMock()
        self.movis_module_patch = patch('ai_video_editor.modules.video_processing.composer.mv', self.mock_movis)
        self.movis_module_patch.start()
        
        self.composer = VideoComposer(output_dir=str(self.output_dir), temp_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.movis_patch.stop()
        self.movis_module_patch.stop()
        shutil.rmtree(self.temp_dir)
    
    def test_empty_execution_timeline(self):
        """Test handling of empty execution timeline."""
        context = Mock(spec=ContentContext)
        context.user_preferences = UserPreferences(quality_mode="medium")
        
        empty_timeline = ExecutionTimeline(
            total_duration=60.0,
            operations=[],
            sync_points=[],
            track_mapping={},
            conflicts_resolved=0,
            optimization_applied=False
        )
        
        composition_plan = self.composer.create_composition_plan_from_timeline(empty_timeline, context)
        
        # Should create valid composition plan even with no operations
        assert composition_plan.output_settings.duration == 60.0
        assert len(composition_plan.layers) == 0
        assert len(composition_plan.transitions) == 0
        assert len(composition_plan.effects) == 0
        assert len(composition_plan.audio_adjustments) == 0
    
    def test_timeline_with_unknown_operation_types(self):
        """Test handling of unknown operation types in timeline."""
        context = Mock(spec=ContentContext)
        context.video_files = ["test.mp4"]
        context.user_preferences = UserPreferences(quality_mode="medium")
        
        # Timeline with unknown operation type
        operations = [
            TrackOperation(
                operation_id="unknown_op",
                track_id="test_track",
                track_type="unknown_type",
                start_time=30.0,
                end_time=35.0,
                operation_type="unknown_operation",
                parameters={},
                priority=5
            )
        ]
        
        timeline = ExecutionTimeline(
            total_duration=120.0,
            operations=operations,
            sync_points=[],
            track_mapping={"test_track": "unknown_type"},
            conflicts_resolved=0,
            optimization_applied=True
        )
        
        # Should handle unknown operations gracefully
        composition_plan = self.composer.create_composition_plan_from_timeline(timeline, context)
        
        # Unknown operations should be skipped
        assert len(composition_plan.layers) == 0
        assert len(composition_plan.transitions) == 0
        assert len(composition_plan.effects) == 0
    
    def test_missing_video_files_in_context(self):
        """Test handling when video files are missing from context."""
        context = Mock(spec=ContentContext)
        context.video_files = []  # No video files
        context.user_preferences = UserPreferences(quality_mode="medium")
        
        video_operation = TrackOperation(
            operation_id="video_cut",
            track_id="main_video",
            track_type="video",
            start_time=30.0,
            end_time=30.5,
            operation_type="cut",
            parameters={},
            priority=8
        )
        
        settings = self.composer.CompositionSettings()
        
        # Should return None when no video files available
        layer = self.composer._create_layer_from_operation(video_operation, context, settings)
        assert layer is None
    
    def test_large_timeline_performance(self):
        """Test performance with large number of operations."""
        context = Mock(spec=ContentContext)
        context.video_files = ["test.mp4"]
        context.user_preferences = UserPreferences(quality_mode="medium")
        
        # Create timeline with many operations
        operations = []
        for i in range(100):  # 100 operations
            operations.append(TrackOperation(
                operation_id=f"op_{i}",
                track_id=f"track_{i % 10}",  # 10 different tracks
                track_type="video" if i % 2 == 0 else "audio",
                start_time=float(i * 2),
                end_time=float(i * 2 + 1),
                operation_type="cut",
                parameters={},
                priority=5
            ))
        
        timeline = ExecutionTimeline(
            total_duration=300.0,
            operations=operations,
            sync_points=[],
            track_mapping={f"track_{i}": "video" if i % 2 == 0 else "audio" for i in range(10)},
            conflicts_resolved=0,
            optimization_applied=True
        )
        
        # Should handle large timeline efficiently
        composition_plan = self.composer.create_composition_plan_from_timeline(timeline, context)
        
        # Verify all operations were processed
        total_elements = (len(composition_plan.layers) + 
                         len(composition_plan.transitions) + 
                         len(composition_plan.effects) + 
                         len(composition_plan.audio_adjustments))
        
        # Should have processed a significant number of operations
        assert total_elements > 50  # At least half should be processed


if __name__ == "__main__":
    pytest.main([__file__])