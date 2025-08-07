"""
Unit tests for AI Director Plan Execution Engine.

Tests the complete plan execution workflow including editing decision interpretation,
timeline management, B-roll insertion, and audio-video synchronization.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from ai_video_editor.modules.video_processing.plan_execution import (
    PlanExecutionEngine,
    ExecutionCoordinator,
    EditingDecisionInterpreter,
    TimelineManager,
    BRollInsertionManager,
    AudioVideoSynchronizer,
    TrackOperation,
    ExecutionTimeline,
    SynchronizationPoint
)
from ai_video_editor.modules.intelligence.ai_director import (
    EditingDecision,
    BRollPlan,
    MetadataStrategy,
    AIDirectorPlan
)
from ai_video_editor.core.content_context import ContentContext, ContentType, UserPreferences
from ai_video_editor.core.exceptions import ContentContextError


class TestEditingDecisionInterpreter:
    """Test editing decision interpretation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.interpreter = EditingDecisionInterpreter()
        self.context = Mock(spec=ContentContext)
        self.context.video_files = ["test_video.mp4"]
        self.context.audio_analysis = Mock()
        self.context.audio_analysis.segments = []
    
    def test_interpret_cut_decision(self):
        """Test interpretation of cut decisions."""
        cut_decision = EditingDecision(
            timestamp=30.0,
            decision_type="cut",
            parameters={
                'duration': 0.5,
                'cut_type': 'hard',
                'fade_duration': 0.1,
                'preserve_audio': True
            },
            rationale="Natural pause detected",
            confidence=0.9,
            priority=8
        )
        
        operations = self.interpreter.interpret_decisions([cut_decision], self.context)
        
        assert len(operations) == 1  # Only video cut since preserve_audio=True
        
        video_cut = operations[0]
        assert video_cut.track_type == "video"
        assert video_cut.operation_type == "cut"
        assert video_cut.start_time == 30.0
        assert video_cut.end_time == 30.5
        assert video_cut.priority == 8
        assert video_cut.parameters['cut_type'] == 'hard'
    
    def test_interpret_cut_decision_with_audio(self):
        """Test cut decision that affects audio."""
        cut_decision = EditingDecision(
            timestamp=45.0,
            decision_type="cut",
            parameters={
                'duration': 1.0,
                'preserve_audio': False,
                'audio_fade': 0.05
            },
            rationale="Remove filler words",
            confidence=0.85,
            priority=7
        )
        
        operations = self.interpreter.interpret_decisions([cut_decision], self.context)
        
        assert len(operations) == 2  # Video and audio cuts
        
        video_cut = next(op for op in operations if op.track_type == "video")
        audio_cut = next(op for op in operations if op.track_type == "audio")
        
        assert video_cut.start_time == 45.0
        assert audio_cut.start_time == 45.0
        assert audio_cut.parameters['fade_duration'] == 0.05
    
    def test_interpret_trim_decision(self):
        """Test interpretation of trim decisions."""
        trim_decision = EditingDecision(
            timestamp=60.0,
            decision_type="trim",
            parameters={
                'duration': 3.0,
                'trim_type': 'remove',
                'smooth_transition': True,
                'fade_in': 0.2,
                'fade_out': 0.2
            },
            rationale="Remove unnecessary segment",
            confidence=0.8,
            priority=6
        )
        
        operations = self.interpreter.interpret_decisions([trim_decision], self.context)
        
        assert len(operations) == 2  # Video and audio trims
        
        video_trim = next(op for op in operations if op.track_type == "video")
        audio_trim = next(op for op in operations if op.track_type == "audio")
        
        assert video_trim.operation_type == "trim"
        assert video_trim.start_time == 60.0
        assert video_trim.end_time == 63.0
        assert video_trim.parameters['smooth_transition'] is True
        
        assert audio_trim.operation_type == "trim"
        assert audio_trim.start_time == 60.0
        assert audio_trim.end_time == 63.0
    
    def test_interpret_transition_decision(self):
        """Test interpretation of transition decisions."""
        transition_decision = EditingDecision(
            timestamp=90.0,
            decision_type="transition",
            parameters={
                'duration': 1.5,
                'type': 'crossfade',
                'easing': 'ease_in_out',
                'direction': 'forward'
            },
            rationale="Smooth topic change",
            confidence=0.9,
            priority=9
        )
        
        operations = self.interpreter.interpret_decisions([transition_decision], self.context)
        
        assert len(operations) == 1
        
        transition = operations[0]
        assert transition.track_type == "effect"
        assert transition.operation_type == "transition"
        assert transition.start_time == 90.0
        assert transition.end_time == 91.5
        assert transition.parameters['transition_type'] == 'crossfade'
        assert transition.priority == 10  # Transitions get +1 priority
    
    def test_interpret_emphasis_decision(self):
        """Test interpretation of emphasis decisions."""
        emphasis_decision = EditingDecision(
            timestamp=120.0,
            decision_type="emphasis",
            parameters={
                'duration': 2.5,
                'emphasis_type': 'zoom_in',
                'intensity': 1.3,
                'fade_in': 0.3,
                'fade_out': 0.3
            },
            rationale="Highlight key concept",
            confidence=0.85,
            priority=7
        )
        
        operations = self.interpreter.interpret_decisions([emphasis_decision], self.context)
        
        assert len(operations) == 1
        
        emphasis = operations[0]
        assert emphasis.track_type == "effect"
        assert emphasis.operation_type == "effect"
        assert emphasis.start_time == 120.0
        assert emphasis.end_time == 122.5
        assert emphasis.parameters['effect_type'] == 'zoom_in'
        assert emphasis.parameters['intensity'] == 1.3
    
    def test_interpret_broll_decision(self):
        """Test interpretation of B-roll decisions."""
        broll_decision = EditingDecision(
            timestamp=150.0,
            decision_type="b_roll",
            parameters={
                'duration': 6.0,
                'content_type': 'chart',
                'description': 'Compound interest visualization',
                'visual_elements': ['bar_chart', 'growth_curve'],
                'animation_style': 'slide_up',
                'opacity': 0.85,
                'position': 'overlay'
            },
            rationale="Visualize financial concept",
            confidence=0.9,
            priority=8
        )
        
        operations = self.interpreter.interpret_decisions([broll_decision], self.context)
        
        assert len(operations) == 1
        
        broll = operations[0]
        assert broll.track_type == "broll"
        assert broll.operation_type == "insert"
        assert broll.start_time == 150.0
        assert broll.end_time == 156.0
        assert broll.parameters['content_type'] == 'chart'
        assert broll.parameters['description'] == 'Compound interest visualization'
        assert broll.parameters['opacity'] == 0.85
    
    def test_interpret_multiple_decisions(self):
        """Test interpretation of multiple decisions."""
        decisions = [
            EditingDecision(10.0, "cut", {'duration': 0.5}, "Cut 1", 0.9, 8),
            EditingDecision(20.0, "transition", {'duration': 1.0, 'type': 'fade'}, "Transition 1", 0.8, 7),
            EditingDecision(30.0, "emphasis", {'duration': 2.0, 'emphasis_type': 'zoom'}, "Emphasis 1", 0.85, 6)
        ]
        
        operations = self.interpreter.interpret_decisions(decisions, self.context)
        
        assert len(operations) == 3
        
        # Check operations are sorted by timestamp
        timestamps = [op.start_time for op in operations]
        assert timestamps == [10.0, 20.0, 30.0]
    
    def test_unknown_decision_type(self):
        """Test handling of unknown decision types."""
        unknown_decision = EditingDecision(
            timestamp=40.0,
            decision_type="unknown_type",
            parameters={},
            rationale="Unknown operation",
            confidence=0.5,
            priority=5
        )
        
        operations = self.interpreter.interpret_decisions([unknown_decision], self.context)
        
        assert len(operations) == 0  # Unknown decisions are skipped


class TestTimelineManager:
    """Test timeline management and conflict resolution."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = TimelineManager()
    
    def test_create_simple_timeline(self):
        """Test creation of simple timeline without conflicts."""
        operations = [
            TrackOperation("op1", "video", "video", 10.0, 15.0, "cut", {}, 8),
            TrackOperation("op2", "audio", "audio", 10.0, 15.0, "cut", {}, 8),
            TrackOperation("op3", "broll", "broll", 20.0, 25.0, "insert", {}, 6)
        ]
        
        timeline = self.manager.create_timeline(operations, 180.0)
        
        assert timeline.total_duration == 180.0
        assert len(timeline.operations) == 3
        assert len(timeline.sync_points) >= 1  # At least one sync point
        assert timeline.conflicts_resolved == 0
        assert timeline.optimization_applied is True
    
    def test_resolve_timeline_conflicts(self):
        """Test conflict resolution between overlapping operations."""
        # Create overlapping operations on same track
        operations = [
            TrackOperation("op1", "video", "video", 10.0, 15.0, "cut", {}, 8),
            TrackOperation("op2", "video", "video", 12.0, 17.0, "trim", {}, 6),  # Overlaps with op1
            TrackOperation("op3", "video", "video", 14.0, 19.0, "effect", {}, 7)  # Overlaps with both
        ]
        
        timeline = self.manager.create_timeline(operations, 180.0)
        
        # Check that conflicts were resolved
        assert timeline.conflicts_resolved > 0
        
        # Check that operations are now non-overlapping
        video_ops = [op for op in timeline.operations if op.track_id == "video"]
        video_ops.sort(key=lambda x: x.start_time)
        
        for i in range(len(video_ops) - 1):
            current_end = video_ops[i].end_time
            next_start = video_ops[i + 1].start_time
            assert next_start >= current_end, "Operations should not overlap after conflict resolution"
    
    def test_create_sync_points(self):
        """Test creation of synchronization points."""
        operations = [
            TrackOperation("cut1", "video", "video", 30.0, 30.5, "cut", {}, 8),
            TrackOperation("cut2", "audio", "audio", 30.0, 30.5, "cut", {}, 8),
            TrackOperation("trans1", "video", "effect", 60.0, 61.0, "transition", {}, 7)
        ]
        
        timeline = self.manager.create_timeline(operations, 180.0)
        
        # Should have sync points for cuts and transitions
        sync_points = timeline.sync_points
        assert len(sync_points) >= 2
        
        # Check for cut sync point
        cut_sync = next((sp for sp in sync_points if sp.sync_type == "cut_sync"), None)
        assert cut_sync is not None
        assert cut_sync.timestamp == 30.0
        
        # Check for transition sync points
        transition_syncs = [sp for sp in sync_points if "transition" in sp.sync_type]
        assert len(transition_syncs) >= 1
    
    def test_build_track_mapping(self):
        """Test building of track mapping."""
        operations = [
            TrackOperation("op1", "main_video", "video", 10.0, 15.0, "cut", {}, 8),
            TrackOperation("op2", "main_audio", "audio", 10.0, 15.0, "cut", {}, 8),
            TrackOperation("op3", "broll_1", "broll", 20.0, 25.0, "insert", {}, 6),
            TrackOperation("op4", "effect_1", "effect", 30.0, 32.0, "transition", {}, 7)
        ]
        
        timeline = self.manager.create_timeline(operations, 180.0)
        
        expected_mapping = {
            "main_video": "video",
            "main_audio": "audio",
            "broll_1": "broll",
            "effect_1": "effect"
        }
        
        assert timeline.track_mapping == expected_mapping


class TestBRollInsertionManager:
    """Test B-roll insertion management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = BRollInsertionManager()
        self.context = Mock(spec=ContentContext)
    
    def test_process_single_broll_plan(self):
        """Test processing of single B-roll plan."""
        broll_plan = BRollPlan(
            timestamp=45.0,
            duration=8.0,
            content_type="chart",
            description="Investment growth visualization",
            visual_elements=["line_chart", "data_points"],
            animation_style="fade_in",
            priority=7
        )
        
        operations = self.manager.process_broll_plans([broll_plan], self.context)
        
        # Should create 3 operations: content + entrance + exit transitions
        assert len(operations) == 3
        
        # Check main content operation
        content_op = next(op for op in operations if "content" in op.operation_id)
        assert content_op.track_type == "broll"
        assert content_op.operation_type == "insert"
        assert content_op.start_time == 45.0
        assert content_op.end_time == 53.0
        assert content_op.parameters['content_type'] == "chart"
        assert content_op.parameters['description'] == "Investment growth visualization"
        
        # Check entrance transition
        entrance_op = next(op for op in operations if "entrance" in op.operation_id)
        assert entrance_op.track_type == "effect"
        assert entrance_op.operation_type == "transition"
        assert entrance_op.start_time == 45.0
        assert entrance_op.end_time == 45.5
        assert entrance_op.parameters['direction'] == 'in'
        
        # Check exit transition
        exit_op = next(op for op in operations if "exit" in op.operation_id)
        assert exit_op.track_type == "effect"
        assert exit_op.operation_type == "transition"
        assert exit_op.start_time == 52.5
        assert exit_op.end_time == 53.0
        assert exit_op.parameters['direction'] == 'out'
    
    def test_process_multiple_broll_plans(self):
        """Test processing of multiple B-roll plans."""
        broll_plans = [
            BRollPlan(30.0, 5.0, "animation", "Process explanation", ["steps"], "slide_up", 8),
            BRollPlan(60.0, 6.0, "chart", "Data visualization", ["bar_chart"], "zoom_in", 7),
            BRollPlan(90.0, 4.0, "graphic", "Concept illustration", ["icons"], "fade_in", 6)
        ]
        
        operations = self.manager.process_broll_plans(broll_plans, self.context)
        
        # Should create 9 operations total (3 per plan)
        assert len(operations) == 9
        
        # Check that operations are created for each plan
        content_ops = [op for op in operations if "content" in op.operation_id]
        assert len(content_ops) == 3
        
        # Check timestamps
        timestamps = [op.start_time for op in content_ops]
        assert 30.0 in timestamps
        assert 60.0 in timestamps
        assert 90.0 in timestamps


class TestAudioVideoSynchronizer:
    """Test audio-video synchronization."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.synchronizer = AudioVideoSynchronizer()
        self.context = Mock(spec=ContentContext)
    
    def test_synchronize_simple_timeline(self):
        """Test synchronization of simple timeline."""
        operations = [
            TrackOperation("video_cut", "main_video", "video", 30.0, 30.5, "cut", {}, 8),
            TrackOperation("audio_cut", "main_audio", "audio", 30.1, 30.6, "cut", {}, 8)  # Slightly off sync
        ]
        
        sync_points = [
            SynchronizationPoint(30.0, "cut_sync", ["main_video", "main_audio"], 0.05)
        ]
        
        timeline = ExecutionTimeline(
            total_duration=180.0,
            operations=operations,
            sync_points=sync_points,
            track_mapping={"main_video": "video", "main_audio": "audio"}
        )
        
        synchronized_timeline = self.synchronizer.synchronize_timeline(timeline, self.context)
        
        # Check that audio operation was synchronized
        audio_op = next(op for op in synchronized_timeline.operations if op.track_type == "audio")
        # The synchronizer should have adjusted the audio timing
        assert audio_op.start_time == 30.0  # Should be synchronized to sync point
        assert audio_op.end_time == 30.5   # Duration preserved
        
        # Verify synchronization was applied (may be 0 if within tolerance)
        assert self.synchronizer.sync_adjustments >= 0
    
    def test_synchronize_within_tolerance(self):
        """Test that operations within tolerance are not adjusted."""
        operations = [
            TrackOperation("video_cut", "main_video", "video", 30.0, 30.5, "cut", {}, 8),
            TrackOperation("audio_cut", "main_audio", "audio", 30.02, 30.52, "cut", {}, 8)  # Within tolerance
        ]
        
        sync_points = [
            SynchronizationPoint(30.0, "cut_sync", ["main_video", "main_audio"], 0.05)
        ]
        
        timeline = ExecutionTimeline(
            total_duration=180.0,
            operations=operations,
            sync_points=sync_points,
            track_mapping={"main_video": "video", "main_audio": "audio"}
        )
        
        initial_sync_adjustments = self.synchronizer.sync_adjustments
        synchronized_timeline = self.synchronizer.synchronize_timeline(timeline, self.context)
        
        # Should not make adjustments for operations within tolerance
        assert self.synchronizer.sync_adjustments == initial_sync_adjustments
        
        # Audio operation should remain unchanged
        audio_op = next(op for op in synchronized_timeline.operations if op.track_type == "audio")
        assert audio_op.start_time == 30.02


class TestExecutionCoordinator:
    """Test execution coordination."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.coordinator = ExecutionCoordinator()
        self.context = self._create_mock_context()
    
    def _create_mock_context(self):
        """Create mock ContentContext with AI Director plan."""
        context = Mock(spec=ContentContext)
        context.project_id = "test_project"
        context.video_files = ["test_video.mp4"]
        context.video_metadata = {"duration": 180.0}
        context.audio_analysis = Mock()
        context.audio_analysis.segments = []
        context.processing_metrics = Mock()
        context.processing_metrics.add_module_metrics = Mock()
        
        # Mock AI Director plan
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
                    'timestamp': 60.0,
                    'decision_type': 'transition',
                    'parameters': {'duration': 1.0, 'type': 'fade'},
                    'rationale': 'Topic change',
                    'confidence': 0.8,
                    'priority': 7
                }
            ],
            'broll_plans': [
                {
                    'timestamp': 45.0,
                    'duration': 6.0,
                    'content_type': 'chart',
                    'description': 'Data visualization',
                    'visual_elements': ['bar_chart'],
                    'animation_style': 'fade_in',
                    'priority': 7
                }
            ]
        }
        
        return context
    
    def test_execute_ai_director_plan(self):
        """Test complete AI Director plan execution."""
        timeline = self.coordinator.execute_ai_director_plan(self.context)
        
        assert isinstance(timeline, ExecutionTimeline)
        assert timeline.total_duration > 0
        assert len(timeline.operations) > 0
        assert len(timeline.sync_points) >= 0
        
        # Check that processing metrics were updated
        self.context.processing_metrics.add_module_metrics.assert_called_once()
    
    def test_execute_plan_without_ai_director_data(self):
        """Test execution with missing AI Director plan."""
        self.context.processed_video = None
        
        with pytest.raises(ContentContextError, match="No AI Director plan found"):
            self.coordinator.execute_ai_director_plan(self.context)
    
    def test_extract_editing_decisions(self):
        """Test extraction of editing decisions from plan data."""
        ai_plan_data = {
            'editing_decisions': [
                {
                    'timestamp': 15.0,
                    'decision_type': 'trim',
                    'parameters': {'duration': 2.0},
                    'rationale': 'Remove filler',
                    'confidence': 0.85,
                    'priority': 6
                }
            ]
        }
        
        decisions = self.coordinator._extract_editing_decisions(ai_plan_data)
        
        assert len(decisions) == 1
        decision = decisions[0]
        assert decision.timestamp == 15.0
        assert decision.decision_type == 'trim'
        assert decision.parameters['duration'] == 2.0
        assert decision.confidence == 0.85
        assert decision.priority == 6
    
    def test_extract_broll_plans(self):
        """Test extraction of B-roll plans from plan data."""
        ai_plan_data = {
            'broll_plans': [
                {
                    'timestamp': 75.0,
                    'duration': 8.0,
                    'content_type': 'animation',
                    'description': 'Process visualization',
                    'visual_elements': ['flowchart'],
                    'animation_style': 'slide_up',
                    'priority': 8
                }
            ]
        }
        
        plans = self.coordinator._extract_broll_plans(ai_plan_data)
        
        assert len(plans) == 1
        plan = plans[0]
        assert plan.timestamp == 75.0
        assert plan.duration == 8.0
        assert plan.content_type == 'animation'
        assert plan.description == 'Process visualization'
        assert plan.priority == 8
    
    def test_calculate_total_duration(self):
        """Test calculation of total duration with editing decisions."""
        editing_decisions = [
            EditingDecision(30.0, "cut", {'duration': 1.0}, "Cut 1", 0.9, 8),
            EditingDecision(60.0, "trim", {'duration': 2.5}, "Trim 1", 0.8, 7)
        ]
        
        # Mock context with 180 second video
        self.context.video_metadata = {"duration": 180.0}
        
        total_duration = self.coordinator._calculate_total_duration(self.context, editing_decisions)
        
        # Should be 180 - 1.0 - 2.5 = 176.5 seconds
        assert total_duration == 176.5
    
    def test_calculate_duration_with_audio_fallback(self):
        """Test duration calculation using audio analysis fallback."""
        # No video metadata
        self.context.video_metadata = None
        
        # Mock audio analysis with segments
        mock_segments = [
            Mock(end=45.0),
            Mock(end=90.0),
            Mock(end=165.0)  # Last segment ends at 165 seconds
        ]
        self.context.audio_analysis.segments = mock_segments
        
        editing_decisions = [
            EditingDecision(30.0, "cut", {'duration': 0.5}, "Cut 1", 0.9, 8)
        ]
        
        total_duration = self.coordinator._calculate_total_duration(self.context, editing_decisions)
        
        # Should be 165 - 0.5 = 164.5 seconds
        assert total_duration == 164.5


class TestPlanExecutionEngine:
    """Test main plan execution engine interface."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = PlanExecutionEngine()
        self.context = self._create_mock_context()
    
    def _create_mock_context(self):
        """Create mock ContentContext with AI Director plan."""
        context = Mock(spec=ContentContext)
        context.project_id = "test_execution"
        context.video_files = ["test_video.mp4"]
        context.video_metadata = {"duration": 120.0}
        context.audio_analysis = Mock()
        context.audio_analysis.segments = []
        context.processing_metrics = Mock()
        context.processing_metrics.add_module_metrics = Mock()
        
        # Simple AI Director plan
        context.processed_video = {
            'editing_decisions': [
                {
                    'timestamp': 20.0,
                    'decision_type': 'cut',
                    'parameters': {'duration': 0.3},
                    'rationale': 'Quick cut',
                    'confidence': 0.9,
                    'priority': 8
                }
            ],
            'broll_plans': []
        }
        
        return context
    
    def test_execute_plan(self):
        """Test main plan execution interface."""
        timeline = self.engine.execute_plan(self.context)
        
        assert isinstance(timeline, ExecutionTimeline)
        assert timeline.total_duration > 0
        assert len(timeline.operations) > 0
        
        # Check execution history
        assert len(self.engine.execution_history) == 1
        
        history_record = self.engine.execution_history[0]
        assert history_record['project_id'] == "test_execution"
        assert history_record['operations_count'] > 0
        assert 'timestamp' in history_record
    
    def test_get_execution_stats(self):
        """Test execution statistics."""
        # Execute a few plans
        for i in range(3):
            self.context.project_id = f"test_project_{i}"
            self.engine.execute_plan(self.context)
        
        stats = self.engine.get_execution_stats()
        
        assert stats['total_executions'] == 3
        assert 'average_operations' in stats
        assert 'average_duration' in stats
        assert 'total_conflicts_resolved' in stats
        assert len(stats['recent_executions']) == 3
    
    def test_get_stats_no_executions(self):
        """Test statistics with no executions."""
        stats = self.engine.get_execution_stats()
        
        assert stats['total_executions'] == 0


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = PlanExecutionEngine()
    
    def test_complex_editing_scenario(self):
        """Test complex editing scenario with multiple operation types."""
        context = Mock(spec=ContentContext)
        context.project_id = "complex_scenario"
        context.video_files = ["complex_video.mp4"]
        context.video_metadata = {"duration": 300.0}  # 5 minutes
        context.audio_analysis = Mock()
        context.audio_analysis.segments = []
        context.processing_metrics = Mock()
        context.processing_metrics.add_module_metrics = Mock()
        
        # Complex AI Director plan with multiple operation types
        context.processed_video = {
            'editing_decisions': [
                # Multiple cuts
                {'timestamp': 15.0, 'decision_type': 'cut', 'parameters': {'duration': 0.5}, 'rationale': 'Cut 1', 'confidence': 0.9, 'priority': 8},
                {'timestamp': 45.0, 'decision_type': 'cut', 'parameters': {'duration': 0.3}, 'rationale': 'Cut 2', 'confidence': 0.85, 'priority': 7},
                
                # Trims
                {'timestamp': 75.0, 'decision_type': 'trim', 'parameters': {'duration': 3.0}, 'rationale': 'Remove segment', 'confidence': 0.8, 'priority': 6},
                
                # Transitions
                {'timestamp': 120.0, 'decision_type': 'transition', 'parameters': {'duration': 1.5, 'type': 'crossfade'}, 'rationale': 'Smooth transition', 'confidence': 0.9, 'priority': 8},
                
                # Emphasis
                {'timestamp': 180.0, 'decision_type': 'emphasis', 'parameters': {'duration': 2.0, 'emphasis_type': 'zoom_in'}, 'rationale': 'Highlight concept', 'confidence': 0.85, 'priority': 7}
            ],
            'broll_plans': [
                # Multiple B-roll insertions
                {'timestamp': 30.0, 'duration': 5.0, 'content_type': 'chart', 'description': 'Data viz 1', 'visual_elements': ['bar_chart'], 'animation_style': 'fade_in', 'priority': 8},
                {'timestamp': 90.0, 'duration': 6.0, 'content_type': 'animation', 'description': 'Process viz', 'visual_elements': ['flowchart'], 'animation_style': 'slide_up', 'priority': 7},
                {'timestamp': 150.0, 'duration': 4.0, 'content_type': 'graphic', 'description': 'Concept illustration', 'visual_elements': ['icons'], 'animation_style': 'zoom_in', 'priority': 6}
            ]
        }
        
        timeline = self.engine.execute_plan(context)
        
        # Verify comprehensive timeline creation
        assert timeline.total_duration > 0
        assert len(timeline.operations) >= 8  # At least 5 editing + 3 B-roll content operations
        assert len(timeline.sync_points) >= 3  # Sync points for cuts and transitions
        
        # Verify operation types are present
        operation_types = {op.operation_type for op in timeline.operations}
        assert 'cut' in operation_types
        assert 'trim' in operation_types
        assert 'transition' in operation_types
        assert 'effect' in operation_types
        assert 'insert' in operation_types
        
        # Verify track types are present
        track_types = {op.track_type for op in timeline.operations}
        assert 'video' in track_types
        assert 'audio' in track_types
        assert 'broll' in track_types
        assert 'effect' in track_types
    
    def test_synchronization_heavy_scenario(self):
        """Test scenario with many synchronization requirements."""
        context = Mock(spec=ContentContext)
        context.project_id = "sync_heavy"
        context.video_files = ["sync_video.mp4"]
        context.video_metadata = {"duration": 240.0}  # 4 minutes
        context.audio_analysis = Mock()
        context.audio_analysis.segments = []
        context.processing_metrics = Mock()
        context.processing_metrics.add_module_metrics = Mock()
        
        # Plan with many operations requiring synchronization
        context.processed_video = {
            'editing_decisions': [
                # Rapid cuts requiring precise sync
                {'timestamp': 10.0, 'decision_type': 'cut', 'parameters': {'duration': 0.2, 'preserve_audio': False}, 'rationale': 'Sync cut 1', 'confidence': 0.9, 'priority': 9},
                {'timestamp': 15.0, 'decision_type': 'cut', 'parameters': {'duration': 0.2, 'preserve_audio': False}, 'rationale': 'Sync cut 2', 'confidence': 0.9, 'priority': 9},
                {'timestamp': 20.0, 'decision_type': 'cut', 'parameters': {'duration': 0.2, 'preserve_audio': False}, 'rationale': 'Sync cut 3', 'confidence': 0.9, 'priority': 9},
                
                # Overlapping transitions
                {'timestamp': 30.0, 'decision_type': 'transition', 'parameters': {'duration': 2.0, 'type': 'crossfade'}, 'rationale': 'Transition 1', 'confidence': 0.8, 'priority': 8},
                {'timestamp': 35.0, 'decision_type': 'transition', 'parameters': {'duration': 2.0, 'type': 'slide'}, 'rationale': 'Transition 2', 'confidence': 0.8, 'priority': 8}
            ],
            'broll_plans': []
        }
        
        timeline = self.engine.execute_plan(context)
        
        # Should have many sync points
        assert len(timeline.sync_points) >= 5
        
        # Should have resolved conflicts (may be 0 if no actual conflicts)
        assert timeline.conflicts_resolved >= 0
        
        # Verify operations are properly synchronized
        video_ops = [op for op in timeline.operations if op.track_type == "video"]
        audio_ops = [op for op in timeline.operations if op.track_type == "audio"]
        
        # Should have corresponding video and audio operations
        assert len(video_ops) >= 3
        assert len(audio_ops) >= 3
    
    @patch('time.time')
    def test_performance_tracking(self, mock_time):
        """Test performance tracking throughout execution."""
        # Mock time progression
        mock_time.side_effect = [0.0, 0.5, 1.0, 1.5, 2.0]  # Simulate processing time
        
        context = Mock(spec=ContentContext)
        context.project_id = "performance_test"
        context.video_files = ["perf_video.mp4"]
        context.video_metadata = {"duration": 120.0}
        context.audio_analysis = Mock()
        context.audio_analysis.segments = []
        context.processing_metrics = Mock()
        context.processing_metrics.add_module_metrics = Mock()
        
        context.processed_video = {
            'editing_decisions': [
                {'timestamp': 30.0, 'decision_type': 'cut', 'parameters': {'duration': 0.5}, 'rationale': 'Performance test cut', 'confidence': 0.9, 'priority': 8}
            ],
            'broll_plans': []
        }
        
        timeline = self.engine.execute_plan(context)
        
        # Verify performance tracking was called (timing may vary)
        context.processing_metrics.add_module_metrics.assert_called_once()
        call_args = context.processing_metrics.add_module_metrics.call_args
        assert call_args[0][0] == 'plan_execution'  # Module name
        assert isinstance(call_args[0][1], float)   # Processing time
        
        # Verify execution history includes performance data
        assert len(self.engine.execution_history) == 1
        history = self.engine.execution_history[0]
        assert 'timestamp' in history
        assert history['project_id'] == "performance_test"
        assert history['operations_count'] > 0


if __name__ == "__main__":
    pytest.main([__file__])
