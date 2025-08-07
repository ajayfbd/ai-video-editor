"""
AI Director Plan Execution Engine - Interprets AI Director plans into executable video operations.

This module implements the critical bridge between AI Director creative decisions and actual
video composition, translating high-level editing plans into precise video operations.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

from ...core.content_context import ContentContext
from ...core.exceptions import ProcessingError, InputValidationError, ContentContextError
from ..intelligence.ai_director import AIDirectorPlan, EditingDecision, BRollPlan, MetadataStrategy


logger = logging.getLogger(__name__)


@dataclass
class TrackOperation:
    """Represents a single operation on a specific track."""
    operation_id: str
    track_id: str
    track_type: str  # "video", "audio", "broll", "effect", "text"
    start_time: float
    end_time: float
    operation_type: str  # "cut", "trim", "transition", "insert", "effect"
    parameters: Dict[str, Any]
    priority: int  # Higher priority operations take precedence
    source_decision: Optional[str] = None  # Reference to original AI Director decision
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation_id': self.operation_id,
            'track_id': self.track_id,
            'track_type': self.track_type,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'operation_type': self.operation_type,
            'parameters': self.parameters,
            'priority': self.priority,
            'source_decision': self.source_decision
        }


@dataclass
class SynchronizationPoint:
    """Represents a critical audio-video synchronization point."""
    timestamp: float
    sync_type: str  # "cut_sync", "transition_sync", "broll_sync"
    affected_tracks: List[str]
    tolerance: float = 0.1  # Acceptable sync tolerance in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'sync_type': self.sync_type,
            'affected_tracks': self.affected_tracks,
            'tolerance': self.tolerance
        }


@dataclass
class ExecutionTimeline:
    """Comprehensive timeline with all operations across all tracks."""
    total_duration: float
    operations: List[TrackOperation]
    sync_points: List[SynchronizationPoint]
    track_mapping: Dict[str, str]  # track_id -> track_type
    conflicts_resolved: int = 0
    optimization_applied: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_duration': self.total_duration,
            'operations': [op.to_dict() for op in self.operations],
            'sync_points': [sp.to_dict() for sp in self.sync_points],
            'track_mapping': self.track_mapping,
            'conflicts_resolved': self.conflicts_resolved,
            'optimization_applied': self.optimization_applied
        }


class EditingDecisionInterpreter:
    """Interprets AI Director editing decisions into specific video operations."""
    
    def __init__(self):
        self.operation_counter = 0
        
        # Decision type mapping to operation handlers
        self.decision_handlers = {
            'cut': self._handle_cut_decision,
            'trim': self._handle_trim_decision,
            'transition': self._handle_transition_decision,
            'emphasis': self._handle_emphasis_decision,
            'b_roll': self._handle_broll_decision
        }
        
        logger.info("EditingDecisionInterpreter initialized")
    
    def interpret_decisions(self, decisions: List[EditingDecision], 
                          context: ContentContext) -> List[TrackOperation]:
        """
        Convert AI Director editing decisions into track operations.
        
        Args:
            decisions: List of EditingDecision objects from AI Director
            context: ContentContext with video and audio information
            
        Returns:
            List of TrackOperation objects ready for timeline execution
        """
        operations = []
        
        for decision in decisions:
            try:
                handler = self.decision_handlers.get(decision.decision_type)
                if handler:
                    decision_operations = handler(decision, context)
                    operations.extend(decision_operations)
                else:
                    logger.warning(f"Unknown decision type: {decision.decision_type}")
                    
            except Exception as e:
                logger.error(f"Failed to interpret decision {decision.decision_type} at {decision.timestamp}: {str(e)}")
                continue
        
        logger.info(f"Interpreted {len(decisions)} decisions into {len(operations)} operations")
        return operations
    
    def _handle_cut_decision(self, decision: EditingDecision, 
                           context: ContentContext) -> List[TrackOperation]:
        """Handle cut editing decisions."""
        operations = []
        
        # Create video cut operation
        video_cut = TrackOperation(
            operation_id=f"cut_video_{self._next_id()}",
            track_id="main_video",
            track_type="video",
            start_time=decision.timestamp,
            end_time=decision.timestamp + decision.parameters.get('duration', 0.5),
            operation_type="cut",
            parameters={
                'cut_type': decision.parameters.get('cut_type', 'hard'),
                'fade_duration': decision.parameters.get('fade_duration', 0.1),
                'preserve_audio': decision.parameters.get('preserve_audio', True)
            },
            priority=decision.priority,
            source_decision=f"cut_{decision.timestamp}"
        )
        operations.append(video_cut)
        
        # Create corresponding audio cut if needed
        if not decision.parameters.get('preserve_audio', True):
            audio_cut = TrackOperation(
                operation_id=f"cut_audio_{self._next_id()}",
                track_id="main_audio",
                track_type="audio",
                start_time=decision.timestamp,
                end_time=decision.timestamp + decision.parameters.get('duration', 0.5),
                operation_type="cut",
                parameters={
                    'fade_duration': decision.parameters.get('audio_fade', 0.05)
                },
                priority=decision.priority,
                source_decision=f"cut_{decision.timestamp}"
            )
            operations.append(audio_cut)
        
        return operations
    
    def _handle_trim_decision(self, decision: EditingDecision, 
                            context: ContentContext) -> List[TrackOperation]:
        """Handle trim editing decisions."""
        operations = []
        
        trim_start = decision.timestamp
        trim_duration = decision.parameters.get('duration', 2.0)
        trim_end = trim_start + trim_duration
        
        # Create video trim operation
        video_trim = TrackOperation(
            operation_id=f"trim_video_{self._next_id()}",
            track_id="main_video",
            track_type="video",
            start_time=trim_start,
            end_time=trim_end,
            operation_type="trim",
            parameters={
                'trim_type': decision.parameters.get('trim_type', 'remove'),
                'smooth_transition': decision.parameters.get('smooth_transition', True),
                'fade_in_duration': decision.parameters.get('fade_in', 0.2),
                'fade_out_duration': decision.parameters.get('fade_out', 0.2)
            },
            priority=decision.priority,
            source_decision=f"trim_{decision.timestamp}"
        )
        operations.append(video_trim)
        
        # Create corresponding audio trim
        audio_trim = TrackOperation(
            operation_id=f"trim_audio_{self._next_id()}",
            track_id="main_audio",
            track_type="audio",
            start_time=trim_start,
            end_time=trim_end,
            operation_type="trim",
            parameters={
                'fade_in_duration': decision.parameters.get('audio_fade_in', 0.1),
                'fade_out_duration': decision.parameters.get('audio_fade_out', 0.1)
            },
            priority=decision.priority,
            source_decision=f"trim_{decision.timestamp}"
        )
        operations.append(audio_trim)
        
        return operations
    
    def _handle_transition_decision(self, decision: EditingDecision, 
                                  context: ContentContext) -> List[TrackOperation]:
        """Handle transition editing decisions."""
        operations = []
        
        transition_duration = decision.parameters.get('duration', 1.0)
        transition_type = decision.parameters.get('type', 'fade')
        
        # Create transition operation
        transition_op = TrackOperation(
            operation_id=f"transition_{self._next_id()}",
            track_id="main_video",
            track_type="effect",
            start_time=decision.timestamp,
            end_time=decision.timestamp + transition_duration,
            operation_type="transition",
            parameters={
                'transition_type': transition_type,
                'duration': transition_duration,
                'easing': decision.parameters.get('easing', 'ease_in_out'),
                'direction': decision.parameters.get('direction', 'forward')
            },
            priority=decision.priority + 1,  # Transitions have higher priority
            source_decision=f"transition_{decision.timestamp}"
        )
        operations.append(transition_op)
        
        return operations
    
    def _handle_emphasis_decision(self, decision: EditingDecision, 
                                context: ContentContext) -> List[TrackOperation]:
        """Handle emphasis editing decisions."""
        operations = []
        
        emphasis_duration = decision.parameters.get('duration', 2.0)
        emphasis_type = decision.parameters.get('emphasis_type', 'zoom_in')
        
        # Create emphasis effect operation
        emphasis_op = TrackOperation(
            operation_id=f"emphasis_{self._next_id()}",
            track_id="main_video",
            track_type="effect",
            start_time=decision.timestamp,
            end_time=decision.timestamp + emphasis_duration,
            operation_type="effect",
            parameters={
                'effect_type': emphasis_type,
                'intensity': decision.parameters.get('intensity', 1.2),
                'duration': emphasis_duration,
                'fade_in': decision.parameters.get('fade_in', 0.3),
                'fade_out': decision.parameters.get('fade_out', 0.3)
            },
            priority=decision.priority,
            source_decision=f"emphasis_{decision.timestamp}"
        )
        operations.append(emphasis_op)
        
        return operations
    
    def _handle_broll_decision(self, decision: EditingDecision, 
                             context: ContentContext) -> List[TrackOperation]:
        """Handle B-roll insertion decisions."""
        operations = []
        
        broll_duration = decision.parameters.get('duration', 5.0)
        
        # Create B-roll insertion operation
        broll_op = TrackOperation(
            operation_id=f"broll_{self._next_id()}",
            track_id=f"broll_{decision.timestamp}",
            track_type="broll",
            start_time=decision.timestamp,
            end_time=decision.timestamp + broll_duration,
            operation_type="insert",
            parameters={
                'content_type': decision.parameters.get('content_type', 'graphic'),
                'description': decision.parameters.get('description', ''),
                'visual_elements': decision.parameters.get('visual_elements', []),
                'animation_style': decision.parameters.get('animation_style', 'fade_in'),
                'opacity': decision.parameters.get('opacity', 0.8),
                'position': decision.parameters.get('position', 'overlay')
            },
            priority=decision.priority,
            source_decision=f"broll_{decision.timestamp}"
        )
        operations.append(broll_op)
        
        return operations
    
    def _next_id(self) -> int:
        """Generate next operation ID."""
        self.operation_counter += 1
        return self.operation_counter


class TimelineManager:
    """Manages multi-track timeline coordination and conflict resolution."""
    
    def __init__(self):
        self.tracks = {}
        self.conflicts_resolved = 0
        
        logger.info("TimelineManager initialized")
    
    def create_timeline(self, operations: List[TrackOperation], 
                       total_duration: float) -> ExecutionTimeline:
        """
        Create comprehensive execution timeline from operations.
        
        Args:
            operations: List of track operations to coordinate
            total_duration: Total duration of the composition
            
        Returns:
            ExecutionTimeline with coordinated operations and sync points
        """
        # Sort operations by timestamp and priority
        sorted_operations = sorted(operations, key=lambda x: (x.start_time, -x.priority))
        
        # Resolve conflicts between overlapping operations
        resolved_operations = self._resolve_conflicts(sorted_operations)
        
        # Create synchronization points
        sync_points = self._create_sync_points(resolved_operations)
        
        # Build track mapping
        track_mapping = self._build_track_mapping(resolved_operations)
        
        timeline = ExecutionTimeline(
            total_duration=total_duration,
            operations=resolved_operations,
            sync_points=sync_points,
            track_mapping=track_mapping,
            conflicts_resolved=self.conflicts_resolved,
            optimization_applied=True
        )
        
        logger.info(f"Created timeline with {len(resolved_operations)} operations, "
                   f"{len(sync_points)} sync points, {self.conflicts_resolved} conflicts resolved")
        
        return timeline
    
    def _resolve_conflicts(self, operations: List[TrackOperation]) -> List[TrackOperation]:
        """Resolve conflicts between overlapping operations."""
        resolved = []
        track_last_end = {}
        
        for operation in operations:
            track_id = operation.track_id
            
            # Check for overlap with previous operation on same track
            if track_id in track_last_end:
                last_end = track_last_end[track_id]
                
                if operation.start_time < last_end:
                    # Conflict detected - resolve based on priority
                    overlap_duration = last_end - operation.start_time
                    
                    if overlap_duration > 0.1:  # Significant overlap
                        # Adjust start time to avoid conflict
                        operation.start_time = last_end + 0.05
                        operation.end_time = operation.start_time + (operation.end_time - operation.start_time)
                        self.conflicts_resolved += 1
                        
                        logger.debug(f"Resolved conflict for operation {operation.operation_id}: "
                                   f"moved start time to {operation.start_time}")
            
            resolved.append(operation)
            track_last_end[track_id] = operation.end_time
        
        return resolved
    
    def _create_sync_points(self, operations: List[TrackOperation]) -> List[SynchronizationPoint]:
        """Create synchronization points for critical timing coordination."""
        sync_points = []
        
        # Find operations that require synchronization
        sync_operations = [op for op in operations if op.operation_type in ['cut', 'trim', 'transition']]
        
        for operation in sync_operations:
            # Create sync point at operation start
            sync_point = SynchronizationPoint(
                timestamp=operation.start_time,
                sync_type=f"{operation.operation_type}_sync",
                affected_tracks=[operation.track_id],
                tolerance=0.05 if operation.operation_type == 'cut' else 0.1
            )
            sync_points.append(sync_point)
            
            # Create sync point at operation end for transitions
            if operation.operation_type == 'transition':
                end_sync = SynchronizationPoint(
                    timestamp=operation.end_time,
                    sync_type="transition_end_sync",
                    affected_tracks=[operation.track_id],
                    tolerance=0.1
                )
                sync_points.append(end_sync)
        
        # Remove duplicate sync points
        unique_sync_points = []
        seen_timestamps = set()
        
        for sync_point in sync_points:
            timestamp_key = round(sync_point.timestamp, 2)
            if timestamp_key not in seen_timestamps:
                unique_sync_points.append(sync_point)
                seen_timestamps.add(timestamp_key)
        
        return sorted(unique_sync_points, key=lambda x: x.timestamp)
    
    def _build_track_mapping(self, operations: List[TrackOperation]) -> Dict[str, str]:
        """Build mapping of track IDs to track types."""
        track_mapping = {}
        
        for operation in operations:
            if operation.track_id not in track_mapping:
                track_mapping[operation.track_id] = operation.track_type
        
        return track_mapping


class BRollInsertionManager:
    """Manages B-roll insertion based on AI Director timing decisions."""
    
    def __init__(self):
        self.broll_counter = 0
        
        logger.info("BRollInsertionManager initialized")
    
    def process_broll_plans(self, broll_plans: List[BRollPlan], 
                           context: ContentContext) -> List[TrackOperation]:
        """
        Convert AI Director B-roll plans into track operations.
        
        Args:
            broll_plans: List of BRollPlan objects from AI Director
            context: ContentContext with content information
            
        Returns:
            List of TrackOperation objects for B-roll insertion
        """
        operations = []
        
        for plan in broll_plans:
            try:
                broll_operations = self._create_broll_operations(plan, context)
                operations.extend(broll_operations)
                
            except Exception as e:
                logger.error(f"Failed to process B-roll plan at {plan.timestamp}: {str(e)}")
                continue
        
        logger.info(f"Processed {len(broll_plans)} B-roll plans into {len(operations)} operations")
        return operations
    
    def _create_broll_operations(self, plan: BRollPlan, 
                               context: ContentContext) -> List[TrackOperation]:
        """Create specific operations for a B-roll plan."""
        operations = []
        
        # Main B-roll content operation
        broll_op = TrackOperation(
            operation_id=f"broll_content_{self._next_id()}",
            track_id=f"broll_track_{plan.timestamp}",
            track_type="broll",
            start_time=plan.timestamp,
            end_time=plan.timestamp + plan.duration,
            operation_type="insert",
            parameters={
                'content_type': plan.content_type,
                'description': plan.description,
                'visual_elements': plan.visual_elements,
                'animation_style': plan.animation_style,
                'opacity': 0.9,  # Slightly transparent overlay
                'position': 'center_overlay',
                'z_index': 10  # Above main video
            },
            priority=plan.priority,
            source_decision=f"broll_plan_{plan.timestamp}"
        )
        operations.append(broll_op)
        
        # Add entrance transition
        entrance_transition = TrackOperation(
            operation_id=f"broll_entrance_{self._next_id()}",
            track_id=f"broll_track_{plan.timestamp}",
            track_type="effect",
            start_time=plan.timestamp,
            end_time=plan.timestamp + 0.5,
            operation_type="transition",
            parameters={
                'transition_type': plan.animation_style,
                'direction': 'in',
                'duration': 0.5
            },
            priority=plan.priority + 1,
            source_decision=f"broll_plan_{plan.timestamp}"
        )
        operations.append(entrance_transition)
        
        # Add exit transition
        exit_transition = TrackOperation(
            operation_id=f"broll_exit_{self._next_id()}",
            track_id=f"broll_track_{plan.timestamp}",
            track_type="effect",
            start_time=plan.timestamp + plan.duration - 0.5,
            end_time=plan.timestamp + plan.duration,
            operation_type="transition",
            parameters={
                'transition_type': plan.animation_style,
                'direction': 'out',
                'duration': 0.5
            },
            priority=plan.priority + 1,
            source_decision=f"broll_plan_{plan.timestamp}"
        )
        operations.append(exit_transition)
        
        return operations
    
    def _next_id(self) -> int:
        """Generate next B-roll ID."""
        self.broll_counter += 1
        return self.broll_counter


class AudioVideoSynchronizer:
    """Ensures audio-video synchronization throughout editing operations."""
    
    def __init__(self):
        self.sync_adjustments = 0
        
        logger.info("AudioVideoSynchronizer initialized")
    
    def synchronize_timeline(self, timeline: ExecutionTimeline, 
                           context: ContentContext) -> ExecutionTimeline:
        """
        Ensure audio-video synchronization across the timeline.
        
        Args:
            timeline: ExecutionTimeline to synchronize
            context: ContentContext with audio/video information
            
        Returns:
            Synchronized ExecutionTimeline
        """
        # Process each synchronization point
        for sync_point in timeline.sync_points:
            self._apply_synchronization(sync_point, timeline, context)
        
        # Verify overall synchronization
        self._verify_global_sync(timeline, context)
        
        logger.info(f"Applied {self.sync_adjustments} synchronization adjustments")
        return timeline
    
    def _apply_synchronization(self, sync_point: SynchronizationPoint, 
                             timeline: ExecutionTimeline, context: ContentContext):
        """Apply synchronization at a specific point."""
        timestamp = sync_point.timestamp
        tolerance = sync_point.tolerance
        
        # Find operations near this sync point - use a wider search range
        nearby_operations = [
            op for op in timeline.operations
            if abs(op.start_time - timestamp) <= 1.0 or abs(op.end_time - timestamp) <= 1.0
        ]
        
        # Group by track type
        video_ops = [op for op in nearby_operations if op.track_type == 'video']
        audio_ops = [op for op in nearby_operations if op.track_type == 'audio']
        
        # Ensure video and audio operations are synchronized
        if video_ops and audio_ops:
            self._sync_video_audio_operations(video_ops, audio_ops, sync_point)
    
    def _sync_video_audio_operations(self, video_ops: List[TrackOperation], 
                                   audio_ops: List[TrackOperation], 
                                   sync_point: SynchronizationPoint):
        """Synchronize video and audio operations."""
        target_timestamp = sync_point.timestamp
        
        # Adjust audio operations to match video timing
        for audio_op in audio_ops:
            time_diff = abs(audio_op.start_time - target_timestamp)
            if time_diff > sync_point.tolerance:
                # Adjust audio timing to match video
                duration = audio_op.end_time - audio_op.start_time
                
                audio_op.start_time = target_timestamp
                audio_op.end_time = target_timestamp + duration
                
                self.sync_adjustments += 1
                
                logger.debug(f"Synchronized audio operation {audio_op.operation_id} "
                           f"by {time_diff:.3f} seconds")
    
    def _verify_global_sync(self, timeline: ExecutionTimeline, context: ContentContext):
        """Verify overall synchronization across the timeline."""
        # Check for major sync issues
        video_operations = [op for op in timeline.operations if op.track_type == 'video']
        audio_operations = [op for op in timeline.operations if op.track_type == 'audio']
        
        # Verify timeline consistency
        if video_operations and audio_operations:
            video_duration = max(op.end_time for op in video_operations)
            audio_duration = max(op.end_time for op in audio_operations)
            
            if abs(video_duration - audio_duration) > 0.5:
                logger.warning(f"Potential sync issue: video duration {video_duration:.2f}s, "
                             f"audio duration {audio_duration:.2f}s")


class ExecutionCoordinator:
    """Orchestrates the entire plan execution process."""
    
    def __init__(self):
        self.decision_interpreter = EditingDecisionInterpreter()
        self.timeline_manager = TimelineManager()
        self.broll_manager = BRollInsertionManager()
        self.synchronizer = AudioVideoSynchronizer()
        
        logger.info("ExecutionCoordinator initialized")
    
    def execute_ai_director_plan(self, context: ContentContext) -> ExecutionTimeline:
        """
        Execute complete AI Director plan from ContentContext.
        
        Args:
            context: ContentContext with AI Director plan in processed_video
            
        Returns:
            ExecutionTimeline ready for VideoComposer
            
        Raises:
            ContentContextError: If AI Director plan is invalid or missing
        """
        start_time = time.time()
        
        try:
            # Validate AI Director plan
            if not context.processed_video:
                raise ContentContextError("No AI Director plan found in ContentContext")
            
            ai_plan_data = context.processed_video
            
            # Extract plan components
            editing_decisions = self._extract_editing_decisions(ai_plan_data)
            broll_plans = self._extract_broll_plans(ai_plan_data)
            
            # Calculate total duration
            total_duration = self._calculate_total_duration(context, editing_decisions)
            
            logger.info(f"Executing AI Director plan with {len(editing_decisions)} editing decisions, "
                       f"{len(broll_plans)} B-roll plans")
            
            # Step 1: Interpret editing decisions
            editing_operations = self.decision_interpreter.interpret_decisions(editing_decisions, context)
            
            # Step 2: Process B-roll plans
            broll_operations = self.broll_manager.process_broll_plans(broll_plans, context)
            
            # Step 3: Combine all operations
            all_operations = editing_operations + broll_operations
            
            # Step 4: Create coordinated timeline
            timeline = self.timeline_manager.create_timeline(all_operations, total_duration)
            
            # Step 5: Apply synchronization
            synchronized_timeline = self.synchronizer.synchronize_timeline(timeline, context)
            
            # Update processing metrics
            processing_time = time.time() - start_time
            context.processing_metrics.add_module_metrics('plan_execution', processing_time, 0)
            
            logger.info(f"AI Director plan execution completed in {processing_time:.2f}s")
            
            return synchronized_timeline
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"AI Director plan execution failed after {processing_time:.2f}s: {str(e)}")
            raise ContentContextError(
                f"AI Director plan execution failed: {str(e)}",
                context_state=context
            )
    
    def _extract_editing_decisions(self, ai_plan_data: Dict[str, Any]) -> List[EditingDecision]:
        """Extract editing decisions from AI Director plan data."""
        decisions = []
        
        editing_decisions_data = ai_plan_data.get('editing_decisions', [])
        
        for decision_data in editing_decisions_data:
            try:
                decision = EditingDecision(
                    timestamp=float(decision_data.get('timestamp', 0.0)),
                    decision_type=decision_data.get('decision_type', 'cut'),
                    parameters=decision_data.get('parameters', {}),
                    rationale=decision_data.get('rationale', ''),
                    confidence=float(decision_data.get('confidence', 0.8)),
                    priority=int(decision_data.get('priority', 5))
                )
                decisions.append(decision)
                
            except Exception as e:
                logger.warning(f"Failed to parse editing decision: {str(e)}")
                continue
        
        return decisions
    
    def _extract_broll_plans(self, ai_plan_data: Dict[str, Any]) -> List[BRollPlan]:
        """Extract B-roll plans from AI Director plan data."""
        plans = []
        
        broll_plans_data = ai_plan_data.get('broll_plans', [])
        
        for plan_data in broll_plans_data:
            try:
                plan = BRollPlan(
                    timestamp=float(plan_data.get('timestamp', 0.0)),
                    duration=float(plan_data.get('duration', 5.0)),
                    content_type=plan_data.get('content_type', 'graphic'),
                    description=plan_data.get('description', ''),
                    visual_elements=plan_data.get('visual_elements', []),
                    animation_style=plan_data.get('animation_style', 'fade_in'),
                    priority=int(plan_data.get('priority', 5))
                )
                plans.append(plan)
                
            except Exception as e:
                logger.warning(f"Failed to parse B-roll plan: {str(e)}")
                continue
        
        return plans
    
    def _calculate_total_duration(self, context: ContentContext, 
                                editing_decisions: List[EditingDecision]) -> float:
        """Calculate total duration considering editing decisions."""
        # Start with original duration
        original_duration = 0.0
        
        if context.video_metadata:
            original_duration = context.video_metadata.get('duration', 0.0)
        
        # If no metadata, estimate from audio analysis
        if original_duration == 0.0 and context.audio_analysis:
            if context.audio_analysis.segments:
                original_duration = max(seg.end for seg in context.audio_analysis.segments)
        
        # Apply cuts and trims
        total_removed = 0.0
        for decision in editing_decisions:
            if decision.decision_type in ['cut', 'trim']:
                removed_duration = decision.parameters.get('duration', 0.0)
                total_removed += removed_duration
        
        final_duration = max(0.0, original_duration - total_removed)
        
        # Ensure minimum duration
        return max(final_duration, 10.0)  # At least 10 seconds


class PlanExecutionEngine:
    """
    Main interface for AI Director Plan Execution Engine.
    
    This class provides the primary interface for executing AI Director plans,
    coordinating all components to translate high-level creative decisions
    into precise video operations.
    """
    
    def __init__(self):
        self.coordinator = ExecutionCoordinator()
        self.execution_history = []
        
        logger.info("PlanExecutionEngine initialized")
    
    def execute_plan(self, context: ContentContext) -> ExecutionTimeline:
        """
        Execute AI Director plan from ContentContext.
        
        Args:
            context: ContentContext with AI Director plan
            
        Returns:
            ExecutionTimeline ready for VideoComposer
            
        Raises:
            ContentContextError: If plan execution fails
        """
        try:
            timeline = self.coordinator.execute_ai_director_plan(context)
            
            # Store execution history
            execution_record = {
                'timestamp': time.time(),
                'project_id': context.project_id,
                'operations_count': len(timeline.operations),
                'sync_points_count': len(timeline.sync_points),
                'conflicts_resolved': timeline.conflicts_resolved,
                'total_duration': timeline.total_duration
            }
            self.execution_history.append(execution_record)
            
            return timeline
            
        except Exception as e:
            logger.error(f"Plan execution failed: {str(e)}")
            raise
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about plan executions."""
        if not self.execution_history:
            return {'total_executions': 0}
        
        return {
            'total_executions': len(self.execution_history),
            'average_operations': sum(record['operations_count'] for record in self.execution_history) / len(self.execution_history),
            'average_duration': sum(record['total_duration'] for record in self.execution_history) / len(self.execution_history),
            'total_conflicts_resolved': sum(record['conflicts_resolved'] for record in self.execution_history),
            'recent_executions': self.execution_history[-5:]  # Last 5 executions
        }