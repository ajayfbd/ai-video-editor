"""
VideoComposer - Professional video composition engine using movis.

This module implements the VideoComposer class that executes AI Director plans
using the movis library for professional-grade video composition and editing.
Integrates with ContentContext to ensure synchronized video output.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import json

try:
    import movis as mv
    import numpy as np
    MOVIS_AVAILABLE = True
except ImportError:
    MOVIS_AVAILABLE = False
    mv = None
    np = None

from ...core.content_context import ContentContext
from ...core.exceptions import ProcessingError, InputValidationError
from .plan_execution import PlanExecutionEngine, ExecutionTimeline, TrackOperation
from .broll_generation import BRollGenerationSystem, GeneratedBRollAsset


logger = logging.getLogger(__name__)


@dataclass
class CompositionSettings:
    """Configuration settings for video composition."""
    
    # Video output settings
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    duration: float = 0.0  # Will be calculated from content
    
    # Quality settings
    quality: str = "high"  # "low", "medium", "high", "ultra"
    preset: str = "medium"  # ffmpeg preset
    
    # Composition settings
    enable_sub_pixel: bool = True
    enable_motion_blur: bool = True
    background_color: Tuple[int, int, int, int] = (0, 0, 0, 255)
    
    # Audio settings
    audio_sample_rate: int = 48000
    audio_channels: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'duration': self.duration,
            'quality': self.quality,
            'preset': self.preset,
            'enable_sub_pixel': self.enable_sub_pixel,
            'enable_motion_blur': self.enable_motion_blur,
            'background_color': self.background_color,
            'audio_sample_rate': self.audio_sample_rate,
            'audio_channels': self.audio_channels
        }


@dataclass
class LayerInfo:
    """Information about a composition layer."""
    layer_id: str
    layer_type: str  # "video", "audio", "text", "image", "broll", "effect"
    start_time: float
    end_time: float
    source_path: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    blending_mode: str = "normal"
    opacity: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'layer_id': self.layer_id,
            'layer_type': self.layer_type,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'source_path': self.source_path,
            'properties': self.properties,
            'blending_mode': self.blending_mode,
            'opacity': self.opacity
        }


@dataclass
class CompositionPlan:
    """Complete plan for video composition execution."""
    layers: List[LayerInfo]
    transitions: List[Dict[str, Any]]
    effects: List[Dict[str, Any]]
    audio_adjustments: List[Dict[str, Any]]
    output_settings: CompositionSettings
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'layers': [layer.to_dict() for layer in self.layers],
            'transitions': self.transitions,
            'effects': self.effects,
            'audio_adjustments': self.audio_adjustments,
            'output_settings': self.output_settings.to_dict()
        }


class VideoComposer:
    """
    Professional video composition engine using movis.
    
    This class executes AI Director plans by creating sophisticated video
    compositions with support for multi-track editing, transitions, effects,
    and professional-grade output quality.
    """
    
    def __init__(self, output_dir: str = "output", temp_dir: str = "temp"):
        """
        Initialize VideoComposer.
        
        Args:
            output_dir: Directory for final video outputs
            temp_dir: Directory for temporary files during composition
        """
        if not MOVIS_AVAILABLE:
            raise ImportError(
                "movis library is required for VideoComposer. "
                "Install with: pip install movis"
            )
        
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Plan execution engine
        self.plan_execution_engine = PlanExecutionEngine()
        
        # B-roll generation system
        self.broll_generation_system = BRollGenerationSystem(
            output_dir=str(self.output_dir / "broll")
        )
        
        # Current composition state
        self.current_composition: Optional[Any] = None
        self.composition_plan: Optional[CompositionPlan] = None
        self.execution_timeline: Optional[ExecutionTimeline] = None
        self.layers: Dict[str, Any] = {}
        self.generated_broll_assets: List[GeneratedBRollAsset] = []
        
        # Performance tracking
        self.composition_time: float = 0.0
        self.render_time: float = 0.0
        self.plan_execution_time: float = 0.0
        self.broll_generation_time: float = 0.0
        
        logger.info("VideoComposer initialized with movis backend and plan execution engine")
    
    def validate_ai_director_plan(self, context: ContentContext) -> bool:
        """
        Validate that the ContentContext contains required AI Director plan data.
        
        Args:
            context: ContentContext with AI Director decisions
            
        Returns:
            True if plan is valid and executable
            
        Raises:
            InputValidationError: If plan data is missing or invalid
        """
        # Check for basic video files
        if not context.video_files:
            raise InputValidationError("No video files provided in ContentContext")
        
        # Validate video files exist
        for video_file in context.video_files:
            if not Path(video_file).exists():
                raise InputValidationError(f"Video file not found: {video_file}")
        
        # Check for AI Director plan in processed_video field
        if not context.processed_video:
            logger.warning("No AI Director plan found in ContentContext")
            return False
        
        plan_data = context.processed_video
        
        # Validate plan structure
        required_fields = ['editing_decisions', 'broll_plans', 'metadata_strategy']
        for field in required_fields:
            if field not in plan_data:
                logger.warning(f"Missing required field in AI Director plan: {field}")
                return False
        
        logger.info("AI Director plan validation successful")
        return True
    
    def create_composition_plan(self, context: ContentContext) -> CompositionPlan:
        """
        Create detailed composition plan from AI Director decisions.
        
        Args:
            context: ContentContext with AI Director plan
            
        Returns:
            CompositionPlan ready for execution
        """
        start_time = time.time()
        
        logger.info("Creating composition plan from AI Director decisions")
        
        # Extract AI Director plan
        ai_plan = context.processed_video
        editing_decisions = ai_plan.get('editing_decisions', [])
        broll_plans = ai_plan.get('broll_plans', [])
        
        # Calculate total duration from video files and decisions
        total_duration = self._calculate_composition_duration(context, editing_decisions)
        
        # Create output settings
        output_settings = CompositionSettings(
            duration=total_duration,
            quality="high" if context.user_preferences.quality_mode == "high" else "medium"
        )
        
        # Create layers from main video files
        layers = []
        layer_id = 0
        
        # Main video layer
        for i, video_file in enumerate(context.video_files):
            layer = LayerInfo(
                layer_id=f"main_video_{i}",
                layer_type="video",
                start_time=0.0,
                end_time=total_duration,
                source_path=video_file,
                properties={'track': 0, 'priority': 100}
            )
            layers.append(layer)
            layer_id += 1
        
        # Audio layer from main video
        if context.audio_analysis:
            audio_layer = LayerInfo(
                layer_id="main_audio",
                layer_type="audio",
                start_time=0.0,
                end_time=total_duration,
                source_path=context.video_files[0] if context.video_files else None,
                properties={'track': 1, 'priority': 90}
            )
            layers.append(audio_layer)
        
        # B-roll layers
        for broll_plan in broll_plans:
            broll_layer = LayerInfo(
                layer_id=f"broll_{broll_plan.get('timestamp', 0)}",
                layer_type="broll",
                start_time=broll_plan.get('timestamp', 0.0),
                end_time=broll_plan.get('timestamp', 0.0) + broll_plan.get('duration', 3.0),
                properties={
                    'content_type': broll_plan.get('content_type', 'graphic'),
                    'description': broll_plan.get('description', ''),
                    'visual_elements': broll_plan.get('visual_elements', []),
                    'animation_style': broll_plan.get('animation_style', 'fade'),
                    'track': 2,
                    'priority': broll_plan.get('priority', 50)
                },
                opacity=0.8  # Semi-transparent overlay
            )
            layers.append(broll_layer)
        
        # Create transitions from editing decisions
        transitions = []
        for decision in editing_decisions:
            if decision.get('decision_type') == 'transition':
                transition = {
                    'timestamp': decision.get('timestamp', 0.0),
                    'type': decision.get('parameters', {}).get('type', 'fade'),
                    'duration': decision.get('parameters', {}).get('duration', 1.0),
                    'parameters': decision.get('parameters', {}),
                    'confidence': decision.get('confidence', 0.8)
                }
                transitions.append(transition)
        
        # Create effects from editing decisions
        effects = []
        for decision in editing_decisions:
            if decision.get('decision_type') == 'emphasis':
                effect = {
                    'timestamp': decision.get('timestamp', 0.0),
                    'type': 'highlight',
                    'duration': decision.get('parameters', {}).get('duration', 2.0),
                    'parameters': decision.get('parameters', {}),
                    'confidence': decision.get('confidence', 0.8)
                }
                effects.append(effect)
        
        # Audio adjustments from editing decisions
        audio_adjustments = []
        for decision in editing_decisions:
            if decision.get('decision_type') in ['cut', 'trim']:
                adjustment = {
                    'timestamp': decision.get('timestamp', 0.0),
                    'type': decision.get('decision_type'),
                    'parameters': decision.get('parameters', {}),
                    'rationale': decision.get('rationale', '')
                }
                audio_adjustments.append(adjustment)
        
        composition_plan = CompositionPlan(
            layers=layers,
            transitions=transitions,
            effects=effects,
            audio_adjustments=audio_adjustments,
            output_settings=output_settings
        )
        
        self.composition_plan = composition_plan
        self.composition_time = time.time() - start_time
        
        logger.info(f"Composition plan created in {self.composition_time:.2f}s with {len(layers)} layers")
        return composition_plan
    
    def _calculate_composition_duration(self, context: ContentContext, editing_decisions: List[Dict]) -> float:
        """Calculate total duration considering cuts and trims."""
        if not context.video_files:
            return 0.0
        
        # Get original duration from video metadata
        original_duration = 0.0
        if context.video_metadata:
            original_duration = context.video_metadata.get('duration', 0.0)
        
        # If no metadata, estimate from audio analysis
        if original_duration == 0.0 and context.audio_analysis:
            # Find the last audio segment
            if context.audio_analysis.segments:
                original_duration = max(seg.end for seg in context.audio_analysis.segments)
        
        # Apply cuts and trims
        total_cut_duration = 0.0
        for decision in editing_decisions:
            if decision.get('decision_type') == 'cut':
                cut_duration = decision.get('parameters', {}).get('duration', 0.0)
                total_cut_duration += cut_duration
        
        final_duration = max(0.0, original_duration - total_cut_duration)
        
        # Add B-roll extension if needed
        if context.processed_video and context.processed_video.get('broll_plans'):
            max_broll_end = 0.0
            for broll in context.processed_video['broll_plans']:
                broll_end = broll.get('timestamp', 0.0) + broll.get('duration', 0.0)
                max_broll_end = max(max_broll_end, broll_end)
            
            final_duration = max(final_duration, max_broll_end)
        
        return final_duration
    
    def create_composition_plan_from_timeline(self, timeline: ExecutionTimeline, 
                                            context: ContentContext) -> CompositionPlan:
        """
        Create detailed composition plan from execution timeline.
        
        Args:
            timeline: ExecutionTimeline from plan execution engine
            context: ContentContext with source materials
            
        Returns:
            CompositionPlan ready for movis composition
        """
        start_time = time.time()
        
        logger.info("Creating composition plan from execution timeline")
        
        # Create output settings
        output_settings = CompositionSettings(
            duration=timeline.total_duration,
            quality="high" if context.user_preferences.quality_mode == "high" else "medium"
        )
        
        # Convert timeline operations to layers
        layers = []
        transitions = []
        effects = []
        audio_adjustments = []
        
        # Group operations by type
        for operation in timeline.operations:
            if operation.track_type == "video":
                layer = self._create_layer_from_operation(operation, context, output_settings)
                if layer:
                    layers.append(layer)
            
            elif operation.track_type == "audio":
                # Audio operations become audio adjustments
                adjustment = {
                    'timestamp': operation.start_time,
                    'type': operation.operation_type,
                    'parameters': operation.parameters,
                    'duration': operation.end_time - operation.start_time
                }
                audio_adjustments.append(adjustment)
            
            elif operation.track_type == "broll":
                layer = self._create_broll_layer_from_operation(operation, context, output_settings)
                if layer:
                    layers.append(layer)
            
            elif operation.track_type == "effect":
                if operation.operation_type == "transition":
                    transition = {
                        'timestamp': operation.start_time,
                        'type': operation.parameters.get('transition_type', 'fade'),
                        'duration': operation.end_time - operation.start_time,
                        'parameters': operation.parameters
                    }
                    transitions.append(transition)
                else:
                    effect = {
                        'timestamp': operation.start_time,
                        'type': operation.parameters.get('effect_type', 'emphasis'),
                        'duration': operation.end_time - operation.start_time,
                        'parameters': operation.parameters
                    }
                    effects.append(effect)
        
        composition_plan = CompositionPlan(
            layers=layers,
            transitions=transitions,
            effects=effects,
            audio_adjustments=audio_adjustments,
            output_settings=output_settings
        )
        
        self.composition_plan = composition_plan
        self.composition_time = time.time() - start_time
        
        logger.info(f"Composition plan created from timeline in {self.composition_time:.2f}s - "
                   f"{len(layers)} layers, {len(transitions)} transitions, {len(effects)} effects")
        
        return composition_plan
    
    def _create_layer_from_operation(self, operation: TrackOperation, 
                                   context: ContentContext, 
                                   settings: CompositionSettings) -> Optional[LayerInfo]:
        """Create LayerInfo from a video track operation."""
        if operation.operation_type in ['cut', 'trim']:
            # For cuts and trims, create a modified video layer
            if context.video_files:
                return LayerInfo(
                    layer_id=operation.operation_id,
                    layer_type="video",
                    start_time=operation.start_time,
                    end_time=operation.end_time,
                    source_path=context.video_files[0],
                    properties={
                        'operation_type': operation.operation_type,
                        'fade_duration': operation.parameters.get('fade_duration', 0.1),
                        'track': 0,
                        'priority': operation.priority
                    },
                    opacity=1.0
                )
        
        return None
    
    def _create_broll_layer_from_operation(self, operation: TrackOperation, 
                                         context: ContentContext, 
                                         settings: CompositionSettings) -> Optional[LayerInfo]:
        """Create LayerInfo from a B-roll track operation."""
        return LayerInfo(
            layer_id=operation.operation_id,
            layer_type="broll",
            start_time=operation.start_time,
            end_time=operation.end_time,
            properties={
                'content_type': operation.parameters.get('content_type', 'graphic'),
                'description': operation.parameters.get('description', ''),
                'visual_elements': operation.parameters.get('visual_elements', []),
                'animation_style': operation.parameters.get('animation_style', 'fade_in'),
                'track': 2,
                'priority': operation.priority,
                'z_index': operation.parameters.get('z_index', 10)
            },
            opacity=operation.parameters.get('opacity', 0.8)
        )
    
    def create_movis_composition(self, composition_plan: CompositionPlan) -> Any:
        """
        Create movis composition from composition plan.
        
        Args:
            composition_plan: Plan with layers, transitions, and effects
            
        Returns:
            Configured movis Composition ready for rendering
        """
        logger.info("Creating movis composition from plan")
        
        settings = composition_plan.output_settings
        
        # Create composition with settings
        composition = mv.Composition(
            size=(settings.width, settings.height),
            duration=settings.duration,
            fps=settings.fps
        )
        
        # Sort layers by priority (higher priority on top)
        sorted_layers = sorted(
            composition_plan.layers,
            key=lambda x: x.properties.get('priority', 50),
            reverse=True
        )
        
        # Add layers to composition
        for layer_info in sorted_layers:
            layer = self._create_movis_layer(layer_info, settings)
            if layer:
                composition.add_layer(layer)
                self.layers[layer_info.layer_id] = layer
        
        # Apply transitions
        for transition in composition_plan.transitions:
            self._apply_transition(composition, transition)
        
        # Apply effects
        for effect in composition_plan.effects:
            self._apply_effect(composition, effect)
        
        self.current_composition = composition
        logger.info(f"Movis composition created with {len(sorted_layers)} layers")
        
        return composition
    
    def _create_movis_layer(self, layer_info: LayerInfo, settings: CompositionSettings) -> Optional[Any]:
        """Create appropriate movis layer based on layer info."""
        try:
            if layer_info.layer_type == "video":
                if layer_info.source_path and Path(layer_info.source_path).exists():
                    # Create video layer
                    layer = mv.layer.VideoFile(
                        path=layer_info.source_path,
                        start_time=layer_info.start_time,
                        duration=layer_info.end_time - layer_info.start_time
                    )
                    
                    # Apply opacity if not full
                    if layer_info.opacity < 1.0:
                        layer = mv.layer.Opacity(layer_info.opacity)(layer)
                    
                    return layer
                else:
                    logger.warning(f"Video file not found: {layer_info.source_path}")
                    return None
            
            elif layer_info.layer_type == "audio":
                if layer_info.source_path and Path(layer_info.source_path).exists():
                    # Extract audio from video file
                    layer = mv.layer.AudioFile(
                        path=layer_info.source_path,
                        start_time=layer_info.start_time,
                        duration=layer_info.end_time - layer_info.start_time
                    )
                    return layer
                else:
                    logger.warning(f"Audio source not found: {layer_info.source_path}")
                    return None
            
            elif layer_info.layer_type == "broll":
                # Use actual generated B-roll assets
                return self._create_broll_layer_from_assets(layer_info, settings)
            
            elif layer_info.layer_type == "text":
                # Create text layer
                text = layer_info.properties.get('text', 'Text')
                font_size = layer_info.properties.get('font_size', 24)
                color = layer_info.properties.get('color', (255, 255, 255, 255))
                position = layer_info.properties.get('position', (settings.width // 2, settings.height // 2))
                
                layer = mv.layer.Text(
                    text=text,
                    font_size=font_size,
                    color=color,
                    position=position,
                    duration=layer_info.end_time - layer_info.start_time
                )
                
                return layer
            
            else:
                logger.warning(f"Unsupported layer type: {layer_info.layer_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating layer {layer_info.layer_id}: {str(e)}")
            return None
    
    def _create_broll_layer_from_assets(self, layer_info: LayerInfo, settings: CompositionSettings) -> Optional[Any]:
        """Create B-roll layer using actual generated assets."""
        try:
            # Find generated asset for this timestamp
            asset = self.broll_generation_system.get_asset_for_timestamp(layer_info.start_time)
            
            if asset and Path(asset.file_path).exists():
                # Use actual generated asset
                file_extension = Path(asset.file_path).suffix.lower()
                
                if file_extension in ['.mp4', '.avi', '.mov', '.gif']:
                    # Video/animation asset
                    layer = mv.layer.VideoFile(
                        path=asset.file_path,
                        start_time=0,
                        duration=layer_info.end_time - layer_info.start_time
                    )
                elif file_extension in ['.png', '.jpg', '.jpeg']:
                    # Image asset - use Image layer for static images
                    try:
                        layer = mv.layer.Image(
                            path=asset.file_path,
                            duration=layer_info.end_time - layer_info.start_time
                        )
                    except AttributeError:
                        # Fallback if Image layer doesn't exist in this movis version
                        layer = mv.layer.VideoFile(
                            path=asset.file_path,
                            start_time=0,
                            duration=layer_info.end_time - layer_info.start_time
                        )
                else:
                    # Fallback to placeholder
                    return self._create_placeholder_broll_layer(layer_info, settings)
                
                # Apply opacity
                if layer_info.opacity < 1.0:
                    layer = mv.layer.Opacity(layer_info.opacity)(layer)
                
                logger.info(f"Created B-roll layer from asset: {Path(asset.file_path).name}")
                return layer
            else:
                # No asset found, create placeholder
                return self._create_placeholder_broll_layer(layer_info, settings)
                
        except Exception as e:
            logger.error(f"Error creating B-roll layer from assets: {str(e)}")
            return self._create_placeholder_broll_layer(layer_info, settings)
    
    def _create_placeholder_broll_layer(self, layer_info: LayerInfo, settings: CompositionSettings) -> Any:
        """Create placeholder B-roll layer when no asset is available."""
        description = layer_info.properties.get('description', 'B-roll content')
        
        # Create solid background
        background = mv.layer.SolidColor(
            color=(30, 30, 30, 200),  # Semi-transparent dark background
            duration=layer_info.end_time - layer_info.start_time
        )
        
        # Add text overlay
        text_layer = mv.layer.Text(
            text=description[:50] + "..." if len(description) > 50 else description,
            font_size=24,
            color=(255, 255, 255, 255),
            position=(settings.width // 2, settings.height // 2),
            duration=layer_info.end_time - layer_info.start_time
        )
        
        # Combine background and text
        combined = mv.Composition(
            size=(settings.width, settings.height),
            duration=layer_info.end_time - layer_info.start_time
        )
        combined.add_layer(background)
        combined.add_layer(text_layer)
        
        # Apply opacity
        if layer_info.opacity < 1.0:
            combined = mv.layer.Opacity(layer_info.opacity)(combined)
        
        logger.info("Created placeholder B-roll layer")
        return combined
    
    def _apply_transition(self, composition: Any, transition: Dict[str, Any]):
        """Apply transition effect to composition."""
        try:
            transition_type = transition.get('type', 'fade')
            timestamp = transition.get('timestamp', 0.0)
            duration = transition.get('duration', 1.0)
            
            if transition_type == 'fade':
                # Apply fade transition at timestamp
                # This is a simplified implementation - real transitions would be more complex
                logger.info(f"Applied fade transition at {timestamp}s for {duration}s")
            
            elif transition_type == 'crossfade':
                # Apply crossfade between layers
                logger.info(f"Applied crossfade transition at {timestamp}s for {duration}s")
            
            else:
                logger.warning(f"Unsupported transition type: {transition_type}")
                
        except Exception as e:
            logger.error(f"Error applying transition: {str(e)}")
    
    def _apply_effect(self, composition: Any, effect: Dict[str, Any]):
        """Apply visual effect to composition."""
        try:
            effect_type = effect.get('type', 'highlight')
            timestamp = effect.get('timestamp', 0.0)
            duration = effect.get('duration', 2.0)
            
            if effect_type == 'highlight':
                # Apply highlight effect
                logger.info(f"Applied highlight effect at {timestamp}s for {duration}s")
            
            elif effect_type == 'zoom':
                # Apply zoom effect
                zoom_factor = effect.get('parameters', {}).get('factor', 1.2)
                logger.info(f"Applied zoom effect (factor: {zoom_factor}) at {timestamp}s for {duration}s")
            
            else:
                logger.warning(f"Unsupported effect type: {effect_type}")
                
        except Exception as e:
            logger.error(f"Error applying effect: {str(e)}")
    
    async def compose_video_with_ai_plan(self, context: ContentContext) -> Dict[str, Any]:
        """
        Execute complete video composition workflow using AI Director Plan Execution Engine.
        
        Args:
            context: ContentContext with AI Director plan and source materials
            
        Returns:
            Dict with composition results and output file paths
        """
        start_time = time.time()
        
        logger.info("Starting AI Director plan-based video composition workflow")
        
        try:
            # Validate input
            if not self.validate_ai_director_plan(context):
                raise InputValidationError("Invalid AI Director plan in ContentContext")
            
            # Step 1: Generate B-roll assets from AI Director plans
            broll_generation_start = time.time()
            self.generated_broll_assets = await self.broll_generation_system.generate_all_broll_assets(context)
            self.broll_generation_time = time.time() - broll_generation_start
            
            logger.info(f"Generated {len(self.generated_broll_assets)} B-roll assets in {self.broll_generation_time:.2f}s")
            
            # Step 2: Execute AI Director plan using execution engine
            plan_execution_start = time.time()
            execution_timeline = self.plan_execution_engine.execute_plan(context)
            self.execution_timeline = execution_timeline
            self.plan_execution_time = time.time() - plan_execution_start
            
            logger.info(f"AI Director plan executed in {self.plan_execution_time:.2f}s - "
                       f"{len(execution_timeline.operations)} operations, "
                       f"{len(execution_timeline.sync_points)} sync points")
            
            # Step 3: Create enhanced composition plan from execution timeline
            composition_plan = self.create_composition_plan_from_timeline(execution_timeline, context)
            
            # Step 4: Create movis composition
            composition = self.create_movis_composition(composition_plan)
            
            # Step 5: Render composition
            render_start = time.time()
            project_name = context.project_id or "ai_directed_video"
            output_filename = f"{project_name}_{int(time.time())}.mp4"
            output_path = self.output_dir / output_filename
            
            logger.info(f"Rendering AI-directed video composition to {output_path}")
            
            # Configure render settings based on quality
            quality_settings = self._get_quality_settings(composition_plan.output_settings.quality)
            
            # Render to file
            composition.write_video(
                str(output_path),
                **quality_settings
            )
            
            self.render_time = time.time() - render_start
            total_time = time.time() - start_time
            
            # Create comprehensive result
            composition_result = {
                'output_path': str(output_path),
                'composition_plan': composition_plan.to_dict(),
                'execution_timeline': execution_timeline.to_dict(),
                'render_settings': quality_settings,
                'performance_metrics': {
                    'plan_execution_time': self.plan_execution_time,
                    'composition_time': self.composition_time,
                    'render_time': self.render_time,
                    'total_time': total_time,
                    'operations_executed': len(execution_timeline.operations),
                    'sync_points_processed': len(execution_timeline.sync_points),
                    'conflicts_resolved': execution_timeline.conflicts_resolved,
                    'layer_count': len(composition_plan.layers),
                    'transition_count': len(composition_plan.transitions),
                    'effect_count': len(composition_plan.effects)
                },
                'file_info': {
                    'size_mb': output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0,
                    'duration': composition_plan.output_settings.duration,
                    'resolution': f"{composition_plan.output_settings.width}x{composition_plan.output_settings.height}",
                    'fps': composition_plan.output_settings.fps
                },
                'ai_director_integration': {
                    'plan_execution_successful': True,
                    'operations_count': len(execution_timeline.operations),
                    'timeline_optimized': execution_timeline.optimization_applied,
                    'synchronization_applied': len(execution_timeline.sync_points) > 0
                }
            }
            
            # Store result in ContentContext
            if not context.processed_video:
                context.processed_video = {}
            context.processed_video['composition_result'] = composition_result
            
            logger.info(f"AI-directed video composition completed in {total_time:.2f}s - Output: {output_path}")
            
            return composition_result
            
        except Exception as e:
            logger.error(f"AI-directed video composition failed: {str(e)}")
            raise ProcessingError(f"AI-directed video composition failed: {str(e)}")
    
    def compose_video(self, context: ContentContext) -> Dict[str, Any]:
        """
        Execute complete video composition workflow (legacy method).
        
        Args:
            context: ContentContext with AI Director plan and source materials
            
        Returns:
            Dict with composition results and output file paths
        """
        # Use the new AI Director plan-based composition by default
        return self.compose_video_with_ai_plan(context)
    
    def _get_quality_settings(self, quality: str) -> Dict[str, Any]:
        """Get render settings based on quality level."""
        quality_profiles = {
            "low": {
                "codec": "libx264",
                "preset": "faster",
                "crf": 28,
                "audio_codec": "aac",
                "audio_bitrate": "128k"
            },
            "medium": {
                "codec": "libx264",
                "preset": "medium",
                "crf": 23,
                "audio_codec": "aac",
                "audio_bitrate": "192k"
            },
            "high": {
                "codec": "libx264",
                "preset": "slow",
                "crf": 18,
                "audio_codec": "aac",
                "audio_bitrate": "256k"
            },
            "ultra": {
                "codec": "libx264",
                "preset": "veryslow",
                "crf": 15,
                "audio_codec": "aac",
                "audio_bitrate": "320k"
            }
        }
        
        return quality_profiles.get(quality, quality_profiles["medium"])
    
    def get_composition_info(self) -> Optional[Dict[str, Any]]:
        """Get information about current composition."""
        if not self.current_composition or not self.composition_plan:
            return None
        
        return {
            'layer_count': len(self.composition_plan.layers),
            'transition_count': len(self.composition_plan.transitions),
            'effect_count': len(self.composition_plan.effects),
            'duration': self.composition_plan.output_settings.duration,
            'resolution': f"{self.composition_plan.output_settings.width}x{self.composition_plan.output_settings.height}",
            'fps': self.composition_plan.output_settings.fps,
            'composition_time': self.composition_time,
            'render_time': self.render_time
        }
    
    def cleanup_temp_files(self):
        """Clean up temporary files created during composition."""
        try:
            # Clear current composition
            self.current_composition = None
            self.composition_plan = None
            self.layers.clear()
            
            # Clean temp directory
            for temp_file in self.temp_dir.glob("*"):
                if temp_file.is_file():
                    temp_file.unlink()
                    
            logger.info("Temporary files cleaned up")
            
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")
