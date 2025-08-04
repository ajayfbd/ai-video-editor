"""
ContextManager - Manages ContentContext lifecycle and state tracking.

This module provides centralized management for ContentContext objects,
including creation, validation, persistence, and state tracking.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path

from .content_context import ContentContext, ContentType, UserPreferences
from .exceptions import ContentContextError, ContextIntegrityError


logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manages ContentContext lifecycle, state tracking, and persistence.
    
    Provides centralized management for ContentContext objects including:
    - Context creation and initialization
    - State validation and integrity checking
    - Checkpoint saving and recovery
    - Context persistence and loading
    """
    
    def __init__(self, storage_path: str = "temp/contexts"):
        """
        Initialize ContextManager.
        
        Args:
            storage_path: Directory path for storing context files
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._active_contexts: Dict[str, ContentContext] = {}
        
        logger.info(f"ContextManager initialized with storage path: {self.storage_path}")
    
    def create_context(self, video_files: List[str], content_type: ContentType,
                      user_preferences: Optional[UserPreferences] = None,
                      project_id: Optional[str] = None) -> ContentContext:
        """
        Create a new ContentContext with proper initialization.
        
        Args:
            video_files: List of video file paths
            content_type: Type of content being processed
            user_preferences: User processing preferences
            project_id: Optional project identifier
            
        Returns:
            Initialized ContentContext object
            
        Raises:
            ContentContextError: If context creation fails
        """
        try:
            if user_preferences is None:
                user_preferences = UserPreferences()
            
            context = ContentContext(
                project_id=project_id or "",
                video_files=video_files,
                content_type=content_type,
                user_preferences=user_preferences
            )
            
            # Validate video files exist
            for video_file in video_files:
                if not os.path.exists(video_file):
                    logger.warning(f"Video file not found: {video_file}")
            
            # Register context as active
            self._active_contexts[context.project_id] = context
            
            # Save initial checkpoint
            self.save_checkpoint(context, "initial")
            
            logger.info(f"Created ContentContext: {context.project_id}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to create ContentContext: {str(e)}")
            raise ContentContextError(f"Context creation failed: {str(e)}")
    
    def get_context(self, project_id: str) -> Optional[ContentContext]:
        """
        Get active ContentContext by project ID.
        
        Args:
            project_id: Project identifier
            
        Returns:
            ContentContext if found, None otherwise
        """
        return self._active_contexts.get(project_id)
    
    def validate_context(self, context: ContentContext) -> Dict[str, Any]:
        """
        Validate ContentContext integrity and completeness.
        
        Args:
            context: ContentContext to validate
            
        Returns:
            Validation result dictionary with status and issues
        """
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'score': 1.0
        }
        
        try:
            # Check required fields
            if not context.project_id:
                validation_result['issues'].append("Missing project_id")
                validation_result['valid'] = False
            
            if not context.video_files:
                validation_result['issues'].append("No video files specified")
                validation_result['valid'] = False
            
            # Check file existence
            missing_files = []
            for video_file in context.video_files:
                if not os.path.exists(video_file):
                    missing_files.append(video_file)
            
            if missing_files:
                validation_result['warnings'].append(f"Missing video files: {missing_files}")
                validation_result['score'] *= 0.8
            
            # Check data consistency
            if context.emotional_markers:
                for marker in context.emotional_markers:
                    if not (0.0 <= marker.intensity <= 1.0):
                        validation_result['issues'].append(f"Invalid emotional intensity: {marker.intensity}")
                        validation_result['valid'] = False
                    
                    if not (0.0 <= marker.confidence <= 1.0):
                        validation_result['issues'].append(f"Invalid emotional confidence: {marker.confidence}")
                        validation_result['valid'] = False
            
            if context.visual_highlights:
                for highlight in context.visual_highlights:
                    if not (0.0 <= highlight.thumbnail_potential <= 1.0):
                        validation_result['issues'].append(f"Invalid thumbnail potential: {highlight.thumbnail_potential}")
                        validation_result['valid'] = False
            
            # Check processing metrics consistency
            if context.processing_metrics.total_processing_time < 0:
                validation_result['issues'].append("Negative total processing time")
                validation_result['valid'] = False
            
            if context.cost_tracking.total_cost < 0:
                validation_result['issues'].append("Negative total cost")
                validation_result['valid'] = False
            
            # Calculate final score
            if validation_result['issues']:
                validation_result['score'] = 0.0
            elif validation_result['warnings']:
                validation_result['score'] *= 0.9
            
            logger.debug(f"Context validation completed for {context.project_id}: "
                        f"valid={validation_result['valid']}, score={validation_result['score']}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Context validation failed: {str(e)}")
            return {
                'valid': False,
                'issues': [f"Validation error: {str(e)}"],
                'warnings': [],
                'score': 0.0
            }
    
    def save_context(self, context: ContentContext) -> bool:
        """
        Save ContentContext to persistent storage.
        
        Args:
            context: ContentContext to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            context_file = self.storage_path / f"{context.project_id}.json"
            
            with open(context_file, 'w', encoding='utf-8') as f:
                f.write(context.to_json())
            
            logger.debug(f"Saved ContentContext: {context.project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save ContentContext {context.project_id}: {str(e)}")
            return False
    
    def load_context(self, project_id: str) -> Optional[ContentContext]:
        """
        Load ContentContext from persistent storage.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Loaded ContentContext or None if not found
        """
        try:
            context_file = self.storage_path / f"{project_id}.json"
            
            if not context_file.exists():
                logger.warning(f"Context file not found: {context_file}")
                return None
            
            with open(context_file, 'r', encoding='utf-8') as f:
                context_data = f.read()
            
            context = ContentContext.from_json(context_data)
            
            # Register as active context
            self._active_contexts[project_id] = context
            
            logger.info(f"Loaded ContentContext: {project_id}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to load ContentContext {project_id}: {str(e)}")
            return None
    
    def save_checkpoint(self, context: ContentContext, checkpoint_name: str) -> bool:
        """
        Save ContentContext checkpoint for recovery.
        
        Args:
            context: ContentContext to checkpoint
            checkpoint_name: Name for the checkpoint
            
        Returns:
            True if checkpoint saved successfully, False otherwise
        """
        try:
            checkpoint_dir = self.storage_path / "checkpoints" / context.project_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_file = checkpoint_dir / f"{checkpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                f.write(context.to_json())
            
            # Update context checkpoint list
            context.add_checkpoint(checkpoint_name)
            
            logger.debug(f"Saved checkpoint '{checkpoint_name}' for context {context.project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint '{checkpoint_name}' for context {context.project_id}: {str(e)}")
            return False
    
    def load_checkpoint(self, project_id: str, checkpoint_name: str) -> Optional[ContentContext]:
        """
        Load ContentContext from checkpoint.
        
        Args:
            project_id: Project identifier
            checkpoint_name: Name of checkpoint to load
            
        Returns:
            Loaded ContentContext from checkpoint or None if not found
        """
        try:
            checkpoint_dir = self.storage_path / "checkpoints" / project_id
            
            if not checkpoint_dir.exists():
                logger.warning(f"No checkpoints found for project: {project_id}")
                return None
            
            # Find the most recent checkpoint with the given name
            checkpoint_files = list(checkpoint_dir.glob(f"{checkpoint_name}_*.json"))
            
            if not checkpoint_files:
                logger.warning(f"Checkpoint '{checkpoint_name}' not found for project: {project_id}")
                return None
            
            # Get the most recent checkpoint
            latest_checkpoint = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                context_data = f.read()
            
            context = ContentContext.from_json(context_data)
            
            # Register as active context
            self._active_contexts[project_id] = context
            
            logger.info(f"Loaded checkpoint '{checkpoint_name}' for context {project_id}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint '{checkpoint_name}' for context {project_id}: {str(e)}")
            return None
    
    def list_checkpoints(self, project_id: str) -> List[Dict[str, Any]]:
        """
        List available checkpoints for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            List of checkpoint information dictionaries
        """
        try:
            checkpoint_dir = self.storage_path / "checkpoints" / project_id
            
            if not checkpoint_dir.exists():
                return []
            
            checkpoints = []
            for checkpoint_file in checkpoint_dir.glob("*.json"):
                try:
                    # Parse checkpoint name and timestamp from filename
                    filename = checkpoint_file.stem
                    parts = filename.rsplit('_', 2)
                    
                    if len(parts) >= 3:
                        checkpoint_name = '_'.join(parts[:-2])
                        timestamp_str = f"{parts[-2]}_{parts[-1]}"
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    else:
                        checkpoint_name = filename
                        timestamp = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                    
                    checkpoints.append({
                        'name': checkpoint_name,
                        'timestamp': timestamp,
                        'file_path': str(checkpoint_file),
                        'size': checkpoint_file.stat().st_size
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to parse checkpoint file {checkpoint_file}: {str(e)}")
                    continue
            
            # Sort by timestamp, most recent first
            checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return checkpoints
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints for project {project_id}: {str(e)}")
            return []
    
    def cleanup_old_checkpoints(self, project_id: str, keep_count: int = 10) -> int:
        """
        Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            project_id: Project identifier
            keep_count: Number of recent checkpoints to keep
            
        Returns:
            Number of checkpoints removed
        """
        try:
            checkpoints = self.list_checkpoints(project_id)
            
            if len(checkpoints) <= keep_count:
                return 0
            
            # Remove old checkpoints
            removed_count = 0
            for checkpoint in checkpoints[keep_count:]:
                try:
                    os.remove(checkpoint['file_path'])
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint['file_path']}: {str(e)}")
            
            logger.info(f"Cleaned up {removed_count} old checkpoints for project {project_id}")
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints for project {project_id}: {str(e)}")
            return 0
    
    def get_context_stats(self, context: ContentContext) -> Dict[str, Any]:
        """
        Get statistics and summary information for a ContentContext.
        
        Args:
            context: ContentContext to analyze
            
        Returns:
            Dictionary with context statistics
        """
        try:
            stats = {
                'project_id': context.project_id,
                'content_type': context.content_type.value,
                'created_at': context._created_at.isoformat(),
                'last_modified': context._last_modified.isoformat(),
                'processing_stage': context._processing_stage,
                'video_files_count': len(context.video_files),
                'emotional_markers_count': len(context.emotional_markers),
                'visual_highlights_count': len(context.visual_highlights),
                'key_concepts_count': len(context.key_concepts),
                'content_themes_count': len(context.content_themes),
                'checkpoints_count': len(context._checkpoints),
                'total_processing_time': context.processing_metrics.total_processing_time,
                'total_cost': context.cost_tracking.total_cost,
                'memory_peak_usage': context.processing_metrics.memory_peak_usage,
                'api_calls_total': sum(context.processing_metrics.api_calls_made.values()),
                'has_trending_keywords': context.trending_keywords is not None,
                'has_competitor_insights': context.competitor_insights is not None,
                'has_engagement_predictions': context.engagement_predictions is not None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get context stats: {str(e)}")
            return {'error': str(e)}
    
    def close_context(self, project_id: str, save: bool = True) -> bool:
        """
        Close and optionally save an active context.
        
        Args:
            project_id: Project identifier
            save: Whether to save context before closing
            
        Returns:
            True if closed successfully, False otherwise
        """
        try:
            context = self._active_contexts.get(project_id)
            
            if not context:
                logger.warning(f"Context not found for closing: {project_id}")
                return False
            
            if save:
                self.save_context(context)
            
            # Remove from active contexts
            del self._active_contexts[project_id]
            
            logger.info(f"Closed context: {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close context {project_id}: {str(e)}")
            return False
    
    def list_active_contexts(self) -> List[str]:
        """
        Get list of active context project IDs.
        
        Returns:
            List of active project IDs
        """
        return list(self._active_contexts.keys())
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """
        Get storage usage statistics.
        
        Returns:
            Dictionary with storage usage information
        """
        try:
            total_size = 0
            file_count = 0
            checkpoint_count = 0
            
            # Count context files
            for context_file in self.storage_path.glob("*.json"):
                total_size += context_file.stat().st_size
                file_count += 1
            
            # Count checkpoint files
            checkpoints_dir = self.storage_path / "checkpoints"
            if checkpoints_dir.exists():
                for checkpoint_file in checkpoints_dir.rglob("*.json"):
                    total_size += checkpoint_file.stat().st_size
                    checkpoint_count += 1
            
            return {
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'context_files': file_count,
                'checkpoint_files': checkpoint_count,
                'storage_path': str(self.storage_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage usage: {str(e)}")
            return {'error': str(e)}