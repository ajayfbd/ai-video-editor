"""
DataValidator - Validates ContentContext integrity across modules.

This module provides comprehensive validation for ContentContext objects
to ensure data integrity and consistency across all processing modules.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import os
import re

from .content_context import (
    ContentContext, EmotionalPeak, VisualHighlight, TrendingKeywords,
    ProcessingMetrics, CostMetrics, ContentType
)
from .exceptions import ContextIntegrityError


logger = logging.getLogger(__name__)


class ValidationRule:
    """Base class for validation rules."""
    
    def __init__(self, name: str, severity: str = "error"):
        """
        Initialize validation rule.
        
        Args:
            name: Rule name
            severity: Rule severity ("error", "warning", "info")
        """
        self.name = name
        self.severity = severity
    
    def validate(self, context: ContentContext) -> Tuple[bool, str]:
        """
        Validate rule against context.
        
        Args:
            context: ContentContext to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        raise NotImplementedError("Subclasses must implement validate method")


class RequiredFieldsRule(ValidationRule):
    """Validates that required fields are present and valid."""
    
    def __init__(self):
        super().__init__("required_fields", "error")
    
    def validate(self, context: ContentContext) -> Tuple[bool, str]:
        """Validate required fields."""
        issues = []
        
        if not context.project_id:
            issues.append("project_id is required")
        
        if not context.video_files:
            issues.append("video_files list cannot be empty")
        
        if not isinstance(context.content_type, ContentType):
            issues.append("content_type must be a valid ContentType enum")
        
        if not context.user_preferences:
            issues.append("user_preferences is required")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "All required fields are present"


class FileExistenceRule(ValidationRule):
    """Validates that referenced files exist."""
    
    def __init__(self):
        super().__init__("file_existence", "warning")
    
    def validate(self, context: ContentContext) -> Tuple[bool, str]:
        """Validate file existence."""
        missing_files = []
        
        for video_file in context.video_files:
            if not os.path.exists(video_file):
                missing_files.append(video_file)
        
        if missing_files:
            return False, f"Missing video files: {missing_files}"
        
        return True, "All referenced files exist"


class EmotionalMarkersRule(ValidationRule):
    """Validates emotional markers data integrity."""
    
    def __init__(self):
        super().__init__("emotional_markers", "error")
    
    def validate(self, context: ContentContext) -> Tuple[bool, str]:
        """Validate emotional markers."""
        issues = []
        
        for i, marker in enumerate(context.emotional_markers):
            if not isinstance(marker, EmotionalPeak):
                issues.append(f"Marker {i}: not an EmotionalPeak instance")
                continue
            
            if marker.timestamp < 0:
                issues.append(f"Marker {i}: negative timestamp {marker.timestamp}")
            
            if not (0.0 <= marker.intensity <= 1.0):
                issues.append(f"Marker {i}: intensity {marker.intensity} not in range [0.0, 1.0]")
            
            if not (0.0 <= marker.confidence <= 1.0):
                issues.append(f"Marker {i}: confidence {marker.confidence} not in range [0.0, 1.0]")
            
            if not marker.emotion or not isinstance(marker.emotion, str):
                issues.append(f"Marker {i}: emotion must be a non-empty string")
            
            if not marker.context or not isinstance(marker.context, str):
                issues.append(f"Marker {i}: context must be a non-empty string")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, f"All {len(context.emotional_markers)} emotional markers are valid"


class VisualHighlightsRule(ValidationRule):
    """Validates visual highlights data integrity."""
    
    def __init__(self):
        super().__init__("visual_highlights", "error")
    
    def validate(self, context: ContentContext) -> Tuple[bool, str]:
        """Validate visual highlights."""
        issues = []
        
        for i, highlight in enumerate(context.visual_highlights):
            if not isinstance(highlight, VisualHighlight):
                issues.append(f"Highlight {i}: not a VisualHighlight instance")
                continue
            
            if highlight.timestamp < 0:
                issues.append(f"Highlight {i}: negative timestamp {highlight.timestamp}")
            
            if not (0.0 <= highlight.thumbnail_potential <= 1.0):
                issues.append(f"Highlight {i}: thumbnail_potential {highlight.thumbnail_potential} not in range [0.0, 1.0]")
            
            if not highlight.description or not isinstance(highlight.description, str):
                issues.append(f"Highlight {i}: description must be a non-empty string")
            
            if not isinstance(highlight.visual_elements, list):
                issues.append(f"Highlight {i}: visual_elements must be a list")
            
            if not isinstance(highlight.faces, list):
                issues.append(f"Highlight {i}: faces must be a list")
            
            # Validate face detections
            for j, face in enumerate(highlight.faces):
                if not hasattr(face, 'bbox') or not isinstance(face.bbox, list) or len(face.bbox) != 4:
                    issues.append(f"Highlight {i}, Face {j}: bbox must be a list of 4 numbers")
                
                if not hasattr(face, 'confidence') or not (0.0 <= face.confidence <= 1.0):
                    issues.append(f"Highlight {i}, Face {j}: confidence must be in range [0.0, 1.0]")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, f"All {len(context.visual_highlights)} visual highlights are valid"


class TrendingKeywordsRule(ValidationRule):
    """Validates trending keywords data integrity."""
    
    def __init__(self):
        super().__init__("trending_keywords", "warning")
    
    def validate(self, context: ContentContext) -> Tuple[bool, str]:
        """Validate trending keywords."""
        if not context.trending_keywords:
            return True, "No trending keywords to validate"
        
        keywords = context.trending_keywords
        issues = []
        
        if not isinstance(keywords, TrendingKeywords):
            return False, "trending_keywords must be a TrendingKeywords instance"
        
        # Validate keyword lists
        for field_name in ['primary_keywords', 'long_tail_keywords', 'trending_hashtags', 
                          'seasonal_keywords', 'competitor_keywords']:
            field_value = getattr(keywords, field_name)
            if not isinstance(field_value, list):
                issues.append(f"{field_name} must be a list")
            elif not all(isinstance(item, str) for item in field_value):
                issues.append(f"{field_name} must contain only strings")
        
        # Validate search volume data
        if not isinstance(keywords.search_volume_data, dict):
            issues.append("search_volume_data must be a dictionary")
        elif not all(isinstance(k, str) and isinstance(v, int) and v >= 0 
                    for k, v in keywords.search_volume_data.items()):
            issues.append("search_volume_data must map strings to non-negative integers")
        
        # Validate timestamp
        if not isinstance(keywords.research_timestamp, datetime):
            issues.append("research_timestamp must be a datetime object")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "Trending keywords data is valid"


class ProcessingMetricsRule(ValidationRule):
    """Validates processing metrics data integrity."""
    
    def __init__(self):
        super().__init__("processing_metrics", "error")
    
    def validate(self, context: ContentContext) -> Tuple[bool, str]:
        """Validate processing metrics."""
        metrics = context.processing_metrics
        issues = []
        
        if not isinstance(metrics, ProcessingMetrics):
            return False, "processing_metrics must be a ProcessingMetrics instance"
        
        # Validate timing data
        if metrics.total_processing_time < 0:
            issues.append(f"total_processing_time cannot be negative: {metrics.total_processing_time}")
        
        if not isinstance(metrics.module_processing_times, dict):
            issues.append("module_processing_times must be a dictionary")
        elif not all(isinstance(k, str) and isinstance(v, (int, float)) and v >= 0 
                    for k, v in metrics.module_processing_times.items()):
            issues.append("module_processing_times must map strings to non-negative numbers")
        
        # Validate memory usage
        if metrics.memory_peak_usage < 0:
            issues.append(f"memory_peak_usage cannot be negative: {metrics.memory_peak_usage}")
        
        # Validate API calls
        if not isinstance(metrics.api_calls_made, dict):
            issues.append("api_calls_made must be a dictionary")
        elif not all(isinstance(k, str) and isinstance(v, int) and v >= 0 
                    for k, v in metrics.api_calls_made.items()):
            issues.append("api_calls_made must map strings to non-negative integers")
        
        # Validate cache hit rate
        if not (0.0 <= metrics.cache_hit_rate <= 1.0):
            issues.append(f"cache_hit_rate must be in range [0.0, 1.0]: {metrics.cache_hit_rate}")
        
        # Validate lists
        if not isinstance(metrics.fallbacks_used, list):
            issues.append("fallbacks_used must be a list")
        elif not all(isinstance(item, str) for item in metrics.fallbacks_used):
            issues.append("fallbacks_used must contain only strings")
        
        if not isinstance(metrics.recovery_actions, list):
            issues.append("recovery_actions must be a list")
        elif not all(isinstance(item, str) for item in metrics.recovery_actions):
            issues.append("recovery_actions must contain only strings")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "Processing metrics are valid"


class CostMetricsRule(ValidationRule):
    """Validates cost metrics data integrity."""
    
    def __init__(self):
        super().__init__("cost_metrics", "error")
    
    def validate(self, context: ContentContext) -> Tuple[bool, str]:
        """Validate cost metrics."""
        costs = context.cost_tracking
        issues = []
        
        if not isinstance(costs, CostMetrics):
            return False, "cost_tracking must be a CostMetrics instance"
        
        # Validate cost values
        cost_fields = ['gemini_api_cost', 'imagen_api_cost', 'total_cost', 'optimization_savings']
        for field_name in cost_fields:
            value = getattr(costs, field_name)
            if not isinstance(value, (int, float)) or value < 0:
                issues.append(f"{field_name} must be a non-negative number: {value}")
        
        # Validate cost per asset
        if not isinstance(costs.cost_per_asset, dict):
            issues.append("cost_per_asset must be a dictionary")
        elif not all(isinstance(k, str) and isinstance(v, (int, float)) and v >= 0 
                    for k, v in costs.cost_per_asset.items()):
            issues.append("cost_per_asset must map strings to non-negative numbers")
        
        # Validate cost consistency
        calculated_total = costs.gemini_api_cost + costs.imagen_api_cost
        if abs(costs.total_cost - calculated_total) > 0.01:  # Allow small floating point differences
            issues.append(f"total_cost {costs.total_cost} doesn't match sum of individual costs {calculated_total}")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "Cost metrics are valid"


class DataConsistencyRule(ValidationRule):
    """Validates data consistency across different parts of the context."""
    
    def __init__(self):
        super().__init__("data_consistency", "warning")
    
    def validate(self, context: ContentContext) -> Tuple[bool, str]:
        """Validate data consistency."""
        issues = []
        
        # Check timestamp consistency in emotional markers
        if len(context.emotional_markers) > 1:
            timestamps = [marker.timestamp for marker in context.emotional_markers]
            if timestamps != sorted(timestamps):
                issues.append("Emotional markers are not in chronological order")
        
        # Check timestamp consistency in visual highlights
        if len(context.visual_highlights) > 1:
            timestamps = [highlight.timestamp for highlight in context.visual_highlights]
            if timestamps != sorted(timestamps):
                issues.append("Visual highlights are not in chronological order")
        
        # Check concept consistency
        if context.key_concepts and context.content_themes:
            # Warn if there's no overlap between key concepts and themes
            concepts_set = set(context.key_concepts)
            themes_set = set(context.content_themes)
            if not concepts_set.intersection(themes_set):
                issues.append("No overlap between key_concepts and content_themes - may indicate inconsistent analysis")
        
        # Check trending keywords consistency
        if context.trending_keywords and context.key_concepts:
            keywords_text = ' '.join(context.trending_keywords.primary_keywords).lower()
            concepts_text = ' '.join(context.key_concepts).lower()
            
            # Simple overlap check
            overlap_count = sum(1 for concept in context.key_concepts 
                              if concept.lower() in keywords_text)
            
            if overlap_count == 0:
                issues.append("No overlap between key_concepts and trending_keywords - may indicate inconsistent research")
        
        if issues:
            return False, "; ".join(issues)
        
        return True, "Data consistency checks passed"


class DataValidator:
    """
    Comprehensive validator for ContentContext integrity.
    
    Validates ContentContext objects to ensure data integrity and consistency
    across all processing modules.
    """
    
    def __init__(self):
        """Initialize DataValidator with validation rules."""
        self.rules = [
            RequiredFieldsRule(),
            FileExistenceRule(),
            EmotionalMarkersRule(),
            VisualHighlightsRule(),
            TrendingKeywordsRule(),
            ProcessingMetricsRule(),
            CostMetricsRule(),
            DataConsistencyRule()
        ]
        
        logger.info(f"DataValidator initialized with {len(self.rules)} validation rules")
    
    def validate(self, context: ContentContext, strict: bool = False) -> Dict[str, Any]:
        """
        Validate ContentContext against all rules.
        
        Args:
            context: ContentContext to validate
            strict: If True, warnings are treated as errors
            
        Returns:
            Validation result dictionary
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': [],
            'rules_passed': 0,
            'rules_failed': 0,
            'score': 1.0
        }
        
        try:
            for rule in self.rules:
                is_valid, message = rule.validate(context)
                
                if is_valid:
                    result['rules_passed'] += 1
                    if rule.severity == 'info':
                        result['info'].append(f"{rule.name}: {message}")
                else:
                    result['rules_failed'] += 1
                    
                    if rule.severity == 'error' or (strict and rule.severity == 'warning'):
                        result['errors'].append(f"{rule.name}: {message}")
                        result['valid'] = False
                    elif rule.severity == 'warning':
                        result['warnings'].append(f"{rule.name}: {message}")
                    else:
                        result['info'].append(f"{rule.name}: {message}")
            
            # Calculate score
            total_rules = len(self.rules)
            if result['rules_failed'] == 0:
                result['score'] = 1.0
            else:
                # Reduce score based on failed rules and their severity
                error_weight = 0.5
                warning_weight = 0.1
                
                error_count = len(result['errors'])
                warning_count = len(result['warnings'])
                
                score_reduction = (error_count * error_weight + warning_count * warning_weight) / total_rules
                result['score'] = max(0.0, 1.0 - score_reduction)
            
            logger.debug(f"Validation completed for context {context.project_id}: "
                        f"valid={result['valid']}, score={result['score']:.2f}, "
                        f"passed={result['rules_passed']}, failed={result['rules_failed']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed with exception: {str(e)}")
            return {
                'valid': False,
                'errors': [f"Validation exception: {str(e)}"],
                'warnings': [],
                'info': [],
                'rules_passed': 0,
                'rules_failed': len(self.rules),
                'score': 0.0
            }
    
    def validate_field_update(self, context: ContentContext, field_name: str, 
                            new_value: Any) -> Dict[str, Any]:
        """
        Validate a specific field update before applying it.
        
        Args:
            context: ContentContext being updated
            field_name: Name of field being updated
            new_value: New value for the field
            
        Returns:
            Validation result for the field update
        """
        result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Create a copy of context with the new value for validation
            context_copy = ContentContext.from_dict(context.to_dict())
            setattr(context_copy, field_name, new_value)
            
            # Run relevant validation rules
            validation_result = self.validate(context_copy)
            
            # Filter results to only include issues related to the updated field
            field_related_errors = [error for error in validation_result['errors'] 
                                  if field_name in error.lower()]
            field_related_warnings = [warning for warning in validation_result['warnings'] 
                                    if field_name in warning.lower()]
            
            result['errors'] = field_related_errors
            result['warnings'] = field_related_warnings
            result['valid'] = len(field_related_errors) == 0
            
            return result
            
        except Exception as e:
            logger.error(f"Field validation failed for {field_name}: {str(e)}")
            return {
                'valid': False,
                'errors': [f"Field validation exception: {str(e)}"],
                'warnings': []
            }
    
    def get_validation_summary(self, context: ContentContext) -> str:
        """
        Get a human-readable validation summary.
        
        Args:
            context: ContentContext to validate
            
        Returns:
            Human-readable validation summary
        """
        result = self.validate(context)
        
        summary_lines = [
            f"Validation Summary for Context {context.project_id}:",
            f"  Overall Status: {'VALID' if result['valid'] else 'INVALID'}",
            f"  Score: {result['score']:.2f}/1.00",
            f"  Rules Passed: {result['rules_passed']}/{len(self.rules)}",
            f"  Rules Failed: {result['rules_failed']}/{len(self.rules)}"
        ]
        
        if result['errors']:
            summary_lines.append("  Errors:")
            for error in result['errors']:
                summary_lines.append(f"    - {error}")
        
        if result['warnings']:
            summary_lines.append("  Warnings:")
            for warning in result['warnings']:
                summary_lines.append(f"    - {warning}")
        
        if result['info']:
            summary_lines.append("  Info:")
            for info in result['info']:
                summary_lines.append(f"    - {info}")
        
        return "\n".join(summary_lines)
    
    def add_custom_rule(self, rule: ValidationRule):
        """
        Add a custom validation rule.
        
        Args:
            rule: Custom validation rule to add
        """
        self.rules.append(rule)
        logger.info(f"Added custom validation rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove a validation rule by name.
        
        Args:
            rule_name: Name of rule to remove
            
        Returns:
            True if rule was removed, False if not found
        """
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                logger.info(f"Removed validation rule: {rule_name}")
                return True
        
        logger.warning(f"Validation rule not found: {rule_name}")
        return False