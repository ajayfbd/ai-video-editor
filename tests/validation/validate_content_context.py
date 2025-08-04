# tests/validation/validate_content_context.py
"""
ContentContext integrity validation for pre-commit hooks.
Ensures ContentContext structure and behavior remain consistent.
"""

import sys
import traceback
from typing import List, Dict, Any
from dataclasses import fields, is_dataclass

from ai_video_editor.core.content_context import (
    ContentContext, EmotionalPeak, VisualHighlight, TrendingKeywords,
    ProcessingMetrics, CostMetrics, ContentType, UserPreferences, FaceDetection
)


class ContentContextValidator:
    """Validates ContentContext integrity and structure."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_all(self) -> bool:
        """Run all ContentContext validations."""
        print("🔍 Validating ContentContext integrity...")
        
        try:
            self.validate_dataclass_structure()
            self.validate_field_types()
            self.validate_default_values()
            self.validate_method_signatures()
            self.validate_serialization()
            self.validate_integration_points()
            
            return self.report_results()
            
        except Exception as e:
            self.errors.append(f"Validation failed with exception: {str(e)}")
            print(f"❌ Validation failed: {e}")
            traceback.print_exc()
            return False
    
    def validate_dataclass_structure(self):
        """Validate that ContentContext maintains proper dataclass structure."""
        if not is_dataclass(ContentContext):
            self.errors.append("ContentContext is not a dataclass")
            return
        
        # Check required fields exist
        required_fields = {
            'project_id', 'video_files', 'content_type', 'user_preferences',
            'audio_transcript', 'emotional_markers', 'key_concepts', 'visual_highlights',
            'content_themes', 'trending_keywords', 'competitor_insights', 'engagement_predictions',
            'thumbnail_concepts', 'generated_thumbnails', 'metadata_variations',
            'processed_video', 'processing_metrics', 'cost_tracking'
        }
        
        context_fields = {field.name for field in fields(ContentContext)}
        missing_fields = required_fields - context_fields
        extra_fields = context_fields - required_fields
        
        if missing_fields:
            self.errors.append(f"ContentContext missing required fields: {missing_fields}")
        
        if extra_fields:
            self.warnings.append(f"ContentContext has unexpected fields: {extra_fields}")
        
        print(f"✅ ContentContext has {len(context_fields)} fields")
    
    def validate_field_types(self):
        """Validate field type annotations."""
        context_fields = {field.name: field.type for field in fields(ContentContext)}
        
        # Check critical field types
        type_checks = {
            'project_id': str,
            'video_files': List[str],
            'content_type': ContentType,
            'emotional_markers': List[EmotionalPeak],
            'visual_highlights': List[VisualHighlight],
            'key_concepts': List[str],
            'content_themes': List[str]
        }
        
        for field_name, expected_type in type_checks.items():
            if field_name not in context_fields:
                continue
                
            actual_type = context_fields[field_name]
            # Note: This is a simplified type check - in practice, you'd need more sophisticated type checking
            if hasattr(expected_type, '__origin__'):
                # Handle generic types like List[str]
                if not str(actual_type).startswith(str(expected_type.__origin__)):
                    self.warnings.append(f"Field {field_name} type mismatch: expected {expected_type}, got {actual_type}")
        
        print("✅ Field types validated")
    
    def validate_default_values(self):
        """Validate default values for optional fields."""
        try:
            # Test creating ContentContext with minimal required fields
            context = ContentContext(
                project_id="test",
                video_files=["test.mp4"],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences()
            )
            
            # Check that list fields have proper defaults
            if not isinstance(context.emotional_markers, list):
                self.errors.append("emotional_markers should default to empty list")
            
            if not isinstance(context.visual_highlights, list):
                self.errors.append("visual_highlights should default to empty list")
            
            if not isinstance(context.key_concepts, list):
                self.errors.append("key_concepts should default to empty list")
            
            print("✅ Default values validated")
            
        except Exception as e:
            self.errors.append(f"Failed to create ContentContext with defaults: {str(e)}")
    
    def validate_method_signatures(self):
        """Validate that ContentContext methods have expected signatures."""
        required_methods = [
            'add_emotional_marker',
            'add_visual_highlight',
            'get_synchronized_concepts'
        ]
        
        for method_name in required_methods:
            if not hasattr(ContentContext, method_name):
                self.errors.append(f"ContentContext missing method: {method_name}")
            else:
                method = getattr(ContentContext, method_name)
                if not callable(method):
                    self.errors.append(f"ContentContext.{method_name} is not callable")
        
        print("✅ Method signatures validated")
    
    def validate_serialization(self):
        """Validate that ContentContext can be serialized and deserialized."""
        try:
            from dataclasses import asdict, fields
            import json
            
            # Create a sample context
            context = ContentContext(
                project_id="serialization_test",
                video_files=["test.mp4"],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences()
            )
            
            # Test serialization
            context_dict = asdict(context)
            
            # Test JSON serialization (with custom handling for enums)
            def json_serializer(obj):
                if hasattr(obj, 'value'):  # Handle enums
                    return obj.value
                return str(obj)
            
            json_str = json.dumps(context_dict, default=json_serializer)
            
            # Test deserialization
            loaded_dict = json.loads(json_str)
            
            print("✅ Serialization validated")
            
        except Exception as e:
            self.errors.append(f"Serialization validation failed: {str(e)}")
    
    def validate_integration_points(self):
        """Validate integration points with other modules."""
        try:
            # Test that ContentContext can be imported by other modules
            from ai_video_editor.core.context_manager import ContextManager
            from ai_video_editor.core.data_validator import DataValidator
            
            # Test basic integration
            context = ContentContext(
                project_id="integration_test",
                video_files=["test.mp4"],
                content_type=ContentType.EDUCATIONAL,
                user_preferences=UserPreferences()
            )
            
            # Test with ContextManager
            context_manager = ContextManager()
            validation_result = context_manager.validate_context(context)
            
            if not isinstance(validation_result, dict) or 'valid' not in validation_result:
                self.errors.append("ContextManager.validate_context should return dict with 'valid' key")
            elif not validation_result.get('valid', False):
                self.warnings.append(f"Context validation issues: {validation_result.get('issues', [])}")
            
            print("✅ Integration points validated")
            
        except ImportError as e:
            self.errors.append(f"Integration validation failed - import error: {str(e)}")
        except Exception as e:
            self.errors.append(f"Integration validation failed: {str(e)}")
    
    def report_results(self) -> bool:
        """Report validation results."""
        print("\n" + "="*60)
        print("CONTENTCONTEXT VALIDATION RESULTS")
        print("="*60)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ All validations passed!")
        elif not self.errors:
            print(f"\n✅ Validation passed with {len(self.warnings)} warnings")
        
        print("="*60)
        
        return len(self.errors) == 0


def main():
    """Main validation function for pre-commit hook."""
    validator = ContentContextValidator()
    success = validator.validate_all()
    
    if not success:
        print("\n💥 ContentContext validation failed!")
        print("Please fix the errors above before committing.")
        sys.exit(1)
    else:
        print("\n🎉 ContentContext validation passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()