# tests/data/test_data_manager.py
"""
Test data management utilities for AI Video Editor testing.
Provides utilities for creating, managing, and validating test data.
"""

import os
import json
import tempfile
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from .sample_data import get_sample_data


class TestDataManager:
    """
    Manages test data files, expected outputs, and validation data.
    Provides utilities for creating consistent test environments.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize test data manager.
        
        Args:
            base_dir: Base directory for test data. If None, uses temporary directory.
        """
        if base_dir is None:
            self.base_dir = tempfile.mkdtemp(prefix="ai_video_editor_test_data_")
            self._cleanup_on_exit = True
        else:
            self.base_dir = base_dir
            self._cleanup_on_exit = False
        
        self.sample_files_dir = os.path.join(self.base_dir, "sample_files")
        self.expected_outputs_dir = os.path.join(self.base_dir, "expected_outputs")
        self.test_results_dir = os.path.join(self.base_dir, "test_results")
        
        # Create directories
        os.makedirs(self.sample_files_dir, exist_ok=True)
        os.makedirs(self.expected_outputs_dir, exist_ok=True)
        os.makedirs(self.test_results_dir, exist_ok=True)
        
        self._created_files = []
        self._test_sessions = {}
    
    def create_sample_video_file(self, filename: str, content_type: str = "educational") -> str:
        """
        Create a mock video file with realistic properties.
        
        Args:
            filename: Name of the video file to create
            content_type: Type of content ('educational', 'music', 'general')
        
        Returns:
            Path to the created video file
        """
        properties = get_sample_data("video_properties", content_type)
        file_path = os.path.join(self.sample_files_dir, filename)
        
        # Create mock video file with metadata
        mock_video_data = self._generate_mock_video_data(properties)
        
        with open(file_path, 'wb') as f:
            f.write(mock_video_data)
        
        # Create metadata file
        metadata_path = file_path + ".metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(properties, f, indent=2)
        
        self._created_files.extend([file_path, metadata_path])
        return file_path
    
    def create_sample_audio_file(self, filename: str, content_type: str = "educational") -> str:
        """
        Create a mock audio file with transcript data.
        
        Args:
            filename: Name of the audio file to create
            content_type: Type of content ('educational', 'music', 'general')
        
        Returns:
            Path to the created audio file
        """
        transcript = get_sample_data("transcript", content_type)
        file_path = os.path.join(self.sample_files_dir, filename)
        
        # Create mock audio file
        mock_audio_data = self._generate_mock_audio_data(transcript)
        
        with open(file_path, 'wb') as f:
            f.write(mock_audio_data)
        
        # Create transcript file
        transcript_path = file_path + ".transcript.json"
        with open(transcript_path, 'w') as f:
            json.dump(transcript, f, indent=2)
        
        self._created_files.extend([file_path, transcript_path])
        return file_path
    
    def create_expected_output(self, test_name: str, output_type: str, data: Any) -> str:
        """
        Create an expected output file for comparison testing.
        
        Args:
            test_name: Name of the test
            output_type: Type of output ('thumbnails', 'metadata', 'context', etc.)
            data: Expected output data
        
        Returns:
            Path to the created expected output file
        """
        filename = f"{test_name}_{output_type}_expected.json"
        file_path = os.path.join(self.expected_outputs_dir, filename)
        
        # Serialize data with proper handling of complex objects
        serializable_data = self._make_serializable(data)
        
        with open(file_path, 'w') as f:
            json.dump({
                "test_name": test_name,
                "output_type": output_type,
                "created_at": datetime.now().isoformat(),
                "data": serializable_data
            }, f, indent=2)
        
        self._created_files.append(file_path)
        return file_path
    
    def load_expected_output(self, test_name: str, output_type: str) -> Optional[Any]:
        """
        Load expected output for comparison.
        
        Args:
            test_name: Name of the test
            output_type: Type of output
        
        Returns:
            Expected output data or None if not found
        """
        filename = f"{test_name}_{output_type}_expected.json"
        file_path = os.path.join(self.expected_outputs_dir, filename)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
                return content.get("data")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading expected output {file_path}: {e}")
            return None
    
    def save_test_result(self, test_name: str, result_type: str, data: Any) -> str:
        """
        Save test result for analysis and debugging.
        
        Args:
            test_name: Name of the test
            result_type: Type of result ('actual_output', 'performance_metrics', etc.)
            data: Result data
        
        Returns:
            Path to the saved result file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_name}_{result_type}_{timestamp}.json"
        file_path = os.path.join(self.test_results_dir, filename)
        
        serializable_data = self._make_serializable(data)
        
        with open(file_path, 'w') as f:
            json.dump({
                "test_name": test_name,
                "result_type": result_type,
                "timestamp": timestamp,
                "data": serializable_data
            }, f, indent=2)
        
        self._created_files.append(file_path)
        return file_path
    
    def compare_outputs(self, test_name: str, output_type: str, actual_data: Any) -> Dict[str, Any]:
        """
        Compare actual output with expected output.
        
        Args:
            test_name: Name of the test
            output_type: Type of output
            actual_data: Actual output data
        
        Returns:
            Comparison result with differences and similarity score
        """
        expected_data = self.load_expected_output(test_name, output_type)
        
        if expected_data is None:
            return {
                "status": "no_expected_data",
                "message": f"No expected data found for {test_name}_{output_type}",
                "similarity_score": 0.0
            }
        
        # Save actual data for debugging
        self.save_test_result(test_name, f"{output_type}_actual", actual_data)
        
        # Perform comparison
        comparison_result = self._deep_compare(expected_data, actual_data)
        
        return {
            "status": "compared",
            "similarity_score": comparison_result["similarity_score"],
            "differences": comparison_result["differences"],
            "matching_fields": comparison_result["matching_fields"],
            "total_fields": comparison_result["total_fields"]
        }
    
    def create_test_session(self, session_name: str) -> str:
        """
        Create a test session for organizing related test data.
        
        Args:
            session_name: Name of the test session
        
        Returns:
            Session ID
        """
        session_id = f"{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = os.path.join(self.test_results_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        self._test_sessions[session_id] = {
            "name": session_name,
            "created_at": datetime.now().isoformat(),
            "directory": session_dir,
            "files": []
        }
        
        return session_id
    
    def add_to_session(self, session_id: str, file_path: str, description: str = ""):
        """
        Add a file to a test session.
        
        Args:
            session_id: ID of the test session
            file_path: Path to the file to add
            description: Optional description of the file
        """
        if session_id not in self._test_sessions:
            raise ValueError(f"Test session {session_id} not found")
        
        session = self._test_sessions[session_id]
        session["files"].append({
            "path": file_path,
            "description": description,
            "added_at": datetime.now().isoformat()
        })
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get summary of a test session.
        
        Args:
            session_id: ID of the test session
        
        Returns:
            Session summary
        """
        if session_id not in self._test_sessions:
            raise ValueError(f"Test session {session_id} not found")
        
        session = self._test_sessions[session_id]
        return {
            "session_id": session_id,
            "name": session["name"],
            "created_at": session["created_at"],
            "directory": session["directory"],
            "file_count": len(session["files"]),
            "files": session["files"]
        }
    
    def cleanup(self):
        """Clean up created files and directories."""
        if self._cleanup_on_exit and os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
    
    def _generate_mock_video_data(self, properties: Dict[str, Any]) -> bytes:
        """Generate mock video data based on properties."""
        # Create a simple mock video file with header information
        header = f"MOCK_VIDEO_{properties['codec']}_{properties['resolution'][0]}x{properties['resolution'][1]}"
        data_size = min(properties.get('file_size', 1000000), 10000000)  # Cap at 10MB for tests
        
        mock_data = header.encode('utf-8')
        mock_data += b'\x00' * (data_size - len(mock_data))
        
        return mock_data
    
    def _generate_mock_audio_data(self, transcript: Dict[str, Any]) -> bytes:
        """Generate mock audio data based on transcript."""
        # Create mock audio data with transcript information
        header = f"MOCK_AUDIO_{transcript['language']}_{len(transcript['segments'])}_segments"
        data_size = len(transcript['full_text']) * 1000  # Rough estimate
        
        mock_data = header.encode('utf-8')
        mock_data += b'\x00' * (data_size - len(mock_data))
        
        return mock_data
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if hasattr(obj, '__dict__'):
            # Handle dataclass or custom objects
            if hasattr(obj, '__dataclass_fields__'):
                from dataclasses import asdict
                return asdict(obj)
            else:
                return obj.__dict__
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj
    
    def _deep_compare(self, expected: Any, actual: Any, path: str = "") -> Dict[str, Any]:
        """Perform deep comparison of two data structures."""
        differences = []
        matching_fields = 0
        total_fields = 0
        
        def compare_recursive(exp, act, current_path):
            nonlocal matching_fields, total_fields
            
            if type(exp) != type(act):
                differences.append({
                    "path": current_path,
                    "type": "type_mismatch",
                    "expected": str(type(exp)),
                    "actual": str(type(act))
                })
                total_fields += 1
                return
            
            if isinstance(exp, dict):
                all_keys = set(exp.keys()) | set(act.keys())
                for key in all_keys:
                    key_path = f"{current_path}.{key}" if current_path else key
                    
                    if key not in exp:
                        differences.append({
                            "path": key_path,
                            "type": "extra_field",
                            "actual": act[key]
                        })
                        total_fields += 1
                    elif key not in act:
                        differences.append({
                            "path": key_path,
                            "type": "missing_field",
                            "expected": exp[key]
                        })
                        total_fields += 1
                    else:
                        compare_recursive(exp[key], act[key], key_path)
            
            elif isinstance(exp, (list, tuple)):
                max_len = max(len(exp), len(act))
                for i in range(max_len):
                    item_path = f"{current_path}[{i}]"
                    
                    if i >= len(exp):
                        differences.append({
                            "path": item_path,
                            "type": "extra_item",
                            "actual": act[i]
                        })
                        total_fields += 1
                    elif i >= len(act):
                        differences.append({
                            "path": item_path,
                            "type": "missing_item",
                            "expected": exp[i]
                        })
                        total_fields += 1
                    else:
                        compare_recursive(exp[i], act[i], item_path)
            
            else:
                total_fields += 1
                if exp == act:
                    matching_fields += 1
                else:
                    differences.append({
                        "path": current_path,
                        "type": "value_mismatch",
                        "expected": exp,
                        "actual": act
                    })
        
        compare_recursive(expected, actual, path)
        
        similarity_score = matching_fields / max(total_fields, 1)
        
        return {
            "differences": differences,
            "matching_fields": matching_fields,
            "total_fields": total_fields,
            "similarity_score": similarity_score
        }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()