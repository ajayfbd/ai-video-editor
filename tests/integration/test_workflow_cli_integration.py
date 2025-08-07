"""
Integration tests for CLI workflow orchestrator integration.

Tests the integration between the CLI interface and the WorkflowOrchestrator,
ensuring proper command handling, progress display, and error reporting.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from click.testing import CliRunner

from ai_video_editor.cli.main import cli
from ai_video_editor.core.workflow_orchestrator import (
    WorkflowOrchestrator,
    WorkflowConfiguration,
    ProcessingMode
)
from ai_video_editor.core.content_context import ContentContext, ContentType
from ai_video_editor.core.config import ProjectSettings


class TestCLIWorkflowIntegration:
    """Test CLI integration with WorkflowOrchestrator."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def test_video_files(self, temp_dir):
        """Create test video files."""
        video_files = []
        for i in range(2):
            video_file = temp_dir / f"test_video_{i}.mp4"
            video_file.write_text(f"Mock video content {i}")
            video_files.append(video_file)
        return video_files
    
    def test_cli_process_command_basic(self, runner, test_video_files):
        """Test basic process command execution."""
        with patch('ai_video_editor.cli.main.asyncio.run') as mock_run:
            # Mock successful workflow execution
            mock_context = Mock(spec=ContentContext)
            mock_run.return_value = mock_context
            
            result = runner.invoke(cli, [
                'process',
                str(test_video_files[0]),
                '--type', 'educational',
                '--quality', 'high',
                '--mode', 'balanced'
            ])
            
            assert result.exit_code == 0
            assert "Starting AI Video Editor pipeline" in result.output
            assert "Content type: educational" in result.output
            assert "Quality: high" in result.output
            assert "Processing mode: balanced" in result.output
            
            # Verify asyncio.run was called
            mock_run.assert_called_once()
    
    def test_cli_process_command_with_options(self, runner, test_video_files, temp_dir):
        """Test process command with various options."""
        output_dir = temp_dir / "output"
        
        with patch('ai_video_editor.cli.main.asyncio.run') as mock_run:
            mock_context = Mock(spec=ContentContext)
            mock_run.return_value = mock_context
            
            result = runner.invoke(cli, [
                'process',
                str(test_video_files[0]),
                str(test_video_files[1]),
                '--output', str(output_dir),
                '--type', 'music',
                '--quality', 'ultra',
                '--mode', 'high_quality',
                '--sequential',
                '--max-memory', '16.0',
                '--timeout', '3600',
                '--no-progress'
            ])
            
            assert result.exit_code == 0
            assert "Content type: music" in result.output
            assert "Quality: ultra" in result.output
            assert "Processing mode: high_quality" in result.output
            
            # Verify the workflow was configured correctly
            mock_run.assert_called_once()
            
            # Get the async function that was passed to asyncio.run
            async_func = mock_run.call_args[0][0]
            
            # We can't easily inspect the closure, but we can verify the call was made
            assert callable(async_func)
    
    @patch('ai_video_editor.core.workflow_orchestrator.WorkflowOrchestrator')
    def test_cli_workflow_orchestrator_creation(self, mock_orchestrator_class, runner, test_video_files):
        """Test that CLI creates WorkflowOrchestrator with correct configuration."""
        mock_orchestrator = Mock()
        mock_orchestrator.process_video = AsyncMock(return_value=Mock(spec=ContentContext))
        mock_orchestrator.get_processing_summary.return_value = {
            'project_id': 'test_project',
            'workflow_duration': 120.0,
            'processing_metrics': {'memory_peak_usage': 4000000000, 'api_calls_made': {'gemini': 5}},
        }
        mock_orchestrator_class.return_value = mock_orchestrator
        
        with patch('ai_video_editor.cli.main.asyncio.run') as mock_run:
            # Make asyncio.run actually call the async function
            def run_async(coro):
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
            
            mock_run.side_effect = run_async
            
            result = runner.invoke(cli, [
                'process',
                str(test_video_files[0]),
                '--type', 'educational',
                '--mode', 'fast',
                '--max-memory', '8.0'
            ])
            
            assert result.exit_code == 0
            
            # Verify WorkflowOrchestrator was created
            mock_orchestrator_class.assert_called_once()
            
            # Get the configuration passed to the orchestrator
            call_args = mock_orchestrator_class.call_args
            config = call_args[1]['config']  # keyword argument
            
            assert isinstance(config, WorkflowConfiguration)
            assert config.processing_mode == ProcessingMode.FAST
            assert config.max_memory_usage_gb == 8.0
            
            # Verify process_video was called
            mock_orchestrator.process_video.assert_called_once()
            
            # Verify processing summary was displayed
            assert "Processing completed successfully" in result.output
    
    def test_cli_process_command_error_handling(self, runner, test_video_files):
        """Test CLI error handling during processing."""
        with patch('ai_video_editor.cli.main.asyncio.run') as mock_run:
            # Mock workflow failure
            mock_run.side_effect = Exception("Workflow failed")
            
            result = runner.invoke(cli, [
                'process',
                str(test_video_files[0])
            ])
            
            assert result.exit_code == 1
            assert "Processing failed" in result.output
    
    def test_cli_process_command_keyboard_interrupt(self, runner, test_video_files):
        """Test CLI handling of keyboard interrupt."""
        with patch('ai_video_editor.cli.main.asyncio.run') as mock_run:
            # Mock keyboard interrupt
            mock_run.side_effect = KeyboardInterrupt()
            
            result = runner.invoke(cli, [
                'process',
                str(test_video_files[0])
            ])
            
            assert result.exit_code == 1
            assert "Processing cancelled by user" in result.output
    
    def test_cli_process_command_missing_files(self, runner):
        """Test CLI handling of missing input files."""
        result = runner.invoke(cli, [
            'process',
            'nonexistent_file.mp4'
        ])
        
        # Click should handle the file existence check
        assert result.exit_code != 0
    
    def test_cli_workflow_command_basic(self, runner):
        """Test basic workflow command."""
        result = runner.invoke(cli, ['workflow', '--list'])
        
        assert result.exit_code == 0
        assert "Active Workflow Projects" in result.output
    
    def test_cli_workflow_command_with_project_id(self, runner):
        """Test workflow command with project ID."""
        result = runner.invoke(cli, ['workflow', 'test_project_123'])
        
        assert result.exit_code == 0
        assert "Workflow Status for Project: test_project_123" in result.output
    
    def test_cli_workflow_command_no_args(self, runner):
        """Test workflow command without arguments."""
        result = runner.invoke(cli, ['workflow'])
        
        assert result.exit_code == 0
        assert "Please specify a project ID" in result.output
        assert "Usage:" in result.output
    
    def test_cli_test_workflow_command(self, runner):
        """Test workflow testing command."""
        with patch('ai_video_editor.core.workflow_orchestrator.WorkflowOrchestrator') as mock_class:
            mock_orchestrator = Mock()
            mock_class.return_value = mock_orchestrator
            
            result = runner.invoke(cli, ['test-workflow'])
            
            assert result.exit_code == 0
            assert "Testing Workflow Orchestrator" in result.output
            assert "Workflow orchestrator created successfully" in result.output
            assert "test completed successfully" in result.output
            
            # Verify orchestrator was created
            mock_class.assert_called_once()
    
    def test_cli_test_workflow_command_with_options(self, runner):
        """Test workflow testing command with options."""
        with patch('ai_video_editor.core.workflow_orchestrator.WorkflowOrchestrator') as mock_class:
            mock_orchestrator = Mock()
            mock_class.return_value = mock_orchestrator
            
            result = runner.invoke(cli, [
                'test-workflow',
                '--stage', 'audio_processing',
                '--mock'
            ])
            
            assert result.exit_code == 0
            assert "Testing Workflow Orchestrator" in result.output
            assert "Stage-specific testing not yet implemented" in result.output
            assert "Mock data testing not yet implemented" in result.output
    
    def test_cli_test_workflow_command_error(self, runner):
        """Test workflow testing command error handling."""
        with patch('ai_video_editor.core.workflow_orchestrator.WorkflowOrchestrator') as mock_class:
            mock_class.side_effect = Exception("Orchestrator creation failed")
            
            result = runner.invoke(cli, ['test-workflow'])
            
            assert result.exit_code == 1
            assert "Workflow test failed" in result.output
    
    @patch('ai_video_editor.cli.main.get_settings')
    def test_cli_configuration_integration(self, mock_get_settings, runner, test_video_files):
        """Test CLI integration with configuration system."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.max_memory_usage_gb = 16.0
        mock_settings.max_concurrent_processes = 4
        mock_get_settings.return_value = mock_settings
        
        with patch('ai_video_editor.cli.main.asyncio.run') as mock_run:
            mock_context = Mock(spec=ContentContext)
            mock_run.return_value = mock_context
            
            result = runner.invoke(cli, [
                'process',
                str(test_video_files[0])
            ])
            
            assert result.exit_code == 0
            
            # Verify settings were loaded
            mock_get_settings.assert_called()
    
    def test_cli_progress_display_integration(self, runner, test_video_files):
        """Test CLI progress display integration."""
        with patch('ai_video_editor.core.workflow_orchestrator.WorkflowOrchestrator') as mock_class:
            mock_orchestrator = Mock()
            mock_orchestrator.process_video = AsyncMock(return_value=Mock(spec=ContentContext))
            mock_orchestrator.get_processing_summary.return_value = {}
            mock_class.return_value = mock_orchestrator
            
            with patch('ai_video_editor.cli.main.asyncio.run') as mock_run:
                def run_async(coro):
                    loop = asyncio.new_event_loop()
                    try:
                        return loop.run_until_complete(coro)
                    finally:
                        loop.close()
                
                mock_run.side_effect = run_async
                
                # Test with progress enabled (default)
                result = runner.invoke(cli, [
                    'process',
                    str(test_video_files[0])
                ])
                
                assert result.exit_code == 0
                
                # Verify orchestrator was configured with progress enabled
                config = mock_class.call_args[1]['config']
                assert config.enable_progress_display is True
                
                # Test with progress disabled
                result = runner.invoke(cli, [
                    'process',
                    str(test_video_files[0]),
                    '--no-progress'
                ])
                
                assert result.exit_code == 0
                
                # Verify orchestrator was configured with progress disabled
                config = mock_class.call_args[1]['config']
                assert config.enable_progress_display is False


class TestCLIWorkflowConfigurationMapping:
    """Test mapping between CLI options and workflow configuration."""
    
    def test_processing_mode_mapping(self):
        """Test processing mode mapping from CLI to configuration."""
        mode_mappings = {
            'fast': ProcessingMode.FAST,
            'balanced': ProcessingMode.BALANCED,
            'high_quality': ProcessingMode.HIGH_QUALITY
        }
        
        for cli_mode, expected_mode in mode_mappings.items():
            assert ProcessingMode(cli_mode) == expected_mode
    
    def test_content_type_mapping(self):
        """Test content type mapping from CLI to configuration."""
        from ai_video_editor.core.config import ContentType
        
        type_mappings = {
            'educational': ContentType.EDUCATIONAL,
            'music': ContentType.MUSIC,
            'general': ContentType.GENERAL
        }
        
        for cli_type, expected_type in type_mappings.items():
            assert ContentType(cli_type) == expected_type
    
    def test_quality_mapping(self):
        """Test quality mapping from CLI to configuration."""
        from ai_video_editor.core.config import VideoQuality
        
        quality_mappings = {
            'low': VideoQuality.LOW,
            'medium': VideoQuality.MEDIUM,
            'high': VideoQuality.HIGH,
            'ultra': VideoQuality.ULTRA
        }
        
        for cli_quality, expected_quality in quality_mappings.items():
            assert VideoQuality(cli_quality) == expected_quality


class TestCLIWorkflowErrorScenarios:
    """Test CLI error scenarios and recovery."""
    
    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def test_video_file(self, temp_dir):
        """Create test video file."""
        video_file = temp_dir / "test_video.mp4"
        video_file.write_text("Mock video content")
        return video_file
    
    def test_cli_invalid_processing_mode(self, runner, test_video_file):
        """Test CLI handling of invalid processing mode."""
        result = runner.invoke(cli, [
            'process',
            str(test_video_file),
            '--mode', 'invalid_mode'
        ])
        
        assert result.exit_code != 0
        assert "Invalid value for '--mode'" in result.output
    
    def test_cli_invalid_content_type(self, runner, test_video_file):
        """Test CLI handling of invalid content type."""
        result = runner.invoke(cli, [
            'process',
            str(test_video_file),
            '--type', 'invalid_type'
        ])
        
        assert result.exit_code != 0
        assert "Invalid value for '--type'" in result.output
    
    def test_cli_invalid_quality(self, runner, test_video_file):
        """Test CLI handling of invalid quality."""
        result = runner.invoke(cli, [
            'process',
            str(test_video_file),
            '--quality', 'invalid_quality'
        ])
        
        assert result.exit_code != 0
        assert "Invalid value for '--quality'" in result.output
    
    def test_cli_negative_memory_limit(self, runner, test_video_file):
        """Test CLI handling of negative memory limit."""
        result = runner.invoke(cli, [
            'process',
            str(test_video_file),
            '--max-memory', '-1.0'
        ])
        
        # The CLI should accept the value, but the orchestrator should handle validation
        # This tests that the CLI doesn't crash on unusual values
        with patch('ai_video_editor.cli.main.asyncio.run') as mock_run:
            mock_run.side_effect = Exception("Expected validation error")
            
            result = runner.invoke(cli, [
                'process',
                str(test_video_file),
                '--max-memory', '-1.0'
            ])
            
            assert result.exit_code == 1
    
    def test_cli_zero_timeout(self, runner, test_video_file):
        """Test CLI handling of zero timeout."""
        with patch('ai_video_editor.cli.main.asyncio.run') as mock_run:
            mock_run.side_effect = Exception("Timeout too short")
            
            result = runner.invoke(cli, [
                'process',
                str(test_video_file),
                '--timeout', '0'
            ])
            
            assert result.exit_code == 1


if __name__ == "__main__":
    pytest.main([__file__])