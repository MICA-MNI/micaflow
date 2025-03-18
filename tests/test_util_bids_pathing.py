import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# filepath: /home/ian/GitHub/micaflow2.0/tests/test_util_bids_pathing.py

# Import the module to test
from micaflow.scripts.util_bids_pathing import (
    print_note, 
    print_warning, 
    print_error, 
    check_paths
)


class TestPrintFunctions:
    """Test suite for the print utility functions."""
    
    def test_print_note(self, capsys):
        """Test that print_note outputs correctly formatted messages."""
        print_note("Test note")
        captured = capsys.readouterr()
        assert "[NOTE]" in captured.out
        assert "Test note" in captured.out
        
        # Test with two messages
        print_note("First message", "Second message")
        captured = capsys.readouterr()
        assert captured.out.count("[NOTE]") == 2
        assert "First message" in captured.out
        assert "Second message" in captured.out
        
    def test_print_warning(self, capsys):
        """Test that print_warning outputs correctly formatted messages."""
        print_warning("Test warning")
        captured = capsys.readouterr()
        assert "[WARNING]" in captured.out
        assert "Test warning" in captured.out
        
    def test_print_error(self, capsys):
        """Test that print_error outputs correctly formatted messages."""
        print_error("Test error")
        captured = capsys.readouterr()
        assert "[ERROR]" in captured.out
        assert "Test error" in captured.out


class TestCheckPaths:
    """Test suite for the check_paths function."""
    
    @pytest.fixture
    def mock_sys_exit(self, monkeypatch):
        """Mock sys.exit to avoid exiting during tests."""
        exit_mock = MagicMock(side_effect=Exception("sys.exit called"))
        monkeypatch.setattr(sys, "exit", exit_mock)
        return exit_mock
    
    @pytest.fixture
    def mock_torch_cuda(self, monkeypatch):
        """Mock torch.cuda.is_available."""
        mock = MagicMock(return_value=True)
        monkeypatch.setattr("torch.cuda.is_available", mock)
        return mock
        
    @pytest.fixture
    def mock_tf_gpu(self, monkeypatch):
        """Mock tf.test.is_gpu_available."""
        mock = MagicMock(return_value=True)
        monkeypatch.setattr("tensorflow.test.is_gpu_available", mock)
        return mock
    
    @pytest.fixture
    def mock_file_exists(self, monkeypatch):
        """Mock os.path.exists to return True for dummy paths in tests."""
        original_exists = os.path.exists
        
        def mock_exists(path):
            if path.startswith('/dummy/'):
                return True
            return original_exists(path)
        
        monkeypatch.setattr(os.path, "exists", mock_exists)
        
    @pytest.fixture
    def mock_create_dir(self, monkeypatch):
        """Mock os.path.exists to return True for dummy paths in tests."""
        original_exists = os.path.exists
        
        def mock_create(path):
            pass
        
        monkeypatch.setattr(os, "makedirs", mock_create)
        
    @pytest.fixture
    def temp_dir_structure(self):
        """Create a temporary directory structure for testing."""
        base_dir = tempfile.mkdtemp()
        
        # Create test subject directories
        os.makedirs(os.path.join(base_dir, "sub-01", "ses-01", "anat"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "sub-01", "ses-01", "dwi"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "sub-01", "anat"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "sub-01", "dwi"), exist_ok=True)
        
        # Create test files
        open(os.path.join(base_dir, "sub-01", "ses-01", "anat", "T1w.nii.gz"), 'w').close()
        open(os.path.join(base_dir, "sub-01", "ses-01", "anat", "FLAIR.nii.gz"), 'w').close()
        open(os.path.join(base_dir, "sub-01", "ses-01", "dwi", "dwi.nii.gz"), 'w').close()
        open(os.path.join(base_dir, "sub-01", "ses-01", "dwi", "dwi.bval"), 'w').close()
        open(os.path.join(base_dir, "sub-01", "ses-01", "dwi", "dwi.bvec"), 'w').close()
        open(os.path.join(base_dir, "sub-01", "ses-01", "dwi", "dwi_inv.nii.gz"), 'w').close()
        
        open(os.path.join(base_dir, "sub-01", "anat", "T1w.nii.gz"), 'w').close()
        open(os.path.join(base_dir, "sub-01", "dwi", "dwi.nii.gz"), 'w').close()
        open(os.path.join(base_dir, "sub-01", "dwi", "dwi.bval"), 'w').close()
        open(os.path.join(base_dir, "sub-01", "dwi", "dwi.bvec"), 'w').close()
        open(os.path.join(base_dir, "sub-01", "dwi", "dwi_inv.nii.gz"), 'w').close()
        
        yield base_dir
        
        # Clean up after test
        shutil.rmtree(base_dir)
    
    def test_empty_subject(self, mock_sys_exit, capsys):
        """Test that an error is raised when subject is empty."""
        with pytest.raises(Exception, match="sys.exit called"):
            check_paths('', '', '', '', '', '', '', '', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "Subject not provided" in captured.out
        assert mock_sys_exit.call_count == 1
    
    def test_empty_session_warning(self, mock_file_exists, capsys):
        """Test warning is shown when session is empty."""
        check_paths('', '', 'sub-01', '', '', '', '/dummy/t1w.nii.gz', '', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "Session not provided" in captured.out
    
    def test_none_session(self, mock_file_exists, capsys):
        """Test that 'None' session is converted to None."""
        result = check_paths('', '', 'sub-01', 'None', '', '', '/dummy/t1w.nii.gz', '', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "Session is set to None" in captured.out
        assert result[3] is None  # SESSION should be None
    
    def test_empty_cpu_default(self, mock_file_exists, capsys):
        """Test that CPU defaults to True when not provided."""
        result = check_paths('', '', 'sub-01', '', '', '', '/dummy/t1w.nii.gz', '', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "CPU not provided" in captured.out
        assert result[5] is True  # CPU should be True
    
    def test_empty_threads_default(self, mock_file_exists, capsys):
        """Test that THREADS defaults to 1 when not provided."""
        result = check_paths('', '', 'sub-01', '', '', '', '/dummy/t1w.nii.gz', '', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "THREADS not provided" in captured.out
        assert result[12] == 1  # THREADS should be 1
    
    def test_missing_t1w(self, mock_sys_exit, capsys):
        """Test that an error is raised when T1w is not provided."""
        with pytest.raises(Exception, match="sys.exit called"):
            check_paths('', '', 'sub-01', '', '', '', '', '', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "T1w file not provided" in captured.out
    
    def test_dwi_missing_bval(self, mock_sys_exit, capsys):
        """Test that an error is raised when DWI is provided but BVAL is not."""
        with pytest.raises(Exception, match="sys.exit called"):
            check_paths('', '', 'sub-01', '', '', '', '/dummy/t1w.nii.gz', '/dummy/dwi.nii.gz', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "BVAL file not provided" in captured.out
    
    def test_dwi_missing_bvec(self, mock_sys_exit, capsys):
        """Test that an error is raised when DWI and BVAL are provided but BVEC is not."""
        with pytest.raises(Exception, match="sys.exit called"):
            check_paths('', '', 'sub-01', '', '', '', '/dummy/t1w.nii.gz', '/dummy/dwi.nii.gz', '/dummy/dwi.bval', '', '', '')
        
        captured = capsys.readouterr()
        assert "BVEC file not provided" in captured.out
    
    def test_dwi_missing_inverse(self, mock_sys_exit, capsys):
        """Test that an error is raised when DWI, BVAL, BVEC are provided but inverse DWI is not."""
        with pytest.raises(Exception, match="sys.exit called"):
            check_paths('', '', 'sub-01', '', '', '', '/dummy/t1w.nii.gz', '/dummy/dwi.nii.gz', '/dummy/dwi.bval', '/dummy/dwi.bvec', '', '')
        
        captured = capsys.readouterr()
        assert "Inverse DWI file not provided" in captured.out
    
    def test_nonexistent_data_directory(self, mock_sys_exit, capsys):
        """Test that an error is raised when the data directory does not exist."""
        with pytest.raises(Exception, match="sys.exit called"):
            check_paths('/nonexistent/path', '', 'sub-01', '', '', '', '/dummy/t1w.nii.gz', '', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "The data directory does not exist" in captured.out
    
    def test_data_directory_with_nonexistent_subject(self, temp_dir_structure, mock_sys_exit, capsys):
        """Test that an error is raised when the subject does not exist in the data directory."""
        with pytest.raises(Exception, match="sys.exit called"):
            check_paths(temp_dir_structure, '', 'nonexistent-subject', '', '', '', 'T1w.nii.gz', '', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "Subject nonexistent-subject does not exist" in captured.out
    
    def test_data_directory_with_nonexistent_session(self, temp_dir_structure, mock_sys_exit, capsys):
        """Test that an error is raised when the session does not exist for the subject."""
        with pytest.raises(Exception, match="sys.exit called"):
            check_paths(temp_dir_structure, '', 'sub-01', 'nonexistent-session', '', '', 'T1w.nii.gz', '', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "Session nonexistent-session does not exist" in captured.out
    
    def test_data_directory_with_nonexistent_t1w(self, temp_dir_structure, mock_sys_exit, capsys):
        """Test that an error is raised when the T1w file does not exist in the data directory."""
        with pytest.raises(Exception, match="sys.exit called"):
            check_paths(temp_dir_structure, '', 'sub-01', 'ses-01', '', '', 'nonexistent-T1w.nii.gz', '', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "T1w file does not exist" in captured.out
    
    def test_data_directory_with_nonexistent_flair(self, temp_dir_structure, mock_sys_exit, capsys):
        """Test that an error is raised when the FLAIR file does not exist in the data directory."""
        with pytest.raises(Exception, match="sys.exit called"):
            check_paths(temp_dir_structure, '', 'sub-01', 'ses-01', '', 'nonexistent-FLAIR.nii.gz', 'T1w.nii.gz', '', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "FLAIR file does not exist" in captured.out
    
    def test_data_directory_with_valid_t1w(self, temp_dir_structure, mock_create_dir, capsys):
        """Test that valid T1w path is constructed correctly using data directory."""
        result = check_paths(temp_dir_structure, '/output', 'sub-01', 'ses-01', '', '', 'T1w.nii.gz', '', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "T1w file exists" in captured.out
        assert "sub-01/ses-01/anat/T1w.nii.gz" in captured.out
    
    def test_data_directory_with_valid_t1w_no_session(self, temp_dir_structure, mock_create_dir, capsys):
        """Test that valid T1w path is constructed correctly with None session."""
        result = check_paths(temp_dir_structure, '/output', 'sub-01', 'None', '', '', 'T1w.nii.gz', '', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "T1w file exists" in captured.out
        assert "sub-01/anat/T1w.nii.gz" in captured.out
    
    
    def test_gpu_check_with_cpu_true(self, mock_torch_cuda, mock_tf_gpu, mock_file_exists, capsys):
        """Test GPU check is skipped when CPU is True."""
        result = check_paths('', '', 'sub-01', '', '', '', '/dummy/t1w.nii.gz', '', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "CPU computation enabled" in captured.out
        assert result[5] is True
        
        # Ensure the GPU checks weren't called
        assert not mock_torch_cuda.called
        assert not mock_tf_gpu.called
    
    def test_gpu_check_with_cpu_false_gpu_available(self, mock_torch_cuda, mock_tf_gpu, mock_file_exists, capsys):
        """Test GPU is enabled when CPU is False and GPUs are available."""
        mock_torch_cuda.return_value = True
        mock_tf_gpu.return_value = True
        
        result = check_paths('', '', 'sub-01', '', 'False', '', '/dummy/t1w.nii.gz', '', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "GPU computation enabled from PyTorch" in captured.out
        assert "GPU computation enabled from Tensorflow" in captured.out
        assert result[5] is False  # CPU should still be False
    
    def test_gpu_check_with_cpu_false_no_pytorch_gpu(self, mock_torch_cuda, mock_tf_gpu, mock_file_exists, capsys):
        """Test CPU is set to True when PyTorch doesn't detect a GPU."""
        mock_torch_cuda.return_value = False
        mock_tf_gpu.return_value = True
        
        result = check_paths('', '', 'sub-01', '', 'False', '', '/dummy/t1w.nii.gz', '', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "GPU computation not enabled on PyTorch" in captured.out
        assert result[5] is True  # CPU should be set to True
    
    def test_gpu_check_with_cpu_false_no_tensorflow_gpu(self, mock_torch_cuda, mock_tf_gpu, mock_file_exists, capsys):
        """Test CPU is set to True when TensorFlow doesn't detect a GPU."""
        mock_torch_cuda.return_value = True
        mock_tf_gpu.return_value = False
        
        result = check_paths('', '', 'sub-01', '', 'False', '', '/dummy/t1w.nii.gz', '', '', '', '', '')
        
        captured = capsys.readouterr()
        assert "GPU computation not enabled on Tensorflow" in captured.out
        assert result[5] is True  # CPU should be set to True
    
    def test_complete_diffusion_data_with_session(self, temp_dir_structure, mock_create_dir, capsys):
        """Test fully specified diffusion data with session."""
        result = check_paths(
            temp_dir_structure, '/output', 'sub-01', 'ses-01', '',
            '', 'T1w.nii.gz', 'dwi.nii.gz', 'dwi.bval', 'dwi.bvec', 'dwi_inv.nii.gz', '4'
        )
        
        captured = capsys.readouterr()
        assert "DWI file provided" in captured.out
        assert "Enabling diffusion pipeline" in captured.out
        assert "All paths are valid" in captured.out
        
        assert result[4] is True  # RUN_DWI should be True
        assert result[8].endswith("dwi.nii.gz")  # DWI_FILE should be updated
        assert result[12] == 4  # THREADS should be 4
    
    def test_complete_diffusion_data_without_session(self, temp_dir_structure, mock_create_dir, capsys):
        """Test fully specified diffusion data without session."""
        result = check_paths(
            temp_dir_structure, '/output', 'sub-01', 'None', '',
            '', 'T1w.nii.gz', 'dwi.nii.gz', 'dwi.bval', 'dwi.bvec', 'dwi_inv.nii.gz', '4'
        )
        
        captured = capsys.readouterr()
        assert "DWI file provided" in captured.out
        assert "Session is set to None" in captured.out
        assert "All paths are valid" in captured.out
        
        assert result[4] is True  # RUN_DWI should be True
        assert result[3] is None  # SESSION should be None
        assert "sub-01/dwi/dwi.nii.gz" in result[8]  # DWI_FILE should be updated with no session path
    
    def test_absolute_paths_without_data_directory(self, mock_create_dir, capsys):
        """Test using absolute paths without a data directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            t1w_path = os.path.join(temp_dir, "t1w.nii.gz")
            dwi_path = os.path.join(temp_dir, "dwi.nii.gz")
            bval_path = os.path.join(temp_dir, "dwi.bval")
            bvec_path = os.path.join(temp_dir, "dwi.bvec")
            inv_path = os.path.join(temp_dir, "dwi_inv.nii.gz")
            
            # Create the files
            for path in [t1w_path, dwi_path, bval_path, bvec_path, inv_path]:
                with open(path, 'w') as f:
                    f.write('')
            
            result = check_paths(
                '', '/output', 'sub-01', '', '',
                '', t1w_path, dwi_path, bval_path, bvec_path, inv_path, '4'
            )
            
            captured = capsys.readouterr()
            assert "Data directory not provided" in captured.out
            assert "file paths are assumed to be absolute paths" in captured.out
            assert f"T1w file exists at path {t1w_path}" in captured.out
            
            assert result[4] is True  # RUN_DWI should be True
            assert result[7] == t1w_path  # T1W_FILE should remain unchanged


if __name__ == "__main__":
    pytest.main(["-v"])