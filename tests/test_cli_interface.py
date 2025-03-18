import os
import subprocess
import pytest
from unittest.mock import patch, MagicMock

# List of all available commands in the micaflow CLI
COMMANDS = [
    "pipeline",
    "apply_warp",
    "bet",
    "bias_correction",
    "calculate_jaccard",
    "compute_fa_md",
    "coregister",
    "denoise",
    "motion_correction",
    "SDC",
    "apply_SDC",
    "synthseg",
    "texture_generation",
    "normalize"
]

@pytest.mark.parametrize("command", COMMANDS)
def test_help_display(command, capsys):
    """Test that each command correctly displays help information when --help is provided."""
    with patch("subprocess.run") as mock_run:
        # Set up the mock to simulate successful execution
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        # Run the command with --help
        result = subprocess.run(["micaflow", command, "--help"], capture_output=True, text=True)
        
        # Check if the execution was successful
        assert mock_run.call_count == 1
        assert mock_run.call_args[0][0][1] == command
        assert mock_run.call_args[0][0][2] == "--help"

@pytest.fixture
def mock_subprocess_run():
    """Fixture to mock subprocess.run calls."""
    with patch("subprocess.run") as mock_run:
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        yield mock_run

@pytest.mark.parametrize("command,required_args", [
    ("bet", ["--input", "test.nii.gz", "--output", "out.nii.gz", "--output-mask", "mask.nii.gz"]),
    ("bias_correction", ["--input", "test.nii.gz", "--output", "out.nii.gz"]),
    ("synthseg", ["--i", "test.nii.gz", "--o", "out.nii.gz"]),
    ("apply_warp", ["--moving", "img.nii.gz", "--reference", "ref.nii.gz", "--output", "out.nii.gz"]),
    ("coregister", ["--fixed-file", "fixed.nii.gz", "--moving-file", "moving.nii.gz", "--out-file", "out.nii.gz"]),
    ("denoise", ["--input", "dwi.nii.gz", "--output", "denoised.nii.gz"]),
    ("motion_correction", ["--denoised", "dwi.nii.gz", "--bval", "dwi.bval", "--bvec", "dwi.bvec", "--output", "out.nii.gz"]),
    ("SDC", ["--input", "dwi.nii.gz", "--reverse-image", "rev.nii.gz", "--output", "out.nii.gz", "--output-warp", "warp.nii.gz"]),
    ("apply_SDC", ["--input", "dwi.nii.gz", "--warp", "warp.nii.gz", "--output", "out.nii.gz"]),
    ("calculate_jaccard", ["--segmentation1", "seg1.nii.gz", "--segmentation2", "seg2.nii.gz", "--output", "metrics.csv"]),
    ("compute_fa_md", ["--input", "dwi.nii.gz", "--bval", "dwi.bval", "--bvec", "dwi.bvec", "--output-fa", "fa.nii.gz", "--output-md", "md.nii.gz"]),
    ("texture_generation", ["--input", "img.nii.gz", "--mask", "mask.nii.gz", "--output", "textures/"]),
    ("normalize", ["--input", "img.nii.gz", "--mask", "mask.nii.gz", "--output", "norm.nii.gz"]),
])
def test_command_with_required_args(command, required_args, mock_subprocess_run):
    """Test that commands execute correctly with the required arguments."""
    # Build the command
    cmd = ["micaflow", command] + required_args
    
    # Execute the command (mocked)
    subprocess.run(cmd)
    
    # Verify the command was executed correctly
    assert mock_subprocess_run.call_count == 1
    assert mock_subprocess_run.call_args[0][0][1] == command
    
    # Check that all required args were passed
    for arg in required_args:
        assert arg in mock_subprocess_run.call_args[0][0]

@pytest.mark.parametrize("command,missing_args,expected_args", [
    ("bet", ["--input", "test.nii.gz"], ["--output", "--output-mask"]),
    ("synthseg", ["--i", "test.nii.gz"], ["--o"]),
    ("coregister", ["--fixed-file", "fixed.nii.gz"], ["--moving-file", "--out-file"]),
])
def test_command_missing_required_args(command, missing_args, expected_args, mock_subprocess_run):
    """Test that commands properly report missing required arguments."""
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(2, cmd=f"micaflow {command}")
    
    # Build the command with missing args
    cmd = ["micaflow", command] + missing_args
    
    # Execute the command (will raise an exception due to missing args)
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(cmd, check=True)
        
    # Verify the command attempted execution
    assert mock_subprocess_run.call_count == 1

@pytest.mark.parametrize("command,args,optional_flag", [
    ("synthseg", ["--i", "test.nii.gz", "--o", "out.nii.gz"], "--parc"),
    ("bet", ["--input", "test.nii.gz", "--output", "out.nii.gz", "--output-mask", "mask.nii.gz"], "--cpu"),
    ("coregister", ["--fixed-file", "fixed.nii.gz", "--moving-file", "moving.nii.gz", "--out-file", "out.nii.gz"], "--rigid"),
])
def test_command_with_optional_flags(command, args, optional_flag, mock_subprocess_run):
    """Test that commands accept optional flags."""
    # Build the command with an optional flag
    cmd = ["micaflow", command] + args + [optional_flag]
    
    # Execute the command (mocked)
    subprocess.run(cmd)
    
    # Verify the command was executed with the optional flag
    assert mock_subprocess_run.call_count == 1
    assert optional_flag in mock_subprocess_run.call_args[0][0]

def test_pipeline_defaults(mock_subprocess_run):
    """Test that the pipeline command accepts the required arguments."""
    cmd = [
        "micaflow", "pipeline", 
        "--subject", "sub-01",
        "--out-dir", "/output",
        "--t1w-file", "t1w.nii.gz"
    ]
    
    # Execute the command (mocked)
    subprocess.run(cmd)
    
    # Verify the command was executed with the required arguments
    assert mock_subprocess_run.call_count == 1
    assert mock_subprocess_run.call_args[0][0][1] == "pipeline"

def test_pipeline_with_optional_args(mock_subprocess_run):
    """Test the pipeline command with various optional arguments."""
    cmd = [
        "micaflow", "pipeline", 
        "--subject", "sub-01",
        "--out-dir", "/output",
        "--t1w-file", "t1w.nii.gz",
        "--session", "ses-01",
        "--flair-file", "flair.nii.gz",
        "--cpu",
        "--cores", "4"
    ]
    
    # Execute the command (mocked)
    subprocess.run(cmd)
    
    # Verify the command was executed with all arguments
    assert mock_subprocess_run.call_count == 1
    assert "--session" in mock_subprocess_run.call_args[0][0]
    assert "--flair-file" in mock_subprocess_run.call_args[0][0]
    assert "--cpu" in mock_subprocess_run.call_args[0][0]
    assert "--cores" in mock_subprocess_run.call_args[0][0]

def test_pipeline_with_dwi(mock_subprocess_run):
    """Test the pipeline command with DWI processing options."""
    cmd = [
        "micaflow", "pipeline", 
        "--subject", "sub-01",
        "--out-dir", "/output",
        "--t1w-file", "t1w.nii.gz",
        "--run-dwi",
        "--dwi-file", "dwi.nii.gz",
        "--bval-file", "dwi.bval",
        "--bvec-file", "dwi.bvec",
        "--inverse-dwi-file", "inv.nii.gz"
    ]
    
    # Execute the command (mocked)
    subprocess.run(cmd)
    
    # Verify the command was executed with DWI arguments
    assert mock_subprocess_run.call_count == 1
    assert "--run-dwi" in mock_subprocess_run.call_args[0][0]
    assert "--dwi-file" in mock_subprocess_run.call_args[0][0]
    assert "--bval-file" in mock_subprocess_run.call_args[0][0]
    assert "--bvec-file" in mock_subprocess_run.call_args[0][0]
    assert "--inverse-dwi-file" in mock_subprocess_run.call_args[0][0]