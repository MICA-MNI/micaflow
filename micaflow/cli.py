#!/usr/bin/env python3
"""
cli - Command-Line Interface for MicaFlow MRI Processing Pipeline

This module provides the main command-line interface (CLI) for the MicaFlow
neuroimaging processing pipeline. It handles command routing, argument parsing,
and execution of both the full pipeline and individual processing modules.

Architecture:
------------
The CLI uses a two-level command structure:
1. Main command: 'micaflow [command]' routes to either the pipeline or a module
2. Module commands: Each processing module has its own subcommand with specific arguments

The CLI intercepts help requests before argparse to provide custom, color-coded help
messages for each command. Arguments are dynamically transformed and forwarded to
the appropriate processing scripts.

Available Commands:
------------------
Pipeline Command:
  pipeline          : Run the full Snakemake-based processing pipeline

Preprocessing Modules:
  bet               : Brain extraction using HD-BET
  bias_correction   : N4 bias field correction
  denoise           : Patch2Self denoising for DWI
  motion_correction : Motion correction for DWI
  normalize         : Intensity normalization

Registration Modules:
  coregister        : Image coregistration using ANTs
  apply_warp        : Apply transformations to images

Distortion Correction:
  SDC               : Susceptibility distortion correction
  apply_SDC         : Apply precomputed SDC warp fields
  synth_b0          : Synthetic B0 generation

Segmentation and Analysis:
  synthseg          : SynthSeg brain segmentation
  texture_generation: Texture feature extraction
  compute_fa_md     : DTI metric computation

Utilities:
  extract_b0        : Extract b=0 volumes from DWI
  calculate_dice    : DICE coefficient calculation

Command Routing:
---------------
1. User runs: micaflow [command] [args...]
2. CLI checks for help flags and intercepts if present
3. Command is routed to either:
   - Snakemake pipeline (for 'pipeline' command)
   - Individual script (for module commands)
4. Arguments are formatted and passed to the target

Argument Transformation:
-----------------------
The CLI transforms Python-style argument names (with underscores) to
command-line style (with hyphens) automatically:

  Python:     output_mask -> CLI: --output-mask
  Python:     shell_dimension -> CLI: --shell-dimension

This ensures consistency between the CLI parser and individual scripts.

Features:
--------
- Color-coded help messages using colorama
- Custom help formatting with extended descriptions
- Automatic help interception for subcommands
- Dynamic argument forwarding to processing scripts
- Support for both pipeline and standalone module execution
- Comprehensive error handling and reporting
- Dry-run support for pipeline execution

Exit Codes:
----------
0 : Success - command completed successfully
1 : Error - command failed, file not found, or invalid arguments

Examples:
--------
# Show main help
$ micaflow
$ micaflow --help

# Show help for specific command
$ micaflow bet --help
$ micaflow synthseg -h

# Run full pipeline
$ micaflow pipeline --subject sub-001 --t1w-file t1.nii.gz --output /out

# Run individual module
$ micaflow bet --input t1.nii.gz --output brain.nii.gz --output-mask mask.nii.gz

# Dry run (pipeline only)
$ micaflow pipeline --dry-run --subject sub-001 --t1w-file t1.nii.gz

Notes:
-----
- The CLI uses subprocess.run() to execute individual scripts
- All scripts are invoked as Python modules: python -m micaflow.scripts.[name]
- The pipeline command uses Snakemake for workflow management
- Configuration can be provided via command-line args or YAML config file
"""

import argparse
import sys
import subprocess
import glob
import os
import time
import json
import datetime
from colorama import init, Fore, Style
import importlib.resources

init()


def get_snakefile_path():
    """
    Get the path to the Snakefile within the installed package using importlib.resources.

    Returns
    -------
    str
        Absolute path to the Snakefile for use with Snakemake.

    Raises
    ------
    pkg_resources.DistributionNotFound
        If the micaflow package is not installed.
    pkg_resources.ResourceNotFound
        If the Snakefile is not found in the package resources.

    Examples
    --------
    >>> snakefile = get_snakefile_path()
    >>> print(snakefile)
    /path/to/site-packages/micaflow/resources/Snakefile

    >>> # Use with snakemake command
    >>> cmd = ['snakemake', '-s', get_snakefile_path(), '--cores', '4']

    Notes
    -----
    - The Snakefile must be in micaflow/resources/Snakefile
    - Path is resolved at runtime based on installation location
    - Used exclusively by the 'pipeline' command
    """
    try:
        # Python 3.9+: files() returns a Traversable object
        snakefile = importlib.resources.files("micaflow.resources").joinpath("Snakefile")
        return str(snakefile)
    except Exception:
        # Fallback for older Python versions
        import pkg_resources
        return pkg_resources.resource_filename("micaflow", "resources/Snakefile")


def print_extended_help():
    # ANSI color codes
    CYAN = Fore.CYAN
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    BLUE = Fore.BLUE
    MAGENTA = Fore.MAGENTA
    BOLD = Style.BRIGHT
    RESET = Style.RESET_ALL

    help_msg = f"""
    {CYAN}{BOLD}╔════════════════════════════════════════════════════════════════╗
    ║                  MICAFLOW MRI PROCESSING PIPELINE              ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    MicaFlow is a comprehensive pipeline for processing structural and 
    diffusion MRI data, with various modules that can be used independently.

    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow {GREEN}[command]{RESET} [options]
      micaflow {GREEN}pipeline{RESET} [options]
      micaflow {GREEN}bids{RESET} [options]
    
    {CYAN}{BOLD}─────────────────── AVAILABLE COMMANDS ───────────────────{RESET}
      {GREEN}pipeline{RESET}          : Run the full processing pipeline (default)
      {GREEN}bids{RESET}              : Run the pipeline on an entire BIDS dataset (batch mode)
      {GREEN}apply_warp{RESET}        : Apply transformation to warp an image to a reference space
      {GREEN}bet{RESET}               : Run brain extraction
      {GREEN}bias_correction{RESET}   : Run N4 Bias Field Correction
      {GREEN}calculate_dice{RESET}    : Calculate DICE score between two segmentations
      {GREEN}compute_fa_md{RESET}     : Compute Fractional Anisotropy and Mean Diffusivity maps
      {GREEN}coregister{RESET}        : Coregister a moving image to a reference image
      {GREEN}denoise{RESET}           : Denoise diffusion-weighted images using Patch2Self
      {GREEN}motion_correction{RESET} : Perform motion correction on diffusion-weighted images
      {GREEN}SDC{RESET}               : Run Susceptibility Distortion Correction on DWI images
      {GREEN}apply_SDC{RESET}         : Apply pre-computed SDC warp field to an image
      {GREEN}synthseg{RESET}          : Run SynthSeg brain segmentation
      {GREEN}texture_generation{RESET}: Generate texture features from neuroimaging data
    
    {CYAN}{BOLD}──────────────── PIPELINE REQUIRED PARAMETERS ────────────{RESET}
      {YELLOW}--subject{RESET} SUBJECT_ID           Subject ID
      {YELLOW}--output{RESET} OUTPUT_DIR           Output directory 
      {YELLOW}--t1w-file{RESET} T1-weighted image file
    
    {CYAN}{BOLD}──────────────── PIPELINE OPTIONAL PARAMETERS ────────────{RESET}
      {YELLOW}--data-directory{RESET} DATA_DIR      Input data directory
      {YELLOW}--session{RESET} SESSION_ID           Session ID (default: none)
      {YELLOW}--flair-file{RESET} FLAIR_FILE        FLAIR image file
      {YELLOW}--dwi-file{RESET} DWI_FILE            Diffusion weighted image
      {YELLOW}--bval-file{RESET}                    B-value file for DWI
      {YELLOW}--bvec-file{RESET}                    B-vector file for DWI
      {YELLOW}--inverse-dwi-file{RESET} INV_FILE    Inverse (PA) DWI for distortion correction
      {YELLOW}--inverse-bval-file{RESET} INV_BVAL   Inverse b-value file for DWI
      {YELLOW}--inverse-bvec-file{RESET} INV_BVEC   Inverse b-vector file for DWI
      {YELLOW}--gpu{RESET}                          Use GPU for computation
      {YELLOW}--cores{RESET}                        Number of CPU cores to use (default: 1)
      {YELLOW}--dry-run{RESET}, {YELLOW}-n{RESET}                  Dry run (don't execute commands)
      {YELLOW}--config-file{RESET} FILE             Path to a YAML configuration file
      {YELLOW}--extract-brain{RESET}                Generate brain-extracted versions of all outputs in a dedicated directory
      {YELLOW}--keep-temp{RESET}                    Keep temporary processing files (useful for debugging)
      {YELLOW}--rm-cerebellum{RESET}                Remove cerebellum from brain extraction outputs
      {YELLOW}--PED{RESET}                          Phase encoding direction of DWI, options are: 'ap', 'pa', 'lr', 'rl', 'si', 'is' (default: 'pa')
      {YELLOW}--shell-dimension{RESET}              Dimension of the DWI image referring to shells (default: 3)
      {YELLOW}--linear{RESET}                       Use linear-only registration to MNI space (if specified alone)
      {YELLOW}--nonlinear{RESET}                    Use nonlinear registration to MNI space (default if neither specified)

    {CYAN}{BOLD}─────────────────── BIDS BATCH USAGE ────────────────────{RESET}
      micaflow {GREEN}bids{RESET} {YELLOW}--bids-dir{RESET} PATH {YELLOW}--output-dir{RESET} PATH [options]

      {BLUE}# Process entire BIDS directory{RESET}
      micaflow {GREEN}bids{RESET} {YELLOW}--bids-dir{RESET} /data/bids {YELLOW}--output-dir{RESET} /data/derivatives \\
        {YELLOW}--cores{RESET} 4 {YELLOW}--gpu{RESET}
      
      {BLUE}# Process specific subjects with custom suffixes{RESET}
      micaflow {GREEN}bids{RESET} {YELLOW}--bids-dir{RESET} /data/bids {YELLOW}--output-dir{RESET} /data/derivatives \\
        {YELLOW}--participant-label{RESET} 001 002 {YELLOW}--dwi-suffix{RESET} dwi_acq-AP.nii.gz

    {CYAN}{BOLD}────────────────── EXAMPLE PIPELINE USAGE ───────────────{RESET}

    {BLUE}# Process a single subject with T1w only{RESET}
    micaflow {GREEN}pipeline{RESET} {YELLOW}--subject{RESET} sub-001 {YELLOW}--session{RESET} ses-01 \\
      {YELLOW}--data-directory{RESET} /data {YELLOW}--t1w-file{RESET} sub-001_ses-01_T1w.nii.gz \\
      {YELLOW}--output{RESET} /output {YELLOW}--cores{RESET} 4
    
    {BLUE}# Process with FLAIR and brain extraction{RESET}
    micaflow {GREEN}pipeline{RESET} {YELLOW}--subject{RESET} sub-001 {YELLOW}--session{RESET} ses-01 \\
      {YELLOW}--data-directory{RESET} /data {YELLOW}--t1w-file{RESET} sub-001_ses-01_T1w.nii.gz \\
      {YELLOW}--flair-file{RESET} sub-001_ses-01_FLAIR.nii.gz {YELLOW}--output{RESET} /output \\
      {YELLOW}--extract-brain{RESET} {YELLOW}--cores{RESET} 4
    
    {BLUE}# Process with diffusion data, keep temporary files and remove cerebellum{RESET}
    micaflow {GREEN}pipeline{RESET} {YELLOW}--subject{RESET} sub-001 {YELLOW}--session{RESET} ses-01 \\
      {YELLOW}--data-directory{RESET} /data {YELLOW}--t1w-file{RESET} sub-001_ses-01_T1w.nii.gz \\
      {YELLOW}--dwi-file{RESET} sub-001_ses-01_dwi.nii.gz \\
      {YELLOW}--bval-file{RESET} sub-001_ses-01_dwi.bval {YELLOW}--bvec-file{RESET} sub-001_ses-01_dwi.bvec \\
      {YELLOW}--inverse-dwi-file{RESET} sub-001_ses-01_acq-PA_dwi.nii.gz \\
      {YELLOW}--output{RESET} /output {YELLOW}--keep-temp{RESET} {YELLOW}--rm-cerebellum{RESET} {YELLOW}--cores{RESET} 4
    
    {CYAN}{BOLD}─────────────────── EXAMPLE MODULE USAGE ────────────────{RESET}

    {BLUE}# Run brain extraction{RESET}
    micaflow {GREEN}bet{RESET} {YELLOW}--input{RESET} t1w.nii.gz {YELLOW}--output{RESET} brain.nii.gz {YELLOW}--output-mask{RESET} mask.nii.gz
      
    {BLUE}# Run brain extraction with cerebellum removal{RESET}
    micaflow {GREEN}bet{RESET} {YELLOW}--input{RESET} t1w.nii.gz {YELLOW}--output{RESET} brain.nii.gz {YELLOW}--output-mask{RESET} mask.nii.gz {YELLOW}--remove-cerebellum{RESET}
    
    {BLUE}# Run SynthSeg segmentation{RESET}
    micaflow {GREEN}synthseg{RESET} {YELLOW}--i{RESET} t1w.nii.gz {YELLOW}--o{RESET} segmentation.nii.gz {YELLOW}--parc{RESET}
      
    {BLUE}# Apply warp transformation{RESET}
    micaflow {GREEN}apply_warp{RESET} {YELLOW}--moving{RESET} t1w.nii.gz {YELLOW}--reference{RESET} template.nii.gz \\
      {YELLOW}--warp{RESET} warp.nii.gz {YELLOW}--affine{RESET} transform.mat {YELLOW}--output{RESET} warped.nii.gz
    
    {CYAN}{BOLD}───────────────── OUTPUT DIRECTORY STRUCTURE ────────────{RESET}
    output/
    └── <subject>/
        └── <session>/
            ├── anat/                 # Anatomical images (bias-corrected)
            ├── brain-extracted/      # Brain-extracted images (with --extract-brain)
            ├── dwi/                  # Processed diffusion data and DTI metrics
            ├── metrics/              # Quality metrics and DICE scores
            ├── temp/                 # Temporary files (preserved with --keep-temp)
            ├── textures/             # Texture features
            └── xfm/                  # Transformation matrices and warps
    
    {CYAN}{BOLD}────────────────────────── NOTES ───────────────────────{RESET}
    {MAGENTA}•{RESET} For help on a specific module: micaflow [command] --help
    {MAGENTA}•{RESET} The pipeline uses Snakemake for workflow management
    {MAGENTA}•{RESET} Config file can be used to specify parameters instead of command line options
    {MAGENTA}•{RESET} Each module can be run independently with its own set of parameters
    {MAGENTA}•{RESET} Use --extract-brain to generate skull-stripped versions of all outputs in a dedicated directory
    {MAGENTA}•{RESET} Use --keep-temp to preserve intermediate files (useful for debugging)
    {MAGENTA}•{RESET} Use --rm-cerebellum to remove cerebellum from brain extraction outputs
    
    For more detailed help on any command, use: micaflow {GREEN}[command]{RESET} {YELLOW}--help{RESET}
    """
    return help_msg


def main():
    """
    Main entry point for the MicaFlow command-line interface.
    
    This function handles all CLI operations including:
    - Argument parsing for pipeline and module commands
    - Help message interception and display
    - Command routing to appropriate handlers
    - Dynamic argument transformation and forwarding
    - Error handling and reporting
    
    The function uses a two-level command structure where the first argument
    determines whether to run the full pipeline or an individual module.
    
    Flow:
    -----
    1. Check for help flags or no arguments -> display help and exit
    2. Intercept subcommand help requests before argparse processing
    3. Parse arguments using argparse with custom formatter
    4. Route to appropriate command handler:
       - 'pipeline': Build and execute Snakemake command
       - Module commands: Transform args and execute module script
    5. Handle errors and report results
    
    Command Routing:
    ---------------
    Pipeline Command:
      - Collects configuration parameters
      - Builds Snakemake command with --config options
      - Optionally loads YAML config file
      - Executes via subprocess.run()
    
    Module Commands:
      - Transform Python-style args to CLI-style (underscores to hyphens)
      - Build command-line argument list
      - Execute via subprocess.run() with python -m
      - Report results and any errors
    
    Argument Transformation:
    -----------------------
    For module commands, arguments are transformed:
    - Python: output_mask -> CLI: --output-mask
    - Python: shell_dimension -> CLI: --shell-dimension
    - Boolean flags: Included only if True
    - Lists: Expanded into multiple values
    
    Exit Codes:
    ----------
    0 : Success - command completed
    1 : Error - command failed or invalid arguments
    
    Examples
    --------
    >>> # Run from command line
    >>> # micaflow
    >>> # micaflow pipeline --subject sub-001 --t1w-file t1.nii.gz
    >>> # micaflow bet --input t1.nii.gz --output brain.nii.gz
    
    >>> # Can also be called programmatically
    >>> if __name__ == "__main__":
    ...     main()
    
    Raises
    ------
    subprocess.CalledProcessError
        If a module or pipeline command fails during execution.
    SystemExit
        Called with exit code 0 or 1 depending on success/failure.
    
    Notes
    -----
    - Uses subprocess.run() for all command execution
    - Scripts executed as Python modules: python -m micaflow.scripts.[name]
    - Help is intercepted before argparse to allow custom formatting
    - Unknown arguments passed through to Snakemake for pipeline command
    - Print statements used for user feedback (not logging framework)
    
    See Also
    --------
    print_extended_help : Extended help message display
    get_snakefile_path : Get path to pipeline Snakefile
    """
    # If no arguments provided, show help and exit
    if len(sys.argv) == 1:
        print(print_extended_help())
        sys.exit(0)

    # Intercept help requests for subcommands before argparse processes them
    # This allows us to show custom help for individual modules
    if (len(sys.argv) >= 3 and sys.argv[2] in ["-h", "--help"]) or len(sys.argv) == 2:
        # Check if the first argument is a valid command
        command = sys.argv[1]
        
        # Map of command names to their Python module paths
        script_map = {
            "apply_warp": "micaflow.scripts.apply_warp",
            "bet": "micaflow.scripts.bet",
            "bias_correction": "micaflow.scripts.bias_correction",
            "calculate_dice": "micaflow.scripts.calculate_dice",
            "compute_fa_md": "micaflow.scripts.compute_fa_md",
            "coregister": "micaflow.scripts.coregister",
            "denoise": "micaflow.scripts.denoise",
            "motion_correction": "micaflow.scripts.motion_correction",
            "SDC": "micaflow.scripts.SDC",
            "apply_SDC": "micaflow.scripts.apply_SDC",
            "synthseg": "micaflow.scripts.synthseg",
            "texture_generation": "micaflow.scripts.texture_generation",
        }

        if command in script_map:
            if command != "pipeline":  # Special case for pipeline
                try:
                    print(f"\n=== Help for '{command}' command ===\n")
                    subprocess.run(
                        ["python", "-m", script_map[command], "--help"], check=True
                    )
                    sys.exit(0)
                except subprocess.CalledProcessError as e:
                    print(f"Error displaying help for {command}: {e}")
                    sys.exit(1)

    # Create custom formatter that includes our extended help
    class CustomHelpFormatter(argparse.HelpFormatter):
        """
        Custom help formatter for argparse that displays extended help.
        
        This formatter overrides the default argparse help to show the
        comprehensive, color-coded help message from print_extended_help().
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, width=100)

        def format_help(self):
            # Include standard argparse help and our extended help
            standard_help = super().format_help()
            if "-h" in sys.argv or "--help" in sys.argv:
                return print_extended_help()
            return standard_help

    parser = argparse.ArgumentParser(
        description="Run the micaflow MRI processing pipeline",
        formatter_class=CustomHelpFormatter,
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Add arguments that match the config parameters
    # Pipeline command (default)
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Run the full micaflow pipeline"
    )
    # Add pipeline arguments
    pipeline_parser.add_argument("--subject", help="Subject ID (e.g., sub-01)")
    pipeline_parser.add_argument("--session", help="Session ID (e.g., ses-01)")
    pipeline_parser.add_argument("--output", help="Output directory")
    pipeline_parser.add_argument(
        "--data-directory", default="", help="Data directory path"
    )
    pipeline_parser.add_argument("--flair-file", help="Path to FLAIR image")
    pipeline_parser.add_argument("--t1w-file", help="Path to T1w image")
    pipeline_parser.add_argument("--dwi-file", help="Path to DWI image")
    pipeline_parser.add_argument("--bval-file", help="Path to bval file")
    pipeline_parser.add_argument("--bvec-file", help="Path to bvec file")
    pipeline_parser.add_argument("--inverse-dwi-file", help="Path to inverse DWI file")
    pipeline_parser.add_argument("--inverse-bval-file", help="Path to inverse bval file")
    pipeline_parser.add_argument("--inverse-bvec-file", help="Path to inverse bvec file")
    pipeline_parser.add_argument(
        "--gpu", action="store_true", help="Use GPU computation"
    )
    pipeline_parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Dry run (don't execute commands)"
    )
    pipeline_parser.add_argument(
        "--cores", type=int, default=1, 
        help="Number of Snakemake jobs to run in parallel (Snakemake --cores)"
    )
    pipeline_parser.add_argument(
        "--config-file", help="Path to a YAML configuration file"
    )
    pipeline_parser.add_argument(
        "--rm-cerebellum", action="store_true", help="Remove cerebellum from images"
    )
    pipeline_parser.add_argument(
        "--keep-temp", action="store_true", help="Keep temporary files after processing"
    )
    pipeline_parser.add_argument(
        "--extract-brain", action="store_true", help="Keep brain-extracted images"
    )
    pipeline_parser.add_argument(
        "--PED", default="pa", help="Phase encoding direction of DWI, options are: 'ap', 'pa', 'lr', 'rl', 'si', 'is'"
    )
    pipeline_parser.add_argument(
        "--shell-dimension", type=int, default=3, help="Dimension of the DWI image referring to shells"
    )
    pipeline_parser.add_argument(
        "--linear", action="store_true", help="Use linear-only registration to MNI space (if specified alone)"
    )
    pipeline_parser.add_argument(
        "--nonlinear", action="store_true", help="Use nonlinear registration to MNI space (default if neither specified)"
    )

    # BIDS Batch Processing Command
    bids_parser = subparsers.add_parser(
        "bids", help="Run the pipeline on an entire BIDS dataset"
    )
    bids_parser.add_argument("--bids-dir", required=True, help="Path to BIDS root directory")
    bids_parser.add_argument("--output-dir", required=True, help="Path to derivatives/output directory")
    bids_parser.add_argument("--participant-label", nargs="+", help="Specific list of subjects to process (without 'sub-' prefix)")
    bids_parser.add_argument("--session-label", nargs="+", help="Specific list of sessions to process (without 'ses-' prefix)")
    
    # Suffix configuration
    bids_parser.add_argument("--t1w-suffix", default="T1w.nii.gz", help="Suffix for T1w images (default: T1w.nii.gz)")
    bids_parser.add_argument("--flair-suffix", help="Suffix for FLAIR images (e.g. FLAIR.nii.gz). If not provided, FLAIR is skipped.")
    bids_parser.add_argument("--dwi-suffix", help="Suffix for DWI images (e.g. dwi.nii.gz). If not provided, DWI is skipped.")
    bids_parser.add_argument("--inverse-dwi-suffix", help="Suffix for Inverse DWI images (e.g. acq-rpe_dwi.nii.gz). If not provided, ignored.")

    # Passthrough arguments
    bids_parser.add_argument("--gpu", action="store_true", help="Use GPU computation")
    bids_parser.add_argument("--dry-run", "-n", action="store_true", help="Print commands without executing")
    bids_parser.add_argument("--cores", type=int, default=1, help="Number of cores per subject")
    bids_parser.add_argument("--rm-cerebellum", action="store_true", help="Remove cerebellum")
    bids_parser.add_argument("--extract-brain", action="store_true", help="Generate brain-extracted outputs")
    bids_parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    bids_parser.add_argument("--linear", action="store_true", help="Use linear-only registration")
    bids_parser.add_argument("--nonlinear", action="store_true", help="Use nonlinear registration")
    bids_parser.add_argument("--PED", default="pa", help="Phase encoding direction (default: pa)")
    bids_parser.add_argument("--shell-dimension", type=int, default=3, help="Shell dimension")
    bids_parser.add_argument("--config-file", help="YAML config file")

    # SynthSeg command
    synthseg_parser = subparsers.add_parser(
        "synthseg", help="Run SynthSeg brain segmentation"
    )
    synthseg_parser.add_argument(
        "--i", help="Image(s) to segment. Can be a path to an image or to a folder."
    )
    synthseg_parser.add_argument(
        "--o",
        help="Segmentation output(s). Must be a folder if --i designates a folder.",
    )
    synthseg_parser.add_argument(
        "--parc",
        action="store_true",
        help="(optional) Whether to perform cortex parcellation.",
    )
    synthseg_parser.add_argument(
        "--robust",
        action="store_true",
        help="(optional) Whether to use robust predictions (slower).",
    )
    synthseg_parser.add_argument(
        "--fast",
        action="store_true",
        help="(optional) Bypass some postprocessing for faster predictions.",
    )
    synthseg_parser.add_argument(
        "--ct",
        action="store_true",
        help="(optional) Clip intensities to [0,80] for CT scans.",
    )
    synthseg_parser.add_argument(
        "--vol",
        help="(optional) Path to output CSV file with volumes (mm3) for all regions and subjects.",
    )
    synthseg_parser.add_argument(
        "--qc",
        help="(optional) Path to output CSV file with qc scores for all subjects.",
    )
    synthseg_parser.add_argument(
        "--post",
        help="(optional) Posteriors output(s). Must be a folder if --i designates a folder.",
    )
    synthseg_parser.add_argument(
        "--resample",
        help="(optional) Resampled image(s). Must be a folder if --i designates a folder.",
    )
    synthseg_parser.add_argument(
        "--crop",
        nargs="+",
        type=int,
        help="(optional) Size of 3D patches to analyse. Default is 192.",
    )
    synthseg_parser.add_argument(
        "--threads", help="(optional) Number of cores to be used. Default is 1."
    )
    synthseg_parser.add_argument(
        "--cpu",
        action="store_true",
        help="(optional) Enforce running with CPU rather than GPU.",
    )
    synthseg_parser.add_argument(
        "--v1",
        action="store_true",
        help="(optional) Use SynthSeg 1.0 (updated 25/06/22).",
    )

    # SDC command
    apply_sdc_parser = subparsers.add_parser("apply_SDC")
    apply_sdc_parser.add_argument(
        "--input",
        required=True,
        help="Path to the motion-corrected DWI image (.nii.gz)",
    )
    apply_sdc_parser.add_argument(
        "--warp",
        required=True,
        help="Path to the warp field estimated from SDC (.nii.gz)",
    )
    apply_sdc_parser.add_argument(
        "--affine",
        required=True,
        help="Path to an image from which to extract the affine matrix",
    )
    apply_sdc_parser.add_argument(
        "--output", required=True, help="Output path for the corrected image"
    )

    # Apply Warp command
    apply_warp_parser = subparsers.add_parser(
        "apply_warp", help="Apply transformation to warp an image to a reference space"
    )
    apply_warp_parser.add_argument(
        "--moving", required=True, help="Path to the moving image that will be warped"
    )
    apply_warp_parser.add_argument(
        "--reference", required=True, help="Path to the reference/target image"
    )
    apply_warp_parser.add_argument(
        "--warp",
        help="Path to the warp field for non-linear transformation",
    )
    apply_warp_parser.add_argument(
        "--secondary-warp",
        help="Path to a secondary warp field to be applied after the primary warp and affine",
    )
    apply_warp_parser.add_argument(
        "--affine", help="Path to the affine transformation file"
    )
    apply_warp_parser.add_argument(
        "--output", required=True, help="Output path for the warped image"
    )
    apply_warp_parser.add_argument(
        "--interpolation",
        default="linear",
        help="Interpolation method (default: linear).",
    )
    apply_warp_parser.add_argument(
        "--transforms", nargs="+", help="List of transforms to apply in order (First -> Last)."
    )

    # Brain Extraction Tool command
    bet_parser = subparsers.add_parser("bet", help="Run HD-BET brain extraction")
    bet_parser.add_argument(
        "--input", required=True, help="Path to the input image (.nii.gz)"
    )
    bet_parser.add_argument(
        "--output",
        required=True,
        help="Path to the output brain-extracted image (.nii.gz)",
    )
    bet_parser.add_argument(
        "--output-mask", help="Path to the output brain mask (.nii.gz)"
    )
    bet_parser.add_argument(
        "--input-mask", help="Path to the input brain mask (.nii.gz) (optional)"
    )
    bet_parser.add_argument(
        "--parcellation", help="Parcellation file for the input image (optional)"
    )
    bet_parser.add_argument(
        "--remove-cerebellum",
        action="store_true",
        help="Remove cerebellum from the input image (optional)",
    )

    # Bias Correction Tool command
    bias_corr_parser = subparsers.add_parser(
        "bias_correction", help="Run N4 Bias Field Correction"
    )
    bias_corr_parser.add_argument(
        "--input", "-i", required=True, help="Path to the input image (.nii.gz)"
    )
    bias_corr_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to the output bias-corrected image (.nii.gz)",
    )
    bias_corr_parser.add_argument(
        "--mask",
        "-m",
        help="Path to a mask image (required for 4D images, optional for 3D)",
    )
    bias_corr_parser.add_argument(
        "--mode",
        choices=["3d", "4d", "auto"],
        default="auto",
        help="Processing mode: 3d=anatomical, 4d=diffusion, auto=detect (default)",
    )
    bias_corr_parser.add_argument(
        "--b0", help="b0 image path, required for 4D diffusion images."
    )
    bias_corr_parser.add_argument(
        "--b0-output", help="Path for the output corrected b0 image (only for 4D DWI)."
    )
    bias_corr_parser.add_argument(
        "--shell-dimension",
        type=int,
        default=3,
        help="Dimension of the DWI image referring to shells (default: 3)",
    )
    bias_corr_parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads to use for bias correction (default: 1)",
    )
    bias_corr_parser.add_argument(
        "--gibbs", action="store_true", help="Apply Gibbs ringing correction"
    )

    # DICE Calculator command
    dice_parser = subparsers.add_parser(
        "calculate_dice",
        help="Calculate DICE between two segmentations",
    )
    dice_parser.add_argument("--input", "-i", required=True, help="First input volume")
    dice_parser.add_argument(
        "--reference", "-r", required=True, help="Reference volume to compare against"
    )
    dice_parser.add_argument(
        "--output", "-o", required=True, help="Output CSV file path"
    )

    # Compute FA/MD command
    compute_fa_md_parser = subparsers.add_parser(
        "compute_fa_md", help="Compute Fractional Anisotropy and Mean Diffusivity maps"
    )
    compute_fa_md_parser.add_argument(
        "--input", required=True, help="Path to the preprocessed DWI image (.nii.gz)"
    )
    compute_fa_md_parser.add_argument(
        "--bval", required=True, help="Path to the b-values file (.bval)"
    )
    compute_fa_md_parser.add_argument(
        "--bvec", required=True, help="Path to the b-vectors file (.bvec)"
    )
    compute_fa_md_parser.add_argument(
        "--mask", help="Optional: Path to a brain mask (.nii.gz)"
    )
    compute_fa_md_parser.add_argument(
        "--output-fa",
        required=True,
        help="Output path for the Fractional Anisotropy map (.nii.gz)",
    )
    compute_fa_md_parser.add_argument(
        "--output-md",
        required=True,
        help="Output path for the Mean Diffusivity map (.nii.gz)",
    )    
    compute_fa_md_parser.add_argument("--b0-volume", type=str,
                        help="Path to the b0 volume to merge with DWI.")
    compute_fa_md_parser.add_argument("--b0-bval", type=str,
                        help="Path to b0 b-value file.")
    compute_fa_md_parser.add_argument("--b0-bvec", type=str,
                        help="Path to b0 b-vector file.")
    compute_fa_md_parser.add_argument("--b0-index", type=int, default=0,
                        help="Index at which to insert b0 volume (default: 0).")

    # Coregistration command
    coreg_parser = subparsers.add_parser(
        "coregister", help="Coregister a moving image to a reference image using label-augmented registration"
    )
    coreg_parser.add_argument(
        "--fixed-file", required=True, 
        help="Path to the fixed/reference image (.nii.gz)"
    )
    coreg_parser.add_argument(
        "--moving-file", required=True, 
        help="Path to the moving image to be registered (.nii.gz)"
    )
    coreg_parser.add_argument(
        "--fixed-segmentation", 
        help="Path to the fixed segmentation image (.nii.gz). If not provided, will be generated automatically."
    )
    coreg_parser.add_argument(
        "--moving-segmentation", 
        help="Path to the moving segmentation image (.nii.gz). If not provided, will be generated automatically."
    )
    coreg_parser.add_argument(
        "--output", required=True,
        help="Output path for the registered image (.nii.gz)"
    )
    coreg_parser.add_argument(
        "--warp-file", default=None, 
        help="Optional path to save the forward warp field (moving to fixed) (.nii.gz)"
    )
    coreg_parser.add_argument(
        "--secondary-warp-file", default=None,
        help="Optional path to save a secondary warp field to be applied after the primary warp and affine (.nii.gz)"   
    )
    coreg_parser.add_argument(
        "--affine-file", default=None,
        help="Optional path to save the forward affine transform (moving to fixed) (.mat)"
    )
    coreg_parser.add_argument(
        "--rev-warp-file", default=None,
        help="Optional path to save the reverse warp field (fixed to moving) (.nii.gz)"
    )
    coreg_parser.add_argument(
        "--secondary-rev-warp-file", default=None,
        help="Optional path to save a secondary reverse warp field to be applied after the primary reverse warp and affine (.nii.gz)"
    )
    coreg_parser.add_argument(
        "--threads", type=int, default=1, 
        help="Number of threads for registration operations (default: 1)"
    )
    coreg_parser.add_argument(
        "--output-segmentation",
        help="Path to save the transformed segmentation alongside the registered image (.nii.gz)"
    )
    coreg_parser.add_argument(
        "--linear-only", action='store_true',
        help="Perform only linear registration (rigid + affine) without nonlinear SyN warping (faster)"
    )
    coreg_parser.add_argument(
        "--disable-robust", action='store_true',
        help="If set, disables robust registration mode in LAMAReg."
    )
    

    # Denoise command
    denoise_parser = subparsers.add_parser(
        "denoise", help="Denoise diffusion-weighted images using Patch2Self"
    )
    denoise_parser.add_argument(
        "--input", required=True, help="Path to the input DWI image (.nii.gz)"
    )
    denoise_parser.add_argument(
        "--bval", required=True, help="Path to the b-values file (.bval)"
    )
    denoise_parser.add_argument(
        "--bvec", required=True, help="Path to the b-vectors file (.bvec)"
    )
    denoise_parser.add_argument(
        "--output", required=True, help="Output path for the denoised image (.nii.gz)"
    )
    denoise_parser.add_argument("--b0-denoise", action='store_true', help="Denoise b0 volumes separately (default: False)")
    denoise_parser.add_argument("--gibbs", action='store_true', help="Apply Gibbs ringing correction (default: False)")
    denoise_parser.add_argument("--threads", type=int, help="Number of threads to use (default: 1)")

    # Motion Correction command
    motion_corr_parser = subparsers.add_parser(
        "motion_correction",
        help="Perform motion correction on diffusion-weighted images",
    )
    motion_corr_parser.add_argument(
        "--denoised", required=True, help="Path to the denoised DWI (NIfTI file)."
    )
    motion_corr_parser.add_argument(
        "--input-bvals",
        type=str,
        required=True,
        help="Path to the bvals file.",
    )
    motion_corr_parser.add_argument(
        "--input-bvecs",
        type=str,
        required=True,
        help="Path to the bvecs file.",
    )
    motion_corr_parser.add_argument(
        "--output-bvecs",
        type=str,
        required=True,
        help="Path to the adjusted bvecs file.",
    )
    motion_corr_parser.add_argument(
        "--output", required=True, help="Output path for the motion-corrected DWI."
    )
    motion_corr_parser.add_argument(
        "--b0",
        type=str,
        help="Path to an external B0 image to use as reference. If not provided, the first volume is used.",
    )
    motion_corr_parser.add_argument(
        "--shell-dimension",
        type=int,
        default=3,
        help="Dimension of the DWI image referring to shells (default: 3)",
    )
    motion_corr_parser.add_argument(
        "--threads",
        type=int,
        help="Number of threads to use (default: 1)",
    )
    motion_corr_parser.add_argument(
        "--temp-dir",
        type=str,
        default="tmp",
        help="Path to the temporary directory for intermediate files (default: tmp)."
    )

    # SDC command (main susceptibility distortion correction)
    sdc_parser = subparsers.add_parser(
        "SDC", help="Run Susceptibility Distortion Correction on DWI images"
    )
    sdc_parser.add_argument(
        "--input", required=True, help="Path to the data image (NIfTI file)"
    )
    sdc_parser.add_argument(
        "--reverse-image",
        required=True,
        help="Path to the reverse phase-encoded image (NIfTI file)",
    )
    sdc_parser.add_argument(
        "--output",
        required=True,
        help="Output name for the corrected image (NIfTI file)",
    )
    sdc_parser.add_argument(
        "--output-warp",
        required=True,
        help="Output name for the warp field (NIfTI file)",
    )
    sdc_parser.add_argument(
        "--phase-encoding",
        type=str,
        default="ap",
        choices=["ap", "pa", "lr", "rl", "si", "is"],
        help="Phase-encoding direction (default: ap)"
    )

    # Texture Generation command
    texture_parser = subparsers.add_parser(
        "texture_generation", help="Generate texture features from neuroimaging data"
    )
    texture_parser.add_argument(
        "--input", "-i", required=True, help="Path to the input image file (.nii.gz)"
    )
    texture_parser.add_argument(
        "--mask", "-m", required=True, help="Path to the binary mask file (.nii.gz)"
    )
    texture_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for texture feature maps",
    )

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize MRI intensity values"
    )
    normalize_parser.add_argument(
        "--input", "-i", required=True, help="Input NIfTI image file (.nii.gz)"
    )
    normalize_parser.add_argument(
        "--output", "-o", required=True, help="Output normalized image file (.nii.gz)"
    )
    normalize_parser.add_argument(
        "--lower-percentile",
        type=float,
        default=1.0,
        help="Lower percentile for clamping (default: 1.0)",
    )
    normalize_parser.add_argument(
        "--upper-percentile",
        type=float,
        default=99.0,
        help="Upper percentile for clamping (default: 99.0)",
    )
    normalize_parser.add_argument(
        "--min-value",
        type=float,
        default=0,
        help="Minimum value in output range (default: 0)",
    )
    normalize_parser.add_argument(
        "--max-value",
        type=float,
        default=100,
        help="Maximum value in output range (default: 100)",
    )

    # Synthetic B0 Generation command
    synth_b0_parser = subparsers.add_parser(
        "synth_b0", help="Create a synthetic B0 image from T1w and distorted B0 images using an ensemble of models"
    )
    synth_b0_parser.add_argument(
        '--t1', required=True, help='Path to T1w input image (.nii.gz)'
    )
    synth_b0_parser.add_argument(
        '--b0', required=True, help='Path to distorted B0 input image (.nii.gz)'
    )
    synth_b0_parser.add_argument(
        '--output', required=True, help='Path for synthetic B0 output image (.nii.gz)'
    )
    synth_b0_parser.add_argument(
        '--intermediate', help='Path to save the synthetic B0 before inverse transform'
    )
    synth_b0_parser.add_argument(
        '--cpu', action='store_true', help='Force CPU usage (default: use GPU if available)'
    )
    synth_b0_parser.add_argument(
        '--phase-encoding', help='Index of the volume to extract from 4D DWI images (if applicable)'
    )
    synth_b0_parser.add_argument(
        '--warp', help='Path to save the warp field'
    )
    synth_b0_parser.add_argument(
        '--temp-dir', help='Path to a temporary directory for intermediate files'
    )
    synth_b0_parser.add_argument(
        '--corrected-b0', help='Path to save the corrected B0 image (optional)')
    synth_b0_parser.add_argument(
        '--dwi', help='Path to the full DWI image'
    )
    synth_b0_parser.add_argument(
        '--shell-dimension', type=int, default=3, help='Dimension of the DWI image referring to shells (default: 3)'
    )
    synth_b0_parser.add_argument(
        '--threads', type=int, help='Number of threads to use (default: all)'
    )
    synth_b0_parser.add_argument(
        '--b0-to-T1-warp', help='Path to save the warp field from B0 to T1w (optional)'
    )
    synth_b0_parser.add_argument(
        '--b0-to-T1-warp-secondary', help='Path to a secondary warp field to be applied after the primary warp and affine (optional)'
    )
    synth_b0_parser.add_argument(
        '--b0-to-T1-affine', help='Path to save the affine transform from B0 to T1w (optional)'
    )

    # Extract B0 command
    extract_b0_parser = subparsers.add_parser(
        "extract_b0", help="Extract b=0 volume from DWI"
    )
    extract_b0_parser.add_argument(
        "--input", required=True, help="Path to DWI image"
    )
    extract_b0_parser.add_argument(
        "--bvals", help="Path to bvals file"
    )
    extract_b0_parser.add_argument(
        "--bvecs", help="Path to bvecs file"
    )
    extract_b0_parser.add_argument(
        "--output", required=True, help="Path for extracted b=0 volume"
    )
    extract_b0_parser.add_argument(
        "--output-dwi", help="Path for output non-b0 volumes"
    )
    extract_b0_parser.add_argument(
        "--output-bvals", help="Path for output bvals file"
    )
    extract_b0_parser.add_argument(
        "--output-bvecs", help="Path for output bvecs file"
    )
    extract_b0_parser.add_argument(
        "--threshold", type=float, default=50, help="Maximum b-value to consider as b=0 (default: 50)"
    )
    extract_b0_parser.add_argument(
        "--index", type=int, help="Directly specify volume index to extract"
    )
    extract_b0_parser.add_argument(
        "--shell-dimension", type=int, default=3, help="Dimension of the DWI image referring to shells (default: 3)"
    )
    extract_b0_parser.add_argument("--b0-bval", help="Path for b0-only bval file")
    extract_b0_parser.add_argument("--b0-bvec", help="Path for b0-only bvec file")
    args, unknown = parser.parse_known_args()

    # If no command is provided, default to pipeline
    if not args.command:
        args.command = "pipeline"

    if args.command == "bids":
        # 1. Identify Subjects
        if args.participant_label:
            subjects = [f"sub-{label}" if not label.startswith("sub-") else label for label in args.participant_label]
        else:
            subjects = [d for d in os.listdir(args.bids_dir) if d.startswith("sub-") and os.path.isdir(os.path.join(args.bids_dir, d))]
        
        subjects.sort()
        if not subjects:
            print(f"{Fore.RED}No subjects found in {args.bids_dir}{Style.RESET_ALL}")
            sys.exit(1)

        print(f"{Fore.CYAN}Found {len(subjects)} subjects to process.{Style.RESET_ALL}")

        # Helper to find unique files by suffix
        def find_file(base, subfolder, suffix, desc):
            path_pattern = os.path.join(base, subfolder, f"*{suffix}")
            matches = glob.glob(path_pattern)
            if len(matches) > 1:
                print(f"{Fore.RED}Error: Multiple {desc} files found matching *{suffix} in {os.path.join(base, subfolder)}{Style.RESET_ALL}")
                for m in matches:
                    print(f"  - {os.path.basename(m)}")
                return None, True # None found, Error=True
            if len(matches) == 0:
                return None, False # None found, Error=False
            return matches[0], False # Found, Error=False

        # 2. Iterate Subjects
        for sub in subjects:
            sub_dir = os.path.join(args.bids_dir, sub)
            
            # Detect sessions
            sessions = [d for d in os.listdir(sub_dir) if d.startswith("ses-") and os.path.isdir(os.path.join(sub_dir, d))]
            
            if not sessions:
                sessions = [None] # No session structure
            else:
                sessions.sort()
                # Filter sessions if requested
                if args.session_label:
                    target_ses = [f"ses-{l}" if not l.startswith("ses-") else l for l in args.session_label]
                    sessions = [s for s in sessions if s in target_ses]

            # 3. Iterate Sessions
            for ses in sessions:
                # Start Timer
                start_time = time.time()
                run_timestamp = datetime.datetime.now().isoformat()

                # Define path
                if ses:
                    base_path = os.path.join(sub_dir, ses)
                    ses_id = ses
                    sub_ses_str = f"{sub}/{ses}"
                else:
                    base_path = sub_dir
                    ses_id = None
                    sub_ses_str = sub

                print(f"\n{Fore.GREEN}Checking {sub_ses_str}...{Style.RESET_ALL}")

                # 4. Find Files
                # T1w (Required)
                t1w, err = find_file(base_path, "anat", args.t1w_suffix, "T1w")
                if err: continue # Skip on error (ambiguity)
                if not t1w:
                    print(f"{Fore.YELLOW}Skipping {sub_ses_str}: No T1w found matching *{args.t1w_suffix}{Style.RESET_ALL}")
                    continue

                # FLAIR (Optional)
                flair = None
                if args.flair_suffix:
                    flair, err = find_file(base_path, "anat", args.flair_suffix, "FLAIR")
                    if err: continue # Skip on ambiguity

                # DWI (Optional)
                dwi = None
                if args.dwi_suffix:
                    dwi, err = find_file(base_path, "dwi", args.dwi_suffix, "DWI")
                    if err: continue # Skip on ambiguity
                
                # Inverse DWI (Optional)
                inv_dwi = None
                if args.inverse_dwi_suffix:
                    inv_dwi, err = find_file(base_path, "dwi", args.inverse_dwi_suffix, "Inverse DWI")
                    if err: continue

                # Check if DWI has bvals/bvecs (Required if DWI is present)
                bval_file = None
                bvec_file = None
                if dwi:
                    # Infer bval/bvec by replacing extension
                    prefix = dwi
                    if prefix.endswith(".nii.gz"):
                        prefix = prefix[:-7]
                    elif prefix.endswith(".nii"):
                        prefix = prefix[:-4]
                    
                    bval_cand = prefix + ".bval"
                    bvec_cand = prefix + ".bvec"
                    
                    if os.path.exists(bval_cand) and os.path.exists(bvec_cand):
                        bval_file = bval_cand
                        bvec_file = bvec_cand
                    else:
                        # Fallback: Instead of skipping subject, just drop DWI and proceed with T1/FLAIR
                        print(f"{Fore.YELLOW}Warning: DWI found but missing .bval or .bvec files for {dwi}. Proceeding without DWI.{Style.RESET_ALL}")
                        dwi = None
                
                # Check Inverse DWI bvals/bvecs
                inv_bval_file = None
                inv_bvec_file = None
                if inv_dwi:
                    prefix = inv_dwi
                    if prefix.endswith(".nii.gz"):
                        prefix = prefix[:-7]
                    elif prefix.endswith(".nii"):
                        prefix = prefix[:-4]
                    
                    cand_bval = prefix + ".bval"
                    cand_bvec = prefix + ".bvec"
                    if os.path.exists(cand_bval) and os.path.exists(cand_bvec):
                        inv_bval_file = cand_bval
                        inv_bvec_file = cand_bvec

                # 5. Build Command
                cmd = [sys.executable, "-m", "micaflow.cli", "pipeline"]
                cmd.extend(["--subject", sub])
                if ses_id:
                    cmd.extend(["--session", ses_id])
                
                cmd.extend(["--output", args.output_dir])
                
                # FIX: Do NOT pass data directory to avoid path duplication logic in pipeline.
                # Instead, pass explicit absolute paths for all files.
                # cmd.extend(["--data-directory", args.bids_dir])
                
                cmd.extend(["--t1w-file", os.path.abspath(t1w)])
                if flair:
                    cmd.extend(["--flair-file", os.path.abspath(flair)])
                
                # DWI handling
                if dwi:
                    cmd.extend(["--dwi-file", os.path.abspath(dwi)])
                    cmd.extend(["--bval-file", os.path.abspath(bval_file)])
                    cmd.extend(["--bvec-file", os.path.abspath(bvec_file)])
                
                if inv_dwi:
                    cmd.extend(["--inverse-dwi-file", os.path.abspath(inv_dwi)])
                    if inv_bval_file:
                        cmd.extend(["--inverse-bval-file", os.path.abspath(inv_bval_file)])
                        cmd.extend(["--inverse-bvec-file", os.path.abspath(inv_bvec_file)])

                # Passthrough args
                if args.gpu: cmd.append("--gpu")
                if args.rm_cerebellum: cmd.append("--rm-cerebellum")
                if args.extract_brain: cmd.append("--extract-brain")
                if args.keep_temp: cmd.append("--keep-temp")
                if args.linear: cmd.append("--linear")
                if args.nonlinear: cmd.append("--nonlinear")
                if args.config_file: cmd.extend(["--config-file", args.config_file])
                
                cmd.extend(["--cores", str(args.cores)])
                cmd.extend(["--PED", args.PED])
                cmd.extend(["--shell-dimension", str(args.shell_dimension)])

                # 6. Execute and Log
                print(f"{Fore.CYAN}Launching pipeline for {sub_ses_str}...{Style.RESET_ALL}")
                
                status = "unknown"
                error_msg = None

                if args.dry_run:
                    print(f"Dry run: {' '.join(cmd)}")
                    status = "dry_run"
                else:
                    try:
                        subprocess.run(cmd, check=True)
                        status = "success"
                    except subprocess.CalledProcessError as e:
                        print(f"{Fore.RED}Pipeline failed for {sub_ses_str}{Style.RESET_ALL}")
                        status = "failed"
                        error_msg = str(e)
                
                # 7. Generate Run Metadata JSON
                # Logic updated: Append to a single summary JSON in the main output directory
                if not args.dry_run:
                    try:
                        duration = time.time() - start_time
                        
                        # Use main output dir
                        os.makedirs(args.output_dir, exist_ok=True)
                        
                        metadata = {
                            "subject": sub,
                            "session": ses_id,
                            "timestamp": run_timestamp,
                            "duration_seconds": round(duration, 2),
                            "status": status,
                            "error": error_msg,
                            "inputs": {
                                "t1w": t1w,
                                "flair": flair,
                                "dwi": dwi,
                                "bval": bval_file,
                                "bvec": bvec_file,
                                "inverse_dwi": inv_dwi,
                                "inverse_bval": inv_bval_file
                            },
                            "config": {
                                "linear": args.linear,
                                "nonlinear": args.nonlinear,
                                "ped": args.PED,
                                "shell_dimension": args.shell_dimension,
                                "extract_brain": args.extract_brain,
                                "rm_cerebellum": args.rm_cerebellum,
                                "gpu": args.gpu
                            },
                            "command_line": cmd
                        }
                        
                        # Define the single summary log file path
                        json_path = os.path.join(args.output_dir, "micaflow_runs_summary.json")
                        
                        # Load existing data if file exists
                        run_history = []
                        if os.path.exists(json_path):
                            try:
                                with open(json_path, "r") as f:
                                    run_history = json.load(f)
                                    if not isinstance(run_history, list):
                                        # If for some reason it's not a list, wrap it or start new
                                        # (Handling backward compatibility if it was dict)
                                        run_history = [] 
                            except json.JSONDecodeError:
                                print(f"{Fore.YELLOW}Warning: Could not decode existing log file. Starting fresh.{Style.RESET_ALL}")
                                run_history = []
                        
                        # Append new run
                        run_history.append(metadata)
                        
                        # Write back to file
                        with open(json_path, "w") as f:
                            json.dump(run_history, f, indent=4)
                        
                        print(f"Run metadata appended to {json_path}")

                    except Exception as e:
                        print(f"{Fore.RED}Error saving run metadata: {e}{Style.RESET_ALL}")

    elif args.command == "pipeline":
        # Get the path to the Snakefile
        snakefile = get_snakefile_path()

        # Build the snakemake command
        cmd = ["snakemake", "-s", snakefile]

        # Add config parameters if provided
        config = {}
        for param in [
            "subject",
            "session",
            "output",
            "data_directory",
            "flair_file",
            "t1w_file",
            "dwi_file",
            "bval_file",
            "bvec_file",
            "inverse_dwi_file",
            "inverse_bval_file",
            "inverse_bvec_file",
            "phase_encoding_direction",
            "rm_cerebellum",
            "gpu",
            "keep_temp",
            "extract_brain",
            "shell_dimension",
            "PED",
            "linear",
            "nonlinear"
        ]:
            if getattr(args, param.replace("-", "_"), None):
                config[param] = getattr(args, param.replace("-", "_"))

        # Add config parameters to command
        if len(config) > 0:
            cmd.append("--config")
        for key, value in config.items():
            cmd.extend([f"{key}={value}"])

        # Add config file if provided
        if args.config_file:
            cmd.extend(["--configfile", args.config_file])

        # Add other snakemake parameters
        if args.dry_run:
            cmd.append("-n")

        cmd.extend(["--cores", str(args.cores)])

        # Add any unknown arguments to pass to snakemake
        if unknown:
            cmd.extend(unknown)
        print(f"Executing: {' '.join(cmd)}")

        # Execute the snakemake command
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running snakemake: {e}")
            sys.exit(1)

    elif args.command == "synthseg":
        # Prepare arguments for SynthSeg
        synthseg_args = []
        for arg_name, arg_value in vars(args).items():
            if arg_name != "command" and arg_value is not None:
                if isinstance(arg_value, bool):
                    if arg_value:
                        synthseg_args.append(f"--{arg_name}")
                elif isinstance(arg_value, list):
                    synthseg_args.append(f"--{arg_name}")
                    synthseg_args.extend([str(x) for x in arg_value])
                else:
                    synthseg_args.append(f"--{arg_name}")
                    synthseg_args.append(str(arg_value))
        try:
            print(f"Running SynthSeg brain segmentation on {args.i}...")
            subprocess.run(
                ["python", "-m", "micaflow.scripts.synthseg"] + synthseg_args,
                check=True,
            )
            print(f"Brain segmentation completed. Output saved to {args.o}")
        except subprocess.CalledProcessError as e:
            print(f"Error running brain segmentation: {e}")
            sys.exit(1)

    elif args.command == "apply_SDC":
        # Prepare arguments for apply_SDC
        apply_sdc_parser = []
        for arg_name, arg_value in vars(args).items():
            if arg_name != "command" and arg_value is not None:
                if isinstance(arg_value, bool):
                    if arg_value:
                        apply_sdc_parser.append(f"--{arg_name}")
                elif isinstance(arg_value, list):
                    apply_sdc_parser.append(f"--{arg_name}")
                    apply_sdc_parser.extend([str(x) for x in arg_value])
                else:
                    apply_sdc_parser.append(f"--{arg_name}")
                    apply_sdc_parser.append(str(arg_value))

        # Run the apply_SDC script
        try:
            print(f"Applying susceptibility distortion correction to {args.input}...")
            subprocess.run(
                ["python", "-m", "micaflow.scripts.apply_SDC"] + apply_sdc_parser, check=True
            )
            print(
                f"Susceptibility distortion correction completed. Output saved to {args.output}"
            )
        except subprocess.CalledProcessError as e:
            print(f"Error applying susceptibility distortion correction: {e}")
            sys.exit(1)

    elif args.command == "apply_warp":
        # Prepare arguments for apply_warp
        apply_warp_args = []
        for arg_name, arg_value in vars(args).items():
            if arg_name != "command" and arg_value is not None:
                arg_name_formatted = arg_name.replace("_", "-")
                if isinstance(arg_value, bool):
                    if arg_value:
                        apply_warp_args.append(f"--{arg_name_formatted}")
                elif isinstance(arg_value, list):
                    apply_warp_args.append(f"--{arg_name_formatted}")
                    apply_warp_args.extend([str(x) for x in arg_value])
                else:
                    apply_warp_args.append(f"--{arg_name_formatted}")
                    apply_warp_args.append(str(arg_value))

        try:
            print(f"Applying warp transformation to {args.moving}...")
            print(len(apply_warp_args))
            subprocess.run(
                ["python", "-m", "micaflow.scripts.apply_warp"] + apply_warp_args,
                check=True,
            )
            print(f"Warp transformation completed. Output saved to {args.output}")
        except subprocess.CalledProcessError as e:
            print(f"Error applying warp transformation: {e}")
            sys.exit(1)

    elif args.command == "bet":
        # Prepare arguments for bet
        bet_args = []
        for arg_name, arg_value in vars(args).items():
            if arg_name != "command" and arg_value is not None:
                # Convert underscores to hyphens in argument names for CLI compatibility
                # This ensures --output_mask becomes --output-mask when passed to the script
                arg_name_formatted = arg_name.replace("_", "-")

                if isinstance(arg_value, bool):
                    if arg_value:
                        bet_args.append(f"--{arg_name_formatted}")
                elif isinstance(arg_value, list):
                    bet_args.append(f"--{arg_name_formatted}")
                    bet_args.extend([str(x) for x in arg_value])
                else:
                    bet_args.append(f"--{arg_name_formatted}")
                    bet_args.append(str(arg_value))

        try:
            print(f"Running brain extraction on {args.input}...")
            subprocess.run(
                ["python", "-m", "micaflow.scripts.bet"] + bet_args, check=True
            )
            print(f"Brain extraction completed. Output saved to {args.output}")
        except subprocess.CalledProcessError as e:
            print(f"Error running brain extraction: {e}")
            sys.exit(1)

    elif args.command == "bias_correction":
        # Prepare arguments for bias_correction
        bias_corr_args = []
        for arg_name, arg_value in vars(args).items():
            if arg_name != "command" and arg_value is not None:
                arg_name_formatted = arg_name.replace("_", "-")
                if isinstance(arg_value, bool):
                    if arg_value:
                        bias_corr_args.append(f"--{arg_name_formatted}")
                elif isinstance(arg_value, list):
                    bias_corr_args.append(f"--{arg_name_formatted}")
                    bias_corr_args.extend([str(x) for x in arg_value])
                else:
                    bias_corr_args.append(f"--{arg_name_formatted}")
                    bias_corr_args.append(str(arg_value))

        # Run the bias_correction script
        try:
            print(f"Running bias field correction on {args.input}...")
            subprocess.run(
                ["python", "-m", "micaflow.scripts.bias_correction"] + bias_corr_args,
                check=True,
            )
            print(f"Bias correction completed. Output saved to {args.output}")
        except subprocess.CalledProcessError as e:
            print(f"Error running bias correction: {e}")
            sys.exit(1)

    elif args.command == "calculate_dice":
        # Prepare arguments for calculate_dice
        dice_args = []
        for arg_name, arg_value in vars(args).items():
            if arg_name != "command" and arg_value is not None:
                if isinstance(arg_value, bool):
                    if arg_value:
                        dice_args.append(f"--{arg_name}")
                elif isinstance(arg_value, list):
                    dice_args.append(f"--{arg_name}")
                    dice_args.extend([str(x) for x in arg_value])
                else:
                    dice_args.append(f"--{arg_name}")
                    dice_args.append(str(arg_value))

        # Run the calculate_dice script
        try:
            print(f"Calculating DICE between {args.input} and {args.reference}...")
            subprocess.run(
                ["python", "-m", "micaflow.scripts.calculate_dice"] + dice_args,
                check=True,
            )
            if args.output:
                print(f"Results saved to {args.output}")
        except subprocess.CalledProcessError as e:
            print(f"Error calculating DICE: {e}")
            sys.exit(1)

    elif args.command == "compute_fa_md":
        # Prepare arguments for compute_fa_md
        compute_fa_md_args = []
        for arg_name, arg_value in vars(args).items():
            if arg_name != "command" and arg_value is not None:
                arg_name_formatted = arg_name.replace("_", "-")

                if isinstance(arg_value, bool):
                    if arg_value:
                        compute_fa_md_args.append(f"--{arg_name_formatted}")
                elif isinstance(arg_value, list):
                    compute_fa_md_args.append(f"--{arg_name_formatted}")
                    compute_fa_md_args.extend([str(x) for x in arg_value])
                else:
                    compute_fa_md_args.append(f"--{arg_name_formatted}")
                    compute_fa_md_args.append(str(arg_value))

        # Run the compute_fa_md script
        try:
            print(f"Computing FA and MD maps from {args.input}...")
            subprocess.run(
                ["python", "-m", "micaflow.scripts.compute_fa_md"] + compute_fa_md_args,
                check=True,
            )
            print(
                f"DTI metrics computed. FA saved to {args.output_fa}, MD saved to {args.output_fa}"
            )
        except subprocess.CalledProcessError as e:
            print(f"Error computing FA/MD maps: {e}")
            sys.exit(1)

    elif args.command == "coregister":
        # Prepare arguments for coregister
        coreg_args = []
        for arg_name, arg_value in vars(args).items():
            if arg_name != "command" and arg_value is not None:
                # Convert hyphens to underscores for the script
                arg_name_formatted = arg_name.replace("_", "-")

                if isinstance(arg_value, bool):
                    if arg_value:
                        coreg_args.append(f"--{arg_name_formatted}")
                elif isinstance(arg_value, list):
                    coreg_args.append(f"--{arg_name_formatted}")
                    coreg_args.extend([str(x) for x in arg_value])
                else:
                    coreg_args.append(f"--{arg_name_formatted}")
                    coreg_args.append(str(arg_value))

        # Run the coregister script
        try:
            print(f"Coregistering {args.moving_file} to {args.fixed_file}...")
            subprocess.run(
                ["python", "-m", "micaflow.scripts.coregister"] + coreg_args, check=True
            )
            print(f"Coregistration completed. Output saved to {args.output}")
            if args.warp_file:
                print(f"Warp field saved to {args.warp_file}")
            if args.affine_file:
                print(f"Affine transformation matrix saved to {args.affine_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error during coregistration: {e}")
            sys.exit(1)

    elif args.command == "denoise":
        # Prepare arguments for denoise
        denoise_args = []
        for arg_name, arg_value in vars(args).items():
            if arg_name != "command" and arg_value is not None:
                arg_name_formatted = arg_name.replace("_", "-")
                if isinstance(arg_value, bool):
                    if arg_value:
                        denoise_args.append(f"--{arg_name_formatted}")
                elif isinstance(arg_value, list):
                    denoise_args.append(f"--{arg_name_formatted}")
                    denoise_args.extend([str(x) for x in arg_value])
                else:
                    denoise_args.append(f"--{arg_name_formatted}")
                    denoise_args.append(str(arg_value))

        # Run the denoise script
        try:
            print(f"Denoising diffusion image {args.input}...")
            subprocess.run(
                ["python", "-m", "micaflow.scripts.denoise"] + denoise_args, check=True
            )
            print(f"Denoising completed. Output saved to {args.output}")
        except subprocess.CalledProcessError as e:
            print(f"Error during denoising: {e}")
            sys.exit(1)

    elif args.command == "motion_correction":
        # Prepare arguments for motion_correction
        motion_corr_args = []
        for arg_name, arg_value in vars(args).items():
            if arg_name != "command" and arg_value is not None:
                arg_name_formatted = arg_name.replace("_", "-")
                if isinstance(arg_value, bool):
                    if arg_value:
                        motion_corr_args.append(f"--{arg_name_formatted}")
                elif isinstance(arg_value, list):
                    motion_corr_args.append(f"--{arg_name_formatted}")
                    motion_corr_args.extend([str(x) for x in arg_value])
                else:
                    motion_corr_args.append(f"--{arg_name_formatted}")
                    motion_corr_args.append(str(arg_value))

        # Run the motion_correction script
        try:
            print(f"Performing motion correction on {args.denoised}...")
            subprocess.run(
                ["python", "-m", "micaflow.scripts.motion_correction"]
                + motion_corr_args,
                check=True,
            )
            print(f"Motion correction completed. Output saved to {args.output}")
        except subprocess.CalledProcessError as e:
            print(f"Error during motion correction: {e}")
            sys.exit(1)

    elif args.command == "SDC":
        # Prepare arguments for SDC
        sdc_args = []
        for arg_name, arg_value in vars(args).items():
            if arg_name != "command" and arg_value is not None:
                # Convert hyphens to underscores for the script
                arg_name_formatted = arg_name.replace("_", "-")

                if isinstance(arg_value, bool):
                    if arg_value:
                        sdc_args.append(f"--{arg_name_formatted}")
                elif isinstance(arg_value, list):
                    sdc_args.append(f"--{arg_name_formatted}")
                    sdc_args.extend([str(x) for x in arg_value])
                else:
                    sdc_args.append(f"--{arg_name_formatted}")
                    sdc_args.append(str(arg_value))

        # Run the SDC script
        try:
            print(f"Running susceptibility distortion correction on {args.input}...")
            subprocess.run(
                ["python", "-m", "micaflow.scripts.SDC"] + sdc_args, check=True
            )
            print(
                f"Susceptibility distortion correction completed. Output saved to {args.output}"
            )
            print(f"Warp field saved to {args.output_warp}")
        except subprocess.CalledProcessError as e:
            print(f"Error running susceptibility distortion correction: {e}")
            sys.exit(1)

    elif args.command == "texture_generation":
        # Prepare arguments for texture_generation
        texture_args = []
        for arg_name, arg_value in vars(args).items():
            if arg_name != "command" and arg_value is not None:
                if isinstance(arg_value, bool):
                    if arg_value:
                        texture_args.append(f"--{arg_name}")
                elif isinstance(arg_value, list):
                    texture_args.append(f"--{arg_name}")
                    texture_args.extend([str(x) for x in arg_value])
                else:
                    texture_args.append(f"--{arg_name}")
                    texture_args.append(str(arg_value))

        # Run the texture_generation script
        try:
            print(
                f"Generating texture features for {args.input} using mask {args.mask}..."
            )
            subprocess.run(
                ["python", "-m", "micaflow.scripts.texture_generation"] + texture_args,
                check=True,
            )
            print(f"Texture generation completed. Output saved to {args.output}")
        except subprocess.CalledProcessError as e:
            print(f"Error during texture generation: {e}")
            sys.exit(1)
    elif args.command == "normalize":
        # Prepare arguments for motion_correction
        normalize_args = []
        for arg_name, arg_value in vars(args).items():
            if arg_name != "command" and arg_value is not None:
                arg_name_formatted = arg_name.replace("_", "-")
               
                if isinstance(arg_value, bool):
                    if arg_value:
                        normalize_args.append(f"--{arg_name_formatted}")
                elif isinstance(arg_value, list):
                    normalize_args.append(f"--{arg_name_formatted}")
                    normalize_args.extend([str(x) for x in arg_value])
                else:
                    normalize_args.append(f"--{arg_name_formatted}")
                    normalize_args.append(str(arg_value))

        # Run the motion_correction script
        try:
            print(f"Performing motion correction on {args.input}...")
            subprocess.run(
                ["python", "-m", "micaflow.scripts.normalize"] + normalize_args,
                check=True,
            )
            print(f"Motion correction completed. Output saved to {args.output}")
        except subprocess.CalledProcessError as e:
            print(f"Error during motion correction: {e}")
            sys.exit(1)

    elif args.command == "synth_b0":
        # Prepare arguments for synth_b0
        synth_b0_args = []
        for arg_name, arg_value in vars(args).items():
            if arg_name != "command" and arg_value is not None:
                # Convert underscores to hyphens in argument names
                arg_name_formatted = arg_name.replace("_", "-")

                if isinstance(arg_value, bool):
                    if arg_value:
                        synth_b0_args.append(f"--{arg_name_formatted}")
                elif isinstance(arg_value, list):
                    synth_b0_args.append(f"--{arg_name_formatted}")
                    synth_b0_args.extend([str(x) for x in arg_value])
                else:
                    synth_b0_args.append(f"--{arg_name_formatted}")
                    synth_b0_args.append(str(arg_value))

        # Run the synth_b0 script
        try:
            print(f"Generating synthetic B0 using T1w ({args.t1}) and B0 ({args.b0}) inputs...")
            print(f"First performing linear registration between B0 and T1w...")
            subprocess.run(
                ["python", "-m", "micaflow.scripts.synth_b0"] + synth_b0_args, check=True
            )
            print(f"Synthetic B0 generation completed. Output saved to {args.output}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating synthetic B0: {e}")
            sys.exit(1)

    elif args.command == "extract_b0":
        # Prepare arguments for extract_b0
        extract_b0_args = []
        for arg_name, arg_value in vars(args).items():
            if arg_name != "command" and arg_value is not None:
                arg_name_formatted = arg_name.replace("_", "-")
                if isinstance(arg_value, bool):
                    if arg_value:
                        extract_b0_args.append(f"--{arg_name_formatted}")
                elif isinstance(arg_value, list):
                    extract_b0_args.append(f"--{arg_name_formatted}")
                    extract_b0_args.extend([str(x) for x in arg_value])
                else:
                    extract_b0_args.append(f"--{arg_name_formatted}")
                    extract_b0_args.append(str(arg_value))

        # Run the extract_b0 script
        try:
            print(f"Extracting b=0 volume from {args.input}...")
            subprocess.run(
                ["python", "-m", "micaflow.scripts.extract_b0"] + extract_b0_args,
                check=True,
            )
            print(f"B0 extraction completed. Output saved to {args.output}")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting b=0 volume: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
