#!/usr/bin/env python3
import argparse
import sys
import pkg_resources
import subprocess
from colorama import init, Fore, Style

init()


def get_snakefile_path():
    """Get the path to the Snakefile within the installed package."""
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
    
    {CYAN}{BOLD}─────────────────── AVAILABLE COMMANDS ───────────────────{RESET}
      {GREEN}pipeline{RESET}          : Run the full processing pipeline (default)
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
      {YELLOW}--t1w-file{RESET} T1W_FILE            T1-weighted image file
    
    {CYAN}{BOLD}──────────────── PIPELINE OPTIONAL PARAMETERS ────────────{RESET}
      {YELLOW}--data-directory{RESET} DATA_DIR      Input data directory
      {YELLOW}--session{RESET} SESSION_ID           Session ID (default: none)
      {YELLOW}--flair-file{RESET} FLAIR_FILE        FLAIR image file
      {YELLOW}--dwi-file{RESET} DWI_FILE            Diffusion weighted image
      {YELLOW}--bval-file{RESET} BVAL_FILE          B-value file for DWI
      {YELLOW}--bvec-file{RESET} BVEC_FILE          B-vector file for DWI
      {YELLOW}--inverse-dwi-file{RESET} INV_FILE    Inverse (PA) DWI for distortion correction
      {YELLOW}--gpu{RESET}                          Use GPU for computation
      {YELLOW}--cores{RESET} N                      Number of CPU cores to use (default: 1)
      {YELLOW}--dry-run{RESET}, {YELLOW}-n{RESET}                  Dry run (don't execute commands)
      {YELLOW}--config-file{RESET} FILE             Path to a YAML configuration file
      {YELLOW}--extract-brain{RESET}                Generate brain-extracted versions of all outputs in a dedicated directory
      {YELLOW}--keep-temp{RESET}                    Keep temporary processing files (useful for debugging)
      {YELLOW}--rm-cerebellum{RESET}                Remove cerebellum from brain extraction outputs
    
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
    # If no arguments provided, show help and exit
    if len(sys.argv) == 1:
        print(print_extended_help())
        sys.exit(0)

    # Intercept help requests for subcommands before argparse processes them

    if (len(sys.argv) >= 3 and sys.argv[2] in ["-h", "--help"]) or len(sys.argv) == 2:
        # Check if the first argument is a valid command
        command = sys.argv[1]
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
    pipeline_parser.add_argument(
        "--gpu", action="store_true", help="Use GPU computation"
    )
    pipeline_parser.add_argument(
        "--threads", type=int, default=1, help="Number of threads to use"
    )
    pipeline_parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Dry run (don't execute commands)"
    )
    pipeline_parser.add_argument(
        "--cores", type=int, default=1, help="Number of CPU cores to use"
    )
    pipeline_parser.add_argument(
        "--config-file", help="Path to a YAML configuration file"
    )
    pipeline_parser.add_argument(
        "--snakemake-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to Snakemake",
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
    sdc_parser = subparsers.add_parser("apply_SDC")
    sdc_parser.add_argument(
        "--input",
        required=True,
        help="Path to the motion-corrected DWI image (.nii.gz)",
    )
    sdc_parser.add_argument(
        "--warp",
        required=True,
        help="Path to the warp field estimated from SDC (.nii.gz)",
    )
    sdc_parser.add_argument(
        "--affine",
        required=True,
        help="Path to an image from which to extract the affine matrix",
    )
    sdc_parser.add_argument(
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
        required=True,
        help="Path to the warp field for non-linear transformation",
    )
    apply_warp_parser.add_argument(
        "--affine", required=True, help="Path to the affine transformation file"
    )
    apply_warp_parser.add_argument(
        "--output", required=True, help="Output path for the warped image"
    )
    apply_warp_parser.add_argument(
        "--interpolation",
        default="linear",
        help="Interpolation method (default: linear).",
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

    # Add this after the bias_corr_parser section but before the args.parse_args() call:

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
        "--affine-file", default=None,
        help="Optional path to save the forward affine transform (moving to fixed) (.mat)"
    )
    coreg_parser.add_argument(
        "--rev-warp-file", default=None,
        help="Optional path to save the reverse warp field (fixed to moving) (.nii.gz)"
    )
    coreg_parser.add_argument(
        "--rev-affine-file", default=None,
        help="Optional path to save the reverse affine transform (fixed to moving) (.mat)"
    )
    coreg_parser.add_argument(
        "--ants-threads", type=int, default=1, 
        help="Number of threads for ANTs registration operations (default: 1)"
    )
    coreg_parser.add_argument(
        "--synthseg-threads", type=int, default=1, 
        help="Number of threads for SynthSeg segmentation operations (default: 1)"
    )
    coreg_parser.add_argument(
        "--shell-channel", type=int,
        help="Index of the shell channel to extract from DWI images for registration"
    )
    coreg_parser.add_argument(
        "--b0-output", 
        help="Path to save the extracted shell volume when processing DWI data (.nii.gz)"
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

    # Motion Correction command
    motion_corr_parser = subparsers.add_parser(
        "motion_correction",
        help="Perform motion correction on diffusion-weighted images",
    )
    motion_corr_parser.add_argument(
        "--denoised", required=True, help="Path to the denoised DWI (NIfTI file)."
    )
    motion_corr_parser.add_argument(
        "--bval", required=True, help="Path to the b-values file (.bval)"
    )
    motion_corr_parser.add_argument(
        "--bvec", required=True, help="Path to the b-vectors file (.bvec)"
    )
    motion_corr_parser.add_argument(
        "--output", required=True, help="Output path for the motion-corrected DWI."
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

    args = parser.parse_args()

    # If no command is provided, default to pipeline
    if not args.command:
        args.command = "pipeline"

    if args.command == "pipeline":
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
            "threads",
            "rm_cerebellum",
            "gpu",
            "keep_temp",
            "extract_brain",
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

        # Add any additional snakemake arguments
        if args.snakemake_args:
            cmd.extend(args.snakemake_args)
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
        sdc_args = []
        for arg_name, arg_value in vars(args).items():
            if arg_name != "command" and arg_value is not None:
                if isinstance(arg_value, bool):
                    if arg_value:
                        sdc_args.append(f"--{arg_name}")
                elif isinstance(arg_value, list):
                    sdc_args.append(f"--{arg_name}")
                    sdc_args.extend([str(x) for x in arg_value])
                else:
                    sdc_args.append(f"--{arg_name}")
                    sdc_args.append(str(arg_value))

        # Run the apply_SDC script
        try:
            print(f"Applying susceptibility distortion correction to {args.input}...")
            subprocess.run(
                ["python", "-m", "micaflow.scripts.apply_SDC"] + sdc_args, check=True
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
                if isinstance(arg_value, bool):
                    if arg_value:
                        apply_warp_args.append(f"--{arg_name}")
                elif isinstance(arg_value, list):
                    apply_warp_args.append(f"--{arg_name}")
                    apply_warp_args.extend([str(x) for x in arg_value])
                else:
                    apply_warp_args.append(f"--{arg_name}")
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
                if isinstance(arg_value, bool):
                    if arg_value:
                        bias_corr_args.append(f"--{arg_name}")
                elif isinstance(arg_value, list):
                    bias_corr_args.append(f"--{arg_name}")
                    bias_corr_args.extend([str(x) for x in arg_value])
                else:
                    bias_corr_args.append(f"--{arg_name}")
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
                if isinstance(arg_value, bool):
                    if arg_value:
                        denoise_args.append(f"--{arg_name}")
                elif isinstance(arg_value, list):
                    denoise_args.append(f"--{arg_name}")
                    denoise_args.extend([str(x) for x in arg_value])
                else:
                    denoise_args.append(f"--{arg_name}")
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
                if isinstance(arg_value, bool):
                    if arg_value:
                        motion_corr_args.append(f"--{arg_name}")
                elif isinstance(arg_value, list):
                    motion_corr_args.append(f"--{arg_name}")
                    motion_corr_args.extend([str(x) for x in arg_value])
                else:
                    motion_corr_args.append(f"--{arg_name}")
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


if __name__ == "__main__":
    main()
