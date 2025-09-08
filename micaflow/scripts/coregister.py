"""
coregister - Label-Augmented Image Registration for Aligning Neuroimaging Data

Part of the micaflow processing pipeline for neuroimaging data.

This module performs comprehensive image registration between two images using LAMAReg
(Label-Augmented Modality-Agnostic Registration), which combines anatomical image information
with segmentation labels to achieve more accurate registration across different imaging modalities. 
The registration aligns a moving image with a fixed reference space, enabling spatial normalization
of neuroimaging data for group analysis, multimodal integration, or atlas-based analyses.

Features:
--------
- Label-augmented registration for improved accuracy across different modalities
- Automatic segmentation generation using SynthSeg when not provided
- Combined rigid, affine, and SyN nonlinear registration in one step
- Bidirectional transformation capability (forward and inverse)
- Option to save all transformation components for later application
- Modality-agnostic approach for consistent registration across different contrasts
- Support for multi-threaded processing for both ANTs and SynthSeg components

API Usage:
---------
micaflow coregister
    --fixed-file <path/to/reference.nii.gz>
    --moving-file <path/to/source.nii.gz>
    --output <path/to/registered.nii.gz>
    [--fixed-segmentation <path/to/fixed_seg.nii.gz>]
    [--moving-segmentation <path/to/moving_seg.nii.gz>]
    [--warp-file <path/to/warp.nii.gz>]
    [--affine-file <path/to/affine.mat>]
    [--rev-warp-file <path/to/reverse_warp.nii.gz>]
    [--rev-affine-file <path/to/reverse_affine.mat>]
    [--ants-threads <int>]
    [--synthseg-threads <int>]

Python Usage:
-----------
>>> from lamareg.scripts.lamar import lamareg
>>> lamareg(
...     input_image="subject_t1w.nii.gz",
...     reference_image="mni152.nii.gz",
...     output_image="registered_t1w.nii.gz",
...     input_parc="subject_seg.nii.gz",  # Optional - will be generated if not provided
...     reference_parc="mni152_seg.nii.gz",  # Optional - will be generated if not provided
...     output_parc="registered_seg.nii.gz",
...     warp_file="warp.nii.gz",
...     affine_file="affine.mat",
...     inverse_warp_file="reverse_warp.nii.gz",
...     inverse_affine_file="reverse_affine.mat",
...     skip_moving_parc=False,  # Set to True if input_parc is provided
...     skip_fixed_parc=False,   # Set to True if reference_parc is provided
...     ants_threads=4,
...     synthseg_threads=2
... )

"""

import argparse
import sys
from colorama import init, Fore, Style
init()


def print_help_message():
    """Print a help message with examples."""
    # ANSI color codes
    CYAN = Fore.CYAN
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    BLUE = Fore.BLUE
    MAGENTA = Fore.MAGENTA
    BOLD = Style.BRIGHT
    RESET = Style.RESET_ALL

    help_text = f"""
    {CYAN}{BOLD}╔════════════════════════════════════════════════════════════════╗
    ║                LABEL-AUGMENTED COREGISTRATION                  ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script performs label-augmented modality-agnostic registration (LAMAReg) 
    between two images. The registration aligns the moving image to match the fixed 
    reference image space, utilizing segmentation labels to improve accuracy across 
    different modalities.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow coregister {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--fixed-file{RESET}   : Path to the fixed/reference image (.nii.gz)
      {YELLOW}--moving-file{RESET}  : Path to the moving image to be registered (.nii.gz)
      {YELLOW}--output{RESET}       : Output path for the registered image (.nii.gz)
    
    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--fixed-segmentation{RESET}  : Path to fixed image segmentation (.nii.gz)
                           If not provided, will be generated automatically
      {YELLOW}--moving-segmentation{RESET} : Path to moving image segmentation (.nii.gz)
                           If not provided, will be generated automatically
      {YELLOW}--warp-file{RESET}      : Path to save the forward warp field (.nii.gz)
      {YELLOW}--affine-file{RESET}    : Path to save the forward affine transform (.mat)
      {YELLOW}--rev-warp-file{RESET}  : Path to save the reverse warp field (.nii.gz)
      {YELLOW}--rev-affine-file{RESET}: Path to save the reverse affine transform (.mat)
      {YELLOW}--ants-threads{RESET}   : Number of threads for ANTs operations (default: 1)
      {YELLOW}--synthseg-threads{RESET}: Number of threads for SynthSeg operations (default: 1)
    
    {CYAN}{BOLD}────────────────── EXAMPLE USAGE ────────────────────────{RESET}
    
    {BLUE}# Basic registration with automatic segmentation generation{RESET}
    micaflow coregister {YELLOW}--fixed-file{RESET} mni152.nii.gz {YELLOW}--moving-file{RESET} subject_t1w.nii.gz \\
      {YELLOW}--output{RESET} registered_t1w.nii.gz {YELLOW}--warp-file{RESET} warp.nii.gz {YELLOW}--affine-file{RESET} affine.mat
    
    {BLUE}# Registration with provided segmentation images{RESET}
    micaflow coregister {YELLOW}--fixed-file{RESET} mni152.nii.gz {YELLOW}--moving-file{RESET} subject_t1w.nii.gz \\
      {YELLOW}--fixed-segmentation{RESET} mni152_seg.nii.gz {YELLOW}--moving-segmentation{RESET} subject_seg.nii.gz \\
      {YELLOW}--output{RESET} registered_t1w.nii.gz
    
    {BLUE}# Multi-threaded registration{RESET}
    micaflow coregister {YELLOW}--fixed-file{RESET} mni152.nii.gz {YELLOW}--moving-file{RESET} subject_t1w.nii.gz \\
      {YELLOW}--output{RESET} registered_t1w.nii.gz {YELLOW}--ants-threads{RESET} 4 {YELLOW}--synthseg-threads{RESET} 2
    
    {CYAN}{BOLD}────────────────────────── NOTES ───────────────────────{RESET}
    {MAGENTA}•{RESET} LAMAReg combines anatomical and label information for robust registration
    {MAGENTA}•{RESET} Segmentations are automatically generated using SynthSeg if not provided
    {MAGENTA}•{RESET} Forward transforms convert from moving space to fixed space
    {MAGENTA}•{RESET} Reverse transforms convert from fixed space to moving space
    {MAGENTA}•{RESET} Output segmentation will be saved alongside the registered image
    {MAGENTA}•{RESET} Use threading options to speed up processing on multi-core systems
    """
    print(help_text)


if __name__ == "__main__":
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Run label-augmented modality-agnostic registration (LAMAReg) between two images."
    )
    parser.add_argument("--fixed-file", required=True, 
                        help="Path to the fixed/reference image.")
    parser.add_argument("--moving-file", required=True, 
                        help="Path to the moving image to be registered.")
    parser.add_argument("--fixed-segmentation", 
                        help="Path to the fixed segmentation image. If not provided, it will be generated automatically.")
    parser.add_argument("--moving-segmentation", 
                        help="Path to the moving segmentation image. If not provided, it will be generated automatically.")
    parser.add_argument("--output", required=True,
                        help="Output path for the registered image.")
    parser.add_argument("--warp-file", default=None, 
                        help="Optional path to save the forward warp field (moving to fixed).")
    parser.add_argument("--affine-file", default=None,
                        help="Optional path to save the forward affine transform (moving to fixed).")
    parser.add_argument("--rev-warp-file", default=None,
                        help="Optional path to save the reverse warp field (fixed to moving).")
    parser.add_argument("--rev-affine-file", default=None,
                        help="Optional path to save the reverse affine transform (fixed to moving).")
    parser.add_argument("--ants-threads", type=int, default=1, 
                        help="Number of threads for ANTs registration operations (default: 1).")
    parser.add_argument("--synthseg-threads", type=int, default=1, 
                        help="Number of threads for SynthSeg segmentation operations (default: 1).")
    args = parser.parse_args()

    if args.fixed_segmentation and args.moving_segmentation:
        print("Using previously generated segmentation images.")
        from lamareg.scripts.lamar import lamareg
        
        lamareg(
            input_image=args.moving_file,
            reference_image=args.fixed_file,
            output_image=args.output,
            input_parc=args.moving_segmentation,
            reference_parc=args.fixed_segmentation,
            output_parc=args.output.replace('.nii.gz', '_parc.nii.gz'),
            affine_file=args.affine_file,
            warp_file=args.warp_file,
            inverse_warp_file=args.rev_warp_file,
            inverse_affine_file=args.rev_affine_file,
            skip_moving_parc=True,
            skip_fixed_parc=True,
            skip_qc=True,
            ants_threads=args.ants_threads,
            synthseg_threads=args.synthseg_threads
        )
    else:
        print("No segmentation images provided. Segmentations will be generated.")
        from lamareg.scripts.lamar import lamareg
        
        # Generate paths for segmentations
        moving_segmentation = args.moving_file.replace('.nii.gz', '_parc.nii.gz')
        fixed_segmentation = args.fixed_file.replace('.nii.gz', '_parc.nii.gz')
        output_segmentation = args.output.replace('.nii.gz', '_parc.nii.gz')
        
        lamareg(
            input_image=args.moving_file,
            reference_image=args.fixed_file,
            output_image=args.output,
            input_parc=moving_segmentation,
            reference_parc=fixed_segmentation,
            output_parc=output_segmentation,
            affine_file=args.affine_file,
            warp_file=args.warp_file,
            inverse_warp_file=args.rev_warp_file,
            inverse_affine_file=args.rev_affine_file,
            skip_moving_parc=False,
            skip_fixed_parc=False,
            skip_qc=True,
            ants_threads=args.ants_threads,
            synthseg_threads=args.synthseg_threads
        )