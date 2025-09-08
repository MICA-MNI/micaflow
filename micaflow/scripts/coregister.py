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
    parser.add_argument("--shell-channel", type=int,
                        help="Index of the shell channel to use for DWI images.")
    parser.add_argument("--b0-output", default=None,
                        help="Optional path to save the extracted shell volume when processing DWI data.")
    args = parser.parse_args()

    # Process DWI shell extraction if requested
    if args.shell_channel is not None and args.b0_output is not None:
        print(f"Extracting shell channel {args.shell_channel} from DWI image...")
        import ants
        
        # Read the full DWI image
        moving_image = ants.image_read(args.moving_file)
        
        # Check if the image is 4D (has multiple volumes)
        if moving_image.dimension == 4:
            # Get the number of volumes
            num_volumes = moving_image.components
            print(f"Found 4D image with {num_volumes} volumes")
            
            # Handle negative indexing for shell_channel
            if args.shell_channel < 0:
                # Convert negative index to positive (e.g. -1 -> last volume)
                adjusted_index = num_volumes + args.shell_channel
                # Check if still in valid range
                if adjusted_index < 0:
                    print(f"Warning: Adjusted shell channel {adjusted_index} is out of range. Using volume 0 instead.")
                    shell_index = 0
                else:
                    shell_index = adjusted_index
                    print(f"Using negative index {args.shell_channel} which maps to volume {shell_index}")
            else:
                # Make sure the requested shell channel is valid
                if args.shell_channel >= num_volumes:
                    print(f"Warning: Requested shell channel {args.shell_channel} exceeds available volumes. Using volume 0 instead.")
                    shell_index = 0
                else:
                    shell_index = args.shell_channel
            
            # Extract the specified volume
            print(f"Extracting volume {shell_index}")
            
            # Use proper ANTs API to extract a single volume
            import numpy as np
            # Create start and size arrays for indexing
            starts = [0, 0, 0, shell_index]
            sizes = list(moving_image.shape[:3]) + [1]  # Take one volume along 4th dimension
            
            # Extract the volume using ANTs getitem function
            extracted_array = moving_image.numpy()[..., shell_index]
            extracted_volume = ants.from_numpy(
                extracted_array,
                origin=moving_image.origin[:3],
                spacing=moving_image.spacing[:3],
                direction=moving_image.direction[:3,:3]
            )
            
            # Save the extracted volume
            ants.image_write(extracted_volume, args.b0_output)
            print(f"Saved extracted volume to: {args.b0_output}")
            
            # Use the extracted volume for registration
            args.moving_file = args.b0_output
        else:
            print("Warning: Shell channel specified but input is not a 4D image. Using original image.")

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