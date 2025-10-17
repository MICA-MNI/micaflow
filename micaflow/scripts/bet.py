"""
bet - Brain Extraction Tool

Part of the micaflow processing pipeline for neuroimaging data.

This module provides brain extraction (skull stripping) functionality using either:
1. SynthSeg-generated parcellations to create brain masks
2. User-provided binary masks

It accurately segments the brain from surrounding tissues in MR images and offers 
options for cerebellum removal.

Features:
--------
- Brain extraction based on SynthSeg parcellation or user-provided mask
- Optional cerebellum removal for specific analyses (FreeSurfer labels: 7, 8, 46, 47, 16, 15, 24)
- Produces skull-stripped images and binary brain masks
- Automatic resampling when input mask/parcellation dimensions don't match the input image

Command-Line Usage:
------------------
# Using SynthSeg parcellation:
micaflow bet \\
    --input <path/to/image.nii.gz> \\
    --output <path/to/brain.nii.gz> \\
    --parcellation <path/to/synthseg_parcellation.nii.gz> \\
    --output-mask <path/to/brain_mask.nii.gz> \\
    [--remove-cerebellum]

# Using pre-computed mask:
micaflow bet \\
    --input <path/to/image.nii.gz> \\
    --output <path/to/brain.nii.gz> \\
    --input-mask <path/to/mask.nii.gz>

Note: Either --parcellation OR --input-mask must be provided.

Python API Usage (subprocess):
-----------------------------
>>> import subprocess
>>> 
>>> # Using SynthSeg parcellation
>>> result = subprocess.run([
...     "micaflow", "bet",
...     "--input", "t1w.nii.gz",
...     "--output", "t1w_brain.nii.gz",
...     "--output-mask", "brain_mask.nii.gz",
...     "--parcellation", "synthseg_parcellation.nii.gz",
...     "--remove-cerebellum"
... ], check=True)
>>> 
>>> # Using pre-computed mask
>>> result = subprocess.run([
...     "micaflow", "bet",
...     "--input", "t1w.nii.gz",
...     "--output", "t1w_brain.nii.gz",
...     "--input-mask", "brain_mask.nii.gz"
... ], check=True)

Python API Usage (direct):
-------------------------
>>> import nibabel as nib
>>> from nilearn.image import resample_to_img
>>> import numpy as np
>>> 
>>> # Load images
>>> input_img = nib.load("t1w.nii.gz")
>>> parcellation_img = nib.load("synthseg_parc.nii.gz")
>>> 
>>> # Resample parcellation to match input
>>> resampled_parc = resample_to_img(parcellation_img, input_img, interpolation="nearest")
>>> parc_data = resampled_parc.get_fdata()
>>> 
>>> # Create brain mask (exclude background label 0)
>>> mask = parc_data > 0
>>> 
>>> # Optional: Remove cerebellum
>>> cerebellum_labels = [7, 8, 46, 47, 16, 15, 24]
>>> for label in cerebellum_labels:
...     mask = mask & (parc_data != label)
>>> 
>>> # Apply mask
>>> brain_data = input_img.get_fdata()
>>> brain_data[~mask] = 0
>>> 
>>> # Save results
>>> brain_img = nib.Nifti1Image(brain_data, input_img.affine)
>>> brain_img.to_filename("t1w_brain.nii.gz")
>>> 
>>> mask_img = nib.Nifti1Image(mask.astype(np.int8), input_img.affine)
>>> mask_img.to_filename("brain_mask.nii.gz")

Exit Codes:
----------
0 : Success
1 : Error (missing required arguments, invalid inputs, or processing failure)

FreeSurfer Label Reference:
--------------------------
Cerebellum and related structures (removed with --remove-cerebellum):
  7  : Left-Cerebellum-White-Matter
  8  : Left-Cerebellum-Cortex
  46 : Right-Cerebellum-White-Matter
  47 : Right-Cerebellum-Cortex
  16 : Brain-Stem
  15 : 4th-Ventricle
  24 : CSF

See Also:
--------
- synthseg : For generating the input parcellation
- For more on FreeSurfer labels: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT

References:
----------
1. Billot B, Greve DN, Puonti O, et al. SynthSeg: Segmentation of brain MRI scans of 
   any contrast and resolution without retraining. Medical Image Analysis. 2023;86:102789.
   doi:10.1016/j.media.2023.102789
"""

import subprocess
import argparse
import os
import shutil
import sys
from colorama import init, Fore, Style
import nibabel as nib
from nilearn.image import resample_to_img
import numpy as np

init()


def print_help_message():
    """Print comprehensive help message with usage examples and notes."""
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
    ║                     BRAIN EXTRACTION TOOL                      ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script performs brain extraction (skull stripping) on MRI images using
    either SynthSeg parcellations or user-provided masks.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow bet {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--input{RESET}, {YELLOW}-i{RESET}      : Path to the input MR image (.nii.gz)
      {YELLOW}--output{RESET}, {YELLOW}-o{RESET}     : Path for the output brain-extracted image (.nii.gz)
      
      {MAGENTA}Note: Must provide ONE of the following:{RESET}
      {YELLOW}--parcellation{RESET}, {YELLOW}-p{RESET} : Path to SynthSeg parcellation file (.nii.gz)
      {YELLOW}--input-mask{RESET}        : Path to binary brain mask (.nii.gz)
    
    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--output-mask{RESET}, {YELLOW}-m{RESET}   : Path for the output brain mask (.nii.gz)
                         {MAGENTA}Only used when --parcellation is provided{RESET}
      {YELLOW}--remove-cerebellum{RESET}, {YELLOW}-r{RESET}: Remove cerebellum from the brain mask
                         {MAGENTA}Only works with --parcellation mode{RESET}
                         {MAGENTA}Removes FreeSurfer labels: 7, 8, 46, 47, 16, 15, 24{RESET}
    
    {CYAN}{BOLD}────────────────── EXAMPLE USAGE ────────────────────────{RESET}
    
    {BLUE}# Mode 1: Using SynthSeg parcellation{RESET}
    micaflow bet {YELLOW}--input{RESET} t1w.nii.gz {YELLOW}--output{RESET} t1w_brain.nii.gz \\
      {YELLOW}--parcellation{RESET} synthseg_parc.nii.gz {YELLOW}--output-mask{RESET} brain_mask.nii.gz
    
    {BLUE}# Mode 2: Using SynthSeg parcellation with cerebellum removal{RESET}
    micaflow bet {YELLOW}--input{RESET} t1w.nii.gz {YELLOW}--output{RESET} t1w_brain.nii.gz \\
      {YELLOW}--parcellation{RESET} synthseg_parc.nii.gz {YELLOW}--output-mask{RESET} brain_mask.nii.gz \\
      {YELLOW}--remove-cerebellum{RESET}
    
    {BLUE}# Mode 3: Using pre-computed mask{RESET}
    micaflow bet {YELLOW}--input{RESET} flair.nii.gz {YELLOW}--output{RESET} flair_brain.nii.gz \\
      {YELLOW}--input-mask{RESET} precomputed_mask.nii.gz
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    {MAGENTA}•{RESET} Either --parcellation OR --input-mask must be provided (not both)
    {MAGENTA}•{RESET} When using --parcellation:
      - Creates brain mask from all non-zero parcellation labels
      - Can optionally save the mask with --output-mask
      - Can remove cerebellum regions with --remove-cerebellum
    {MAGENTA}•{RESET} When using --input-mask:
      - Directly applies the provided binary mask
      - --output-mask and --remove-cerebellum options are ignored
    {MAGENTA}•{RESET} Automatic resampling is performed if mask/parcellation doesn't match input image
    {MAGENTA}•{RESET} Resampling uses nearest neighbor interpolation to preserve discrete labels
    {MAGENTA}•{RESET} Cerebellum labels (FreeSurfer): 
      7=L-Cereb-WM, 8=L-Cereb-Cortex, 46=R-Cereb-WM, 47=R-Cereb-Cortex, 
      16=Brain-Stem, 15=4th-Ventricle, 24=CSF
    
    {CYAN}{BOLD}──────────────────── EXIT CODES ─────────────────────────{RESET}
    {GREEN}0{RESET} : Success - brain extraction completed
    {YELLOW}1{RESET} : Error - invalid arguments or processing failure
    """
    print(help_text)


if __name__ == "__main__":
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Perform brain extraction using SynthSeg parcellation or binary mask",
        add_help=False  # We use custom help
    )
    parser.add_argument("--input", "-i", required=True, help="Input MR image file")
    parser.add_argument("--input-mask", help="Input binary brain mask image file (optional, mutually exclusive with --parcellation)")
    parser.add_argument(
        "--output", "-o", required=True, help="Output brain-extracted image file"
    )
    parser.add_argument(
        "--output-mask",
        "-m",
        help="Output brain mask file (only used with --parcellation mode)",
    )
    parser.add_argument(
        "--parcellation",
        "-p",
        help="SynthSeg parcellation file (optional, mutually exclusive with --input-mask)",
    )
    parser.add_argument(
        "--remove-cerebellum",
        "-r",
        action="store_true",
        help="Remove cerebellum from brain mask (only works with --parcellation mode)",
    )

    args = parser.parse_args()
    
    # Validate that either input-mask or parcellation is provided
    if not args.input_mask and not args.parcellation:
        print(f"{Fore.RED}Error: Must provide either --input-mask or --parcellation{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Run 'micaflow bet --help' for usage information{Style.RESET_ALL}")
        sys.exit(1)
    
    if args.input_mask and args.parcellation:
        print(f"{Fore.RED}Error: Cannot use both --input-mask and --parcellation{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Run 'micaflow bet --help' for usage information{Style.RESET_ALL}")
        sys.exit(1)
    
    # Warn if incompatible options are used
    if args.input_mask and args.remove_cerebellum:
        print(f"{Fore.YELLOW}Warning: --remove-cerebellum is ignored when using --input-mask{Style.RESET_ALL}")
    
    if args.input_mask and args.output_mask:
        print(f"{Fore.YELLOW}Warning: --output-mask is ignored when using --input-mask{Style.RESET_ALL}")
    
    try:
        input_abs_path = os.path.abspath(args.input)
        input_img = nib.load(args.input)
    except Exception as e:
        print(f"{Fore.RED}Error loading input image: {e}{Style.RESET_ALL}")
        sys.exit(1)
    
    if args.input_mask:
        # Mode 1: Using input mask
        print("Using input mask mode")
        try:
            input_mask_img = nib.load(args.input_mask)
        except Exception as e:
            print(f"{Fore.RED}Error loading input mask: {e}{Style.RESET_ALL}")
            sys.exit(1)
            
        input_brain = input_img.get_fdata()

        # Check if mask and image have the same shape and affine
        shapes_match = input_img.shape == input_mask_img.shape
        affines_match = np.allclose(input_img.affine, input_mask_img.affine, rtol=1e-5)

        if not (shapes_match and affines_match):
            print(f"{Fore.YELLOW}Warning: Mask and input image do not match in shape or physical space.{Style.RESET_ALL}")
            print(f"  Image shape: {input_img.shape}, mask shape: {input_mask_img.shape}")
            print(f"  Image affine:\n{input_img.affine}\n  Mask affine:\n{input_mask_img.affine}")
            print("Resampling mask to match input image...")
            resampled_mask_img = resample_to_img(input_mask_img, input_img, interpolation="nearest")
            input_mask = resampled_mask_img.get_fdata().astype(bool)
        else:
            input_mask = input_mask_img.get_fdata().astype(bool)

        input_brain[~input_mask] = 0
        input_brain = nib.Nifti1Image(input_brain, input_img.affine)
        input_brain.to_filename(args.output)
        print(f"{Fore.GREEN}Brain extraction complete. Output saved to: {args.output}{Style.RESET_ALL}")
        
    else:
        # Mode 2: Using parcellation
        print("Using SynthSeg parcellation mode")
        if not args.output_mask:
            print(f"{Fore.YELLOW}Warning: --output-mask not provided. Brain mask will not be saved.{Style.RESET_ALL}")
        
        try:
            synthseg_img = nib.load(args.parcellation)
        except Exception as e:
            print(f"{Fore.RED}Error loading parcellation: {e}{Style.RESET_ALL}")
            sys.exit(1)
            
        input_brain = input_img.get_fdata()

        # Resample synthseg to match input dimensions and space
        # Using nearest interpolation to preserve label values
        print("Resampling parcellation to match input image...")
        resampled_synthseg_img = resample_to_img(
            synthseg_img, input_img, interpolation="nearest"
        )
        synthseg_brain = resampled_synthseg_img.get_fdata()

        mask = synthseg_brain > 0
        print(f"Mask shape: {mask.shape}")
        print(f"Input brain shape: {input_brain.shape}")
        
        if args.remove_cerebellum:
            print("Removing cerebellum regions...")
            # FreeSurfer labels for cerebellum and brainstem structures
            # 7: L-Cereb-WM, 8: L-Cereb-Cortex, 46: R-Cereb-WM, 47: R-Cereb-Cortex
            # 16: Brain-Stem, 15: 4th-Ventricle, 24: CSF
            cerebellum_labels = [7, 8, 46, 47, 16, 15, 24]
            for label in cerebellum_labels:
                mask = mask & (synthseg_brain != label)
            print(f"  Excluded {len(cerebellum_labels)} cerebellar/brainstem labels")

        # Apply the mask to the input image
        input_brain[~mask] = 0
        input_brain = nib.Nifti1Image(input_brain, input_img.affine)
        input_brain.to_filename(args.output)
        print(f"{Fore.GREEN}Brain extraction complete. Output saved to: {args.output}{Style.RESET_ALL}")
        
        if args.output_mask:
            mask = nib.Nifti1Image(mask.astype(np.int8), input_img.affine)
            mask.to_filename(args.output_mask)
            print(f"{Fore.GREEN}Brain mask saved to: {args.output_mask}{Style.RESET_ALL}")
    
    sys.exit(0)  # Explicit success exit
