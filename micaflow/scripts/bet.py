"""
bet - Brain Extraction Tool using SynthSeg

Part of the micaflow processing pipeline for neuroimaging data.

This module provides brain extraction (skull stripping) functionality using the SynthSeg-generated mask, a deep learning-based approach that accurately segments the
brain from surrounding tissues in MR images. SynthSeg offers superior performance over traditional
methods, particularly for clinical and non-standard MR images.

Features:
--------
- Deep learning-based brain extraction with state-of-the-art accuracy
- Compatible with various MRI modalities (T1w, T2w, FLAIR)
- Produces both skull-stripped images and binary brain masks
- Robust to imaging artifacts and pathologies

API Usage:
---------
micaflow bet
    --input <path/to/image.nii.gz>
    --output <path/to/brain.nii.gz>
    --output-mask <path/to/brain_mask.nii.gz>
    --parcellation <path/to/parcellation.nii.gz>
    [--remove-cerebellum]

Python Usage:
-----------
>>> import subprocess
>>> from micaflow.scripts.bet import run_hdbet
>>> run_hdbet(
...     input_file="t1w.nii.gz",
...     output_file="brain.nii.gz",
...     mask_file="brain_mask.nii.gz",
...     parcellation_file="parcellation.nii.gz",
... )

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
    ║                            BET                                 ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script performs brain extraction (skull stripping) on MRI images 
    using SynthSeg. It accurately segments the brain 
    from surrounding tissues.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow bet {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--input{RESET}, {YELLOW}-i{RESET}      : Path to the input MR image (.nii.gz)
      {YELLOW}--output{RESET}, {YELLOW}-o{RESET}     : Path for the output brain-extracted image (.nii.gz)
      {YELLOW}--output-mask{RESET}, {YELLOW}-m{RESET}: Path for the output brain mask (.nii.gz)
      {YELLOW}--parcellation{RESET}, {YELLOW}-p{RESET}: Path to the parcellation file (.nii.gz)
    
    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--remove-cerebellum{RESET}, {YELLOW}-r{RESET}: Remove cerebellum from the input image (optional)
      {YELLOW}--input-mask{RESET}                : Path to the input mask image (.nii.gz) (optional)
    
    {CYAN}{BOLD}────────────────── EXAMPLE USAGE ────────────────────────{RESET}
    
    {GREEN}# Run BET{RESET}
    micaflow bet {YELLOW}--input{RESET} t1w.nii.gz {YELLOW}--output{RESET} t1w_brain.nii.gz {YELLOW}--output-mask{RESET} t1w_brain_mask.nii.gz {YELLOW}--parcellation{RESET} parcellation.nii.gz
    {GREEN}# Run BET with cerebellum removal{RESET}
    micaflow bet {YELLOW}--input{RESET} t1w.nii.gz {YELLOW}--output{RESET} t1w_brain.nii.gz {YELLOW}--output-mask{RESET} t1w_brain_mask.nii.gz {YELLOW}--parcellation{RESET} parcellation.nii.gz {YELLOW}--remove-cerebellum{RESET}
    {GREEN}# Run BET with input mask{RESET}
    micaflow bet {YELLOW}--input{RESET} t1w.nii.gz {YELLOW}--output{RESET} t1w_brain.nii.gz {YELLOW}--output-mask{RESET} t1w_brain_mask.nii.gz {YELLOW}--input-mask{RESET} input_mask.nii.gz
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    - The output is a brain-extracted image and a binary brain mask
    
    """
    print(help_text)


if __name__ == "__main__":
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Perform brain extraction using HD-BET"
    )
    parser.add_argument("--input", "-i", required=True, help="Input MR image file")
    parser.add_argument("--input-mask", help="Input mask image file (optional)")
    parser.add_argument(
        "--output", "-o", required=True, help="Output brain-extracted image file"
    )
    parser.add_argument(
        "--output-mask",
        "-m",
        help="Output brain-extracted mask image file",
    )
    parser.add_argument(
        "--parcellation",
        "-p",
        help="Parcellation file for the input image (optional)",
    )
    parser.add_argument(
        "--remove-cerebellum",
        "-r",
        action="store_true",
        help="Remove cerebellum from the input image (optional)",
    )

    args = parser.parse_args()
    input_abs_path = os.path.abspath(args.input)
    input_img = nib.load(args.input)
    if args.input_mask:
        # If an input mask is provided, load it and apply it to the input image
        print("Using input mask")
        input_mask = nib.load(args.input_mask)
        input_brain = input_img.get_fdata()
        input_mask = nib.load(args.input_mask).get_fdata().astype(bool)
        input_brain[~input_mask] = 0
        input_brain = nib.Nifti1Image(input_brain, nib.load(args.input).affine)
        input_brain.to_filename(args.output)
    else:
        synthseg_img = nib.load(args.parcellation)
        input_brain = input_img.get_fdata()

        # Resample synthseg to match input dimensions and space
        # Using nearest interpolation to preserve label values
        resampled_synthseg_img = resample_to_img(
            synthseg_img, input_img, interpolation="nearest"
        )
        synthseg_brain = resampled_synthseg_img.get_fdata()

        mask = synthseg_brain > 0
        print("mask.shape", mask.shape)
        print("input_brain.shape", input_brain.shape)
        if args.remove_cerebellum:
            # If removing cerebellum, exclude these labels from the mask
            cerebellum_labels = [7, 8, 46, 47, 16, 15, 24]
            for label in cerebellum_labels:
                mask = mask & (synthseg_brain != label)

        # Apply the mask to the input image
        input_brain[~mask] = 0
        input_brain = nib.Nifti1Image(input_brain, nib.load(args.input).affine)
        input_brain.to_filename(args.output)
        mask = nib.Nifti1Image(mask.astype(np.int8), nib.load(args.input).affine)
        mask.to_filename(args.output_mask)
