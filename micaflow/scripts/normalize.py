#!/usr/bin/env python3
"""
normalize_intensity - Percentile-based Intensity Normalization for MRI Data

Part of the micaflow processing pipeline for neuroimaging data.

This script performs intensity normalization on MRI data by:
1. Clamping values at the 1st and 99th percentiles to reduce outlier effects
2. Rescaling the clamped values to a standardized 0-100 range

This normalization helps improve consistency across different scans and scanners,
making downstream analysis and visualization more robust.

API Usage:
---------
micaflow normalize_intensity 
    --input <path/to/image.nii.gz>
    --output <path/to/normalized.nii.gz>
    [--lower-percentile <value>]
    [--upper-percentile <value>]
    [--min-value <value>]
    [--max-value <value>]

Python Usage:
-----------
>>> from micaflow.scripts.normalize_intensity import normalize_intensity
>>> normalize_intensity(
...     input_file="t1w.nii.gz",
...     output_file="t1w_normalized.nii.gz",
...     lower_percentile=1,
...     upper_percentile=99,
...     min_val=0,
...     max_val=100
... )
"""

import os
import argparse
import sys
import nibabel as nib
import numpy as np
from colorama import init, Fore, Style

init()

def print_help_message():
    """Print a help message with formatted text."""
    YELLOW = Fore.YELLOW
    GREEN = Fore.GREEN
    CYAN = Fore.CYAN
    BOLD = Style.BRIGHT
    RESET = Style.RESET_ALL
    
    help_text = f"""
    {CYAN}{BOLD}╔════════════════════════════════════════════════════════════════╗{RESET}
    {CYAN}{BOLD}║                 INTENSITY NORMALIZATION                        ║{RESET}
    {CYAN}{BOLD}╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script normalizes MRI intensity values by clamping at specified 
    percentiles and rescaling to a standard range.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow normalize_intensity [options]
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--input, -i{RESET}  : Path to the input image file (.nii.gz)
      {YELLOW}--output, -o{RESET} : Path for the normalized output image (.nii.gz)
    
    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--lower-percentile{RESET} : Lower percentile for clamping (default: 1.0)
      {YELLOW}--upper-percentile{RESET} : Upper percentile for clamping (default: 99.0)
      {YELLOW}--min-value{RESET}        : Minimum value in output range (default: 0)
      {YELLOW}--max-value{RESET}        : Maximum value in output range (default: 100)
    
    {CYAN}{BOLD}──────────────────── EXAMPLE USAGE ──────────────────────{RESET}
    
    {GREEN}# Basic usage with default parameters{RESET}
    micaflow normalize_intensity {YELLOW}--input{RESET} t1w.nii.gz {YELLOW}--output{RESET} t1w_norm.nii.gz
    
    {GREEN}# Custom percentiles and range{RESET}
    micaflow normalize_intensity {YELLOW}--input{RESET} t1w.nii.gz {YELLOW}--output{RESET} t1w_norm.nii.gz \
      {YELLOW}--lower-percentile{RESET} 2.0 {YELLOW}--upper-percentile{RESET} 98.0 {YELLOW}--min-value{RESET} 0 {YELLOW}--max-value{RESET} 1
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    - Clamping at percentiles helps reduce the effect of outliers
    - Data type is preserved in the output image
    - Non-brain voxels (zeros) remain zero after normalization
    
    """
    print(help_text)

def normalize_intensity(input_file, output_file, lower_percentile=1.0, upper_percentile=99.0, 
                        min_val=0, max_val=100, verbose=True):
    """
    Normalize intensity of a NIfTI image by clamping at percentiles and rescaling.
    
    Args:
        input_file (str): Path to the input NIfTI file (.nii.gz).
        output_file (str): Path where the normalized image will be saved.
        lower_percentile (float): Lower percentile for clamping (0-100).
        upper_percentile (float): Upper percentile for clamping (0-100).
        min_val (float): Minimum value in the output range.
        max_val (float): Maximum value in the output range.
        verbose (bool): Whether to print progress messages.
    
    Returns:
        None: The normalized image is saved to the specified output path.
    """
    if verbose:
        print(f"Loading image: {input_file}")
    
    # Load the image
    img = nib.load(input_file)
    data = img.get_fdata()
    
    # Create a mask of non-zero voxels
    mask = data > 0
    
    if not np.any(mask):
        if verbose:
            print("Warning: Input image contains no non-zero values. Output will be a copy of input.")
        nib.save(img, output_file)
        return
    
    # Calculate percentiles on non-zero voxels only
    if verbose:
        print(f"Calculating {lower_percentile}th and {upper_percentile}th percentiles...")
    
    p_low = np.percentile(data[mask], lower_percentile)
    p_high = np.percentile(data[mask], upper_percentile)
    
    if verbose:
        print(f"Clamping values between {p_low:.4f} and {p_high:.4f}")
    
    # Clamp the data (only non-zero voxels)
    data_masked = data[mask]
    data_masked = np.clip(data_masked, p_low, p_high)
    
    # Normalize to the desired range
    if verbose:
        print(f"Normalizing to range [{min_val}, {max_val}]")
    
    if p_high > p_low:  # Avoid division by zero
        data_masked = ((data_masked - p_low) / (p_high - p_low)) * (max_val - min_val) + min_val
    
    # Put the normalized data back
    normalized_data = np.zeros_like(data)
    normalized_data[mask] = data_masked
    
    # Create a new image with the same header
    normalized_img = nib.Nifti1Image(normalized_data, img.affine, header=img.header)
    
    # Save the normalized image
    if verbose:
        print(f"Saving normalized image to {output_file}")
    
    nib.save(normalized_img, output_file)
    
    if verbose:
        print("Normalization complete!")

if __name__ == "__main__":
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)
    
    parser = argparse.ArgumentParser(description="Normalize MRI intensity values")
    parser.add_argument("--input", "-i", required=True, help="Input NIfTI image file (.nii.gz)")
    parser.add_argument("--output", "-o", required=True, help="Output normalized image file (.nii.gz)")
    parser.add_argument("--lower-percentile", type=float, default=1.0, 
                        help="Lower percentile for clamping (default: 1.0)")
    parser.add_argument("--upper-percentile", type=float, default=99.0, 
                        help="Upper percentile for clamping (default: 99.0)")
    parser.add_argument("--min-value", type=float, default=0, 
                        help="Minimum value in output range (default: 0)")
    parser.add_argument("--max-value", type=float, default=100, 
                        help="Maximum value in output range (default: 100)")
    args = parser.parse_args()
    
    # Validate percentile values
    if args.lower_percentile < 0 or args.lower_percentile > 100:
        print("Error: Lower percentile must be between 0 and 100")
        sys.exit(1)
    if args.upper_percentile < 0 or args.upper_percentile > 100:
        print("Error: Upper percentile must be between 0 and 100")
        sys.exit(1)
    if args.lower_percentile >= args.upper_percentile:
        print("Error: Lower percentile must be less than upper percentile")
        sys.exit(1)
    
    # Call the normalization function
    normalize_intensity(
        args.input, 
        args.output,
        args.lower_percentile,
        args.upper_percentile, 
        args.min_value,
        args.max_value
    )