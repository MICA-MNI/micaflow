#!/usr/bin/env python3
"""
normalize_intensity - Percentile-based Intensity Normalization for MRI Data

Part of the micaflow processing pipeline for neuroimaging data.

This module performs intensity normalization on MRI data using percentile-based clamping
and rescaling. Intensity normalization is crucial for improving consistency across different
scans, scanners, and acquisition protocols, making downstream analysis more robust and
enabling valid comparisons between subjects and timepoints.

Why Intensity Normalization?
----------------------------
MRI signal intensities are arbitrary and vary significantly due to:
- Scanner hardware differences (field strength, coils, gradient systems)
- Acquisition parameters (TR, TE, flip angle, bandwidth)
- Reconstruction algorithms and vendor-specific processing
- Subject-specific factors (head size, tissue properties)

Without normalization:
- Visual comparison across scans is difficult
- Quantitative analyses become unreliable
- Machine learning models perform poorly
- Atlas-based methods may fail
- Group statistics are confounded by intensity variations

Percentile-based Normalization:
------------------------------
This method normalizes intensities by:
1. Computing percentiles on non-zero (brain) voxels only
2. Clamping extreme values at specified percentiles (default: 1st and 99th)
3. Rescaling clamped values to a standardized range (default: 0-100)

Advantages:
- Robust to outliers (extreme bright/dark spots)
- Preserves relative intensity relationships within normal range
- No assumptions about tissue distributions

Limitations:
- Does not account for spatial intensity variations (bias fields)
- May not be appropriate for quantitative sequences (T1/T2 mapping)
- Assumes background is zero (requires brain extraction for best results)
- Cannot correct for different tissue contrasts across modalities

Features:
--------
- Percentile-based clamping to reduce outlier effects
- Customizable percentile thresholds (default: 1st and 99th)
- Customizable output range (default: 0-100)
- Operates only on non-zero voxels (preserves background)
- Preserves original image geometry and header information
- Handles edge cases (no non-zero voxels, uniform intensity)
- Fast processing suitable for large datasets

Command-Line Usage:
------------------
# Basic normalization with default parameters (1st-99th percentiles, 0-100 range)
micaflow normalize_intensity \\
    --input <path/to/image.nii.gz> \\
    --output <path/to/normalized.nii.gz>

# Custom percentiles (more aggressive outlier removal)
micaflow normalize_intensity \\
    --input <path/to/image.nii.gz> \\
    --output <path/to/normalized.nii.gz> \\
    --lower-percentile 2.0 \\
    --upper-percentile 98.0

# Normalize to [0, 1] range (useful for machine learning)
micaflow normalize_intensity \\
    --input <path/to/image.nii.gz> \\
    --output <path/to/normalized.nii.gz> \\
    --min-value 0 \\
    --max-value 1

# Conservative normalization (less outlier removal)
micaflow normalize_intensity \\
    --input <path/to/image.nii.gz> \\
    --output <path/to/normalized.nii.gz> \\
    --lower-percentile 0.5 \\
    --upper-percentile 99.5

Python API Usage:
----------------
>>> from micaflow.scripts.normalize_intensity import normalize_intensity
>>> 
>>> # Basic normalization
>>> normalize_intensity(
...     input_file="t1w.nii.gz",
...     output_file="t1w_normalized.nii.gz"
... )
>>> 
>>> # Custom percentiles and range
>>> normalize_intensity(
...     input_file="t1w.nii.gz",
...     output_file="t1w_normalized.nii.gz",
...     lower_percentile=2,
...     upper_percentile=98,
...     min_val=0,
...     max_val=1
... )
>>> 
>>> # Silent mode (no progress messages)
>>> normalize_intensity(
...     input_file="t1w.nii.gz",
...     output_file="t1w_normalized.nii.gz",
...     verbose=False
... )

Pipeline Integration:
--------------------
Intensity normalization is typically applied AFTER bias field correction:

Structural MRI Pipeline:
1. Denoising (optional)
2. Brain extraction (bet) 
3. Bias field correction (bias_correction)
4. Intensity normalization (normalize_intensity) ← You are here
5. Registration to standard space (coregister)
6. Tissue segmentation

Diffusion MRI Pipeline:
Note: For DWI, normalization is less common. If needed, apply to b0 only:
1. Denoising (denoise)
2. Motion/eddy correction (motion_correction)
3. Distortion correction (apply_SDC)
4. Bias field correction (bias_correction)
5. B0 intensity normalization (optional)

Exit Codes:
----------
0 : Success - normalization completed
1 : Error - invalid inputs, file not found, or processing failure

Technical Notes:
---------------
- Default percentiles: 1st and 99th (clips ~2% of voxels)
- Default output range: 0-100
- Percentile calculation: Only on non-zero voxels (excludes background)
- Zero voxels (background) remain zero after normalization
- Formula: norm = ((clipped - p_low) / (p_high - p_low)) × (max - min) + min
- Processing time: < 1 minute for typical 3D volumes
- Memory efficient: Operates on masked data only
- Data type: Preserved from input (float32/float64)
- NIfTI header: Preserved including orientation and spacing
- Edge case handling: Returns copy if no non-zero voxels found


See Also:
--------
- bias_correction : Remove intensity inhomogeneities before normalization
- bet : Brain extraction for better percentile estimation
- coregister : Register normalized images to standard space

"""

import os
import argparse
import sys
import nibabel as nib
import numpy as np
from colorama import init, Fore, Style

init()

# ANSI color codes
CYAN = Fore.CYAN
GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
BLUE = Fore.BLUE
MAGENTA = Fore.MAGENTA
RED = Fore.RED
BOLD = Style.BRIGHT
RESET = Style.RESET_ALL


def print_help_message():
    """Print comprehensive help message with examples and technical details."""
    
    help_text = f"""
    {CYAN}{BOLD}╔════════════════════════════════════════════════════════════════╗
    ║                  INTENSITY NORMALIZATION                       ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script normalizes MRI intensity values using percentile-based clamping
    and rescaling to improve consistency across scans and scanners.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow normalize_intensity {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--input, -i{RESET}   : Path to the input image file (.nii.gz)
      {YELLOW}--output, -o{RESET}  : Path for the normalized output image (.nii.gz)
    
    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--lower-percentile{RESET}: Lower percentile for clamping (default: 1.0)
                         {MAGENTA}Range: 0-100, must be < upper percentile{RESET}
      {YELLOW}--upper-percentile{RESET}: Upper percentile for clamping (default: 99.0)
                         {MAGENTA}Range: 0-100, must be > lower percentile{RESET}
      {YELLOW}--min-value{RESET}       : Minimum value in output range (default: 0)
      {YELLOW}--max-value{RESET}       : Maximum value in output range (default: 100)
    
    {CYAN}{BOLD}──────────────────── EXAMPLE USAGE ───────────────────────{RESET}
    
    {BLUE}# Example 1: Basic normalization (1st-99th percentiles, 0-100){RESET}
    micaflow normalize_intensity \\
      {YELLOW}--input{RESET} t1w.nii.gz \\
      {YELLOW}--output{RESET} t1w_normalized.nii.gz
    
    {BLUE}# Example 2: Aggressive outlier removal (2nd-98th percentiles){RESET}
    micaflow normalize_intensity \\
      {YELLOW}--input{RESET} t1w.nii.gz \\
      {YELLOW}--output{RESET} t1w_normalized.nii.gz \\
      {YELLOW}--lower-percentile{RESET} 2.0 \\
      {YELLOW}--upper-percentile{RESET} 98.0
    
    {BLUE}# Example 3: Normalize to [0, 1] range (for ML){RESET}
    micaflow normalize_intensity \\
      {YELLOW}--input{RESET} t1w.nii.gz \\
      {YELLOW}--output{RESET} t1w_normalized.nii.gz \\
      {YELLOW}--min-value{RESET} 0 \\
      {YELLOW}--max-value{RESET} 1
    
    {BLUE}# Example 4: Conservative normalization (0.5th-99.5th percentiles){RESET}
    micaflow normalize_intensity \\
      {YELLOW}--input{RESET} t1w.nii.gz \\
      {YELLOW}--output{RESET} t1w_normalized.nii.gz \\
      {YELLOW}--lower-percentile{RESET} 0.5 \\
      {YELLOW}--upper-percentile{RESET} 99.5
    
    {CYAN}{BOLD}─────────── WHY INTENSITY NORMALIZATION? ────────────────{RESET}
    
    {GREEN}Problem:{RESET}
    MRI intensities are arbitrary and vary across:
    {MAGENTA}•{RESET} Different scanners and vendors
    {MAGENTA}•{RESET} Acquisition protocols and parameters
    {MAGENTA}•{RESET} Subjects and scanning sessions
    
    {GREEN}Solution:{RESET}
    Percentile-based normalization provides:
    {MAGENTA}•{RESET} Consistent intensity ranges across scans
    {MAGENTA}•{RESET} Robustness to outliers (bright/dark spots)
    
    {GREEN}Method:{RESET}
    {MAGENTA}1.{RESET} Compute percentiles on non-zero voxels only
    {MAGENTA}2.{RESET} Clamp intensities at specified percentiles
    {MAGENTA}3.{RESET} Rescale to standardized range (e.g., 0-100)
    {MAGENTA}4.{RESET} Background (zero) voxels remain unchanged
    
    {CYAN}{BOLD}────────────────────── WHEN TO USE ──────────────────────{RESET}
    
    {GREEN}Recommended:{RESET}
    {MAGENTA}•{RESET} Before registration across subjects/sessions
    {MAGENTA}•{RESET} Before group-level statistical analysis
    {MAGENTA}•{RESET} For machine learning applications
    {MAGENTA}•{RESET} When combining data from multiple scanners
    {MAGENTA}•{RESET} After bias field correction
    {MAGENTA}•{RESET} For visualization and QC
    
    {RED}Not recommended:{RESET}
    {MAGENTA}•{RESET} Quantitative sequences (T1/T2 mapping, ASL)
    {MAGENTA}•{RESET} Before bias field correction
    {MAGENTA}•{RESET} When absolute intensities have meaning
    {MAGENTA}•{RESET} For sequences with specific intensity scales
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    {MAGENTA}•{RESET} Default: Clamp at 1st and 99th percentiles (clips ~2% of voxels)
    {MAGENTA}•{RESET} Default output range: 0-100
    {MAGENTA}•{RESET} Percentiles computed on NON-ZERO voxels only
    {MAGENTA}•{RESET} Background (zero) voxels remain zero
    {MAGENTA}•{RESET} Formula: ((clipped - p_low) / (p_high - p_low)) × (max - min) + min
    {MAGENTA}•{RESET} Processing time: < 1 minute for typical 3D volumes
    {MAGENTA}•{RESET} Preserves image geometry and header information
    {MAGENTA}•{RESET} Works with any MRI contrast (T1w, T2w, FLAIR, etc.)
    {MAGENTA}•{RESET} Best applied to brain-extracted images
    
    {CYAN}{BOLD}─────────────── PIPELINE POSITION ───────────────────────{RESET}
    1. Brain extraction (bet)
    2. Bias field correction (bias_correction)
    {GREEN}3. Intensity normalization (normalize_intensity){RESET} {MAGENTA}← You are here{RESET}
    4. Registration (coregister)
    5. Analysis/segmentation
    
    {CYAN}{BOLD}────────────────────── EXIT CODES ───────────────────────{RESET}
    {GREEN}0{RESET} : Success - normalization completed
    {RED}1{RESET} : Error - invalid inputs, file not found, or processing failure
    
    {CYAN}{BOLD}───────────────── COMMON ISSUES ─────────────────────────{RESET}
    {YELLOW}Issue:{RESET} Output looks over-bright or over-dark
    {GREEN}Solution:{RESET} Adjust percentiles (more conservative: 0.5-99.5)
    
    {YELLOW}Issue:{RESET} Important features are clipped
    {GREEN}Solution:{RESET} Use more conservative percentiles or check for artifacts
    
    {YELLOW}Issue:{RESET} Background not zero after normalization
    {GREEN}Solution:{RESET} Apply brain extraction first, or check input for NaN values
    
    {YELLOW}Issue:{RESET} All voxels have same intensity after normalization
    {GREEN}Solution:{RESET} Check input - may have uniform intensity or only zeros
    """
    print(help_text)


def normalize_intensity(input_file, output_file, lower_percentile=1.0, upper_percentile=99.0, 
                        min_val=0, max_val=100, verbose=True):
    """
    Normalize intensity of a NIfTI image using percentile-based clamping and rescaling.
    
    This function performs intensity normalization by:
    1. Computing intensity percentiles on non-zero voxels
    2. Clamping values at specified percentiles to reduce outlier effects
    3. Rescaling clamped values to a specified output range
    
    The normalization is robust to outliers and works across different MRI contrasts
    without assumptions about intensity distributions.
    
    Parameters
    ----------
    input_file : str
        Path to the input NIfTI file (.nii.gz).
    output_file : str
        Path where the normalized image will be saved (.nii.gz).
    lower_percentile : float, optional
        Lower percentile for intensity clamping (0-100). Default: 1.0.
        Values below this percentile are clipped to the percentile value.
    upper_percentile : float, optional
        Upper percentile for intensity clamping (0-100). Default: 99.0.
        Values above this percentile are clipped to the percentile value.
    min_val : float, optional
        Minimum value in the normalized output range. Default: 0.
        Lower percentile will be mapped to this value.
    max_val : float, optional
        Maximum value in the normalized output range. Default: 100.
        Upper percentile will be mapped to this value.
    verbose : bool, optional
        Whether to print progress messages. Default: True.
        
    Returns
    -------
    str
        Path to the saved normalized image (same as output_file).
        
    Raises
    ------
    FileNotFoundError
        If input file cannot be found.
    ValueError
        If percentiles are invalid or image has no non-zero voxels.
        
    Notes
    -----
    - Percentile calculation excludes zero-valued voxels (background)
    - Zero voxels remain zero in the output (background preservation)
    - If all voxels are zero, output is a copy of input
    - If p_high == p_low (uniform intensity), no rescaling is applied
    - Preserves original NIfTI header and spatial information
    - Data type is preserved from input image
    
    The normalization formula is:
    norm = ((clipped - p_low) / (p_high - p_low)) × (max_val - min_val) + min_val
    
    where:
    - clipped: intensity values clamped at percentiles
    - p_low: lower percentile value
    - p_high: upper percentile value
    
    Examples
    --------
    >>> # Basic normalization with defaults
    >>> normalize_intensity(
    ...     input_file="t1w.nii.gz",
    ...     output_file="t1w_norm.nii.gz"
    ... )
    >>> 
    >>> # Aggressive outlier removal
    >>> normalize_intensity(
    ...     input_file="t1w.nii.gz",
    ...     output_file="t1w_norm.nii.gz",
    ...     lower_percentile=2.0,
    ...     upper_percentile=98.0
    ... )
    >>> 
    >>> # Normalize to [0, 1] for machine learning
    >>> normalize_intensity(
    ...     input_file="t1w.nii.gz",
    ...     output_file="t1w_norm.nii.gz",
    ...     min_val=0,
    ...     max_val=1
    ... )
    >>> 
    >>> # Silent mode
    >>> output = normalize_intensity(
    ...     input_file="t1w.nii.gz",
    ...     output_file="t1w_norm.nii.gz",
    ...     verbose=False
    ... )
    """
    if verbose:
        print(f"{CYAN}Loading image...{RESET}")
        print(f"  File: {input_file}")
    
    # Validate input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Load the image
    img = nib.load(input_file)
    data = img.get_fdata()
    
    if verbose:
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
    
    # Create a mask of non-zero voxels (exclude background)
    mask = data > 0
    num_nonzero = np.sum(mask)
    
    if verbose:
        print(f"  Non-zero voxels: {num_nonzero:,} / {data.size:,} ({100*num_nonzero/data.size:.1f}%)")
    
    if not np.any(mask):
        print(f"{YELLOW}Warning: Input image contains no non-zero values.{RESET}")
        print(f"{YELLOW}Output will be a copy of input.{RESET}")
        nib.save(img, output_file)
        return output_file
    
    # Calculate percentiles on non-zero voxels only
    if verbose:
        print(f"\n{CYAN}Computing intensity percentiles...{RESET}")
        print(f"  Lower percentile: {lower_percentile}%")
        print(f"  Upper percentile: {upper_percentile}%")
    
    p_low = np.percentile(data[mask], lower_percentile)
    p_high = np.percentile(data[mask], upper_percentile)
    
    if verbose:
        print(f"  Percentile values: [{p_low:.4f}, {p_high:.4f}]")
        
        # Calculate how many voxels will be clipped
        num_below = np.sum(data[mask] < p_low)
        num_above = np.sum(data[mask] > p_high)
        print(f"  Voxels to clip: {num_below + num_above:,} "
              f"({100*(num_below + num_above)/num_nonzero:.1f}%)")
        print(f"    Below threshold: {num_below:,}")
        print(f"    Above threshold: {num_above:,}")
    
    # Check for edge case: uniform intensity
    if p_high == p_low:
        print(f"{YELLOW}Warning: Uniform intensity detected (p_low == p_high).{RESET}")
        print(f"{YELLOW}Output will have constant value.{RESET}")
    
    # Clamp the data (only non-zero voxels)
    if verbose:
        print(f"\n{CYAN}Clamping and rescaling...{RESET}")
        print(f"  Clamping range: [{p_low:.4f}, {p_high:.4f}]")
        print(f"  Output range: [{min_val}, {max_val}]")
    
    data_masked = data[mask].copy()
    data_masked = np.clip(data_masked, p_low, p_high)
    
    # Normalize to the desired range
    if p_high > p_low:  # Avoid division by zero
        data_masked = ((data_masked - p_low) / (p_high - p_low)) * (max_val - min_val) + min_val
    else:
        # If uniform intensity, set to middle of range
        data_masked[:] = (min_val + max_val) / 2
    
    # Put the normalized data back (zeros remain zero)
    normalized_data = np.zeros_like(data)
    normalized_data[mask] = data_masked
    
    if verbose:
        print(f"\n{CYAN}Output statistics:{RESET}")
        print(f"  Min: {np.min(normalized_data[mask]):.4f}")
        print(f"  Max: {np.max(normalized_data[mask]):.4f}")
        print(f"  Mean: {np.mean(normalized_data[mask]):.4f}")
        print(f"  Std: {np.std(normalized_data[mask]):.4f}")
    
    # Create a new image with the same header
    normalized_img = nib.Nifti1Image(normalized_data, img.affine, header=img.header)
    
    # Save the normalized image
    if verbose:
        print(f"\n{CYAN}Saving normalized image...{RESET}")
        print(f"  Output: {output_file}")
    
    nib.save(normalized_img, output_file)
    
    if verbose:
        print(f"\n{GREEN}{BOLD}Intensity normalization completed successfully!{RESET}")
        print(f"  Input: {input_file}")
        print(f"  Output: {output_file}")
        print(f"  Percentiles: [{lower_percentile}%, {upper_percentile}%]")
        print(f"  Output range: [{min_val}, {max_val}]")
    
    return output_file


if __name__ == "__main__":
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="Normalize MRI intensity values using percentile-based clamping",
        add_help=False  # Use custom help
    )
    parser.add_argument(
        "--input", "-i", 
        required=True, 
        help="Input NIfTI image file (.nii.gz)"
    )
    parser.add_argument(
        "--output", "-o", 
        required=True, 
        help="Output normalized image file (.nii.gz)"
    )
    parser.add_argument(
        "--lower-percentile", 
        type=float, 
        default=1.0, 
        help="Lower percentile for clamping (default: 1.0)"
    )
    parser.add_argument(
        "--upper-percentile", 
        type=float, 
        default=99.0, 
        help="Upper percentile for clamping (default: 99.0)"
    )
    parser.add_argument(
        "--min-value", 
        type=float, 
        default=0, 
        help="Minimum value in output range (default: 0)"
    )
    parser.add_argument(
        "--max-value", 
        type=float, 
        default=100, 
        help="Maximum value in output range (default: 100)"
    )
    
    args = parser.parse_args()
    
    try:
        # Validate percentile values
        if args.lower_percentile < 0 or args.lower_percentile > 100:
            raise ValueError(f"Lower percentile must be between 0 and 100, got {args.lower_percentile}")
        
        if args.upper_percentile < 0 or args.upper_percentile > 100:
            raise ValueError(f"Upper percentile must be between 0 and 100, got {args.upper_percentile}")
        
        if args.lower_percentile >= args.upper_percentile:
            raise ValueError(f"Lower percentile ({args.lower_percentile}) must be less than "
                           f"upper percentile ({args.upper_percentile})")
        
        if args.min_value >= args.max_value:
            raise ValueError(f"Minimum value ({args.min_value}) must be less than "
                           f"maximum value ({args.max_value})")
        
        # Call the normalization function
        output_path = normalize_intensity(
            args.input, 
            args.output,
            args.lower_percentile,
            args.upper_percentile, 
            args.min_value,
            args.max_value
        )
        
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\n{RED}{BOLD}File not found:{RESET}")
        print(f"  {str(e)}")
        sys.exit(1)
        
    except ValueError as e:
        print(f"\n{RED}{BOLD}Value error:{RESET}")
        print(f"  {str(e)}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n{RED}{BOLD}Error during normalization:{RESET}")
        print(f"  {str(e)}")
        print(f"\n{YELLOW}Run 'micaflow normalize_intensity --help' for usage information.{RESET}")
        sys.exit(1)