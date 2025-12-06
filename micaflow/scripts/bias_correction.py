"""
bias_correction - N4 Bias Field Correction for MRI Data

Part of the micaflow processing pipeline for neuroimaging data.

This module corrects intensity non-uniformity (bias field) in MR images using the 
N4ITK algorithm from Advanced Normalization Tools (ANTs). Bias field artifacts appear 
as smooth, spatially-varying intensity distortions caused by magnetic field 
inhomogeneities and are a common problem in MRI that can negatively impact 
quantitative analysis, segmentation, and registration.

What is Bias Field?
------------------
Bias field (also called intensity non-uniformity or INU) is a low-frequency,
smooth variation in signal intensity across an MRI scan that doesn't reflect 
true tissue properties. It's caused by:
- Main magnetic field (B0) inhomogeneities
- Radio-frequency (RF) field (B1) inhomogeneities  
- Patient positioning and anatomy
- Receiver coil sensitivity variations
- Gradient non-linearities

Effects of bias field:
- Same tissue appears with different intensities in different locations
- Degrades tissue segmentation accuracy
- Affects quantitative intensity measurements
- Impairs image registration
- Reduces reproducibility across scanners/sessions

Why N4 Bias Field Correction?
-----------------------------
The N4ITK (N4) algorithm is the current state-of-the-art for bias correction:
- Improved version of the classic N3 algorithm
- Non-parametric, doesn't assume specific bias field shape
- Works on variety of MRI contrasts (T1w, T2w, FLAIR, DWI)
- Fast iterative B-spline approximation
- Robust to different tissue contrasts
- Widely validated and adopted in neuroimaging

Advantages over N3:
- Better convergence properties
- More accurate field estimation
- Faster computation
- Better handling of noise

How N4 Works:
------------
1. Initialize smooth bias field estimate (B-spline)
2. Iterate to refine estimate:
   a. Sharpen histogram by dividing image by current bias estimate
   b. Smooth the sharpened image
   c. Update bias field estimate
   d. Check convergence
3. Apply final bias field correction: corrected = original / bias_field
4. Return corrected image

Key parameters:
- Convergence threshold: When to stop iterations
- B-spline fitting: Distance between knots
- Shrink factor: Speed up by processing at lower resolution
- Number of iterations: Maximum refinement steps

Algorithm Details:
-----------------
N4 uses a multi-resolution B-spline approximation:
- Level 1: Coarse bias field estimate (large B-spline knot spacing)
- Level 2-4: Progressively finer estimates (smaller knot spacing)
- Iterative refinement at each level
- Convergence when change in coefficients falls below threshold

For 4D images (DWI):
- Bias field estimated from b=0 volume (best SNR)
- Same bias field applied to all diffusion volumes
- Assumes bias field is constant across all b-values
- More efficient than per-volume correction

Features:
--------
- Supports both 3D anatomical and 4D diffusion-weighted images
- Automatic detection of image dimensionality (3D vs 4D)
- Optional brain mask input for improved correction accuracy
- Automatic mask generation if not provided (3D only)
- Efficient 4D processing: estimates bias from b=0, applies to all volumes
- Automatic resampling of mismatched masks
- Maintains all image header information (spacing, origin, direction)
- Returns bias field for quality control (optional)
- Multi-resolution processing for speed

Command-Line Usage:
------------------
# 3D anatomical image (auto-detects dimensionality)
micaflow bias_correction \\
    --input T1w.nii.gz \\
    --output T1w_corrected.nii.gz

# 3D with explicit mask
micaflow bias_correction \\
    --input T1w.nii.gz \\
    --output T1w_corrected.nii.gz \\
    --mask brain_mask.nii.gz

# 4D diffusion image (requires b0 and mask)
micaflow bias_correction \\
    --input DWI.nii.gz \\
    --output DWI_corrected.nii.gz \\
    --mask brain_mask.nii.gz \\
    --b0 b0.nii.gz \\
    --b0-output b0_corrected.nii.gz \\
    --mode 4d

# Explicit 3D mode
micaflow bias_correction \\
    --input FLAIR.nii.gz \\
    --output FLAIR_corrected.nii.gz \\
    --mask brain_mask.nii.gz \\
    --mode 3d

# Custom shell dimension (if volumes not in dimension 3)
micaflow bias_correction \\
    --input DWI.nii.gz \\
    --output DWI_corrected.nii.gz \\
    --b0 b0.nii.gz \\
    --b0-output b0_corrected.nii.gz \\
    --shell-dimension 4

# With Gibbs ringing removal (requires DIPY)
micaflow bias_correction \\
    --input DWI.nii.gz \\
    --output DWI_corrected.nii.gz \\
    --mask brain_mask.nii.gz \\
    --b0 b0.nii.gz \\
    --b0-output b0_corrected.nii.gz \\
    --mode 4d \\
    --gibbs

Python API Usage:
----------------
>>> from micaflow.scripts.bias_correction import run_bias_field_correction
>>> 
>>> # Basic 3D usage (auto mode)
>>> run_bias_field_correction(
...     image_path="T1w.nii.gz",
...     output_path="T1w_corrected.nii.gz"
... )
>>> 
>>> # 3D with mask
>>> run_bias_field_correction(
...     image_path="T1w.nii.gz",
...     output_path="T1w_corrected.nii.gz",
...     mask_path="brain_mask.nii.gz"
... )
>>> 
>>> # 4D diffusion image
>>> run_bias_field_correction(
...     image_path="DWI.nii.gz",
...     output_path="DWI_corrected.nii.gz",
...     mask_path="brain_mask.nii.gz",
...     b0_path="b0.nii.gz",
...     b0_corrected_path="b0_corrected.nii.gz",
...     mode="4d",
...     shell_dimension=3
... )

Pipeline Integration:
--------------------
Bias correction is typically the FIRST preprocessing step:

Structural MRI Pipeline:
1. Bias field correction ← You are here
2. Brain extraction (skull stripping)
3. Tissue segmentation
4. Registration to template
5. Morphometric analysis

Diffusion MRI Pipeline:
1. Bias field correction (from b=0) ← You are here
2. Denoising
3. Motion and eddy current correction
4. Susceptibility distortion correction
5. Tensor fitting and tractography

Why first?
- Most other algorithms assume uniform intensities
- Improves brain extraction accuracy
- Essential for accurate segmentation
- Improves registration convergence
- Must be done before normalization

Exit Codes:
----------
0 : Success - bias correction completed
1 : Error - invalid inputs, file not found, or processing failure

Output Files:
------------
For 3D images:
- Corrected image with same dimensions as input
- Intensity values divided by estimated bias field

For 4D images:
- Corrected 4D DWI volume
- Corrected b=0 volume (if --b0-output specified)
- Same number of volumes as input

Technical Notes:
---------------
- Algorithm: N4ITK (Improved N3)
- Convergence: Typically 3-5 iterations
- B-spline knot spacing: 200mm (coarse to fine)
- Processing time: 
  * 3D: 30-120 seconds depending on resolution
  * 4D: 1-5 minutes (bias from b=0, applied to all)
- Memory: ~2-4 GB for typical 3D volumes
- Shrink factor: 4 (processes at 1/4 resolution internally)
- Number of fitting levels: 4
- Wiener filter noise: 0.01

3D vs 4D Processing:
--------------------
3D Mode (Anatomical):
- Single volume processing
- Mask optional (auto-generated if not provided)
- Direct N4 application
- ~30-120 seconds

4D Mode (Diffusion):
- Bias estimated from b=0 volume only
- Same bias applied to all volumes
- Mask required (or auto-generated from b=0)
- Assumes bias constant across b-values
- More efficient than per-volume correction
- ~1-5 minutes for typical DWI

Quality Control:
---------------
Visual inspection recommended:
1. Check that bias field is smooth and plausible
2. Verify improved intensity uniformity
3. Look for over-correction (intensity inversions)
4. Ensure tissue contrast is preserved
5. Compare before/after histograms

Expected improvements:
- More uniform white matter intensities
- Sharper tissue boundaries
- Better tissue separability in histogram
- Reduced intensity variation in same tissue

Warning signs:
- Rippling or oscillations in corrected image
- Inverted intensities in some regions
- Loss of contrast between tissues
- Extreme intensity changes
- Artifacts at image boundaries

Limitations:
-----------
- Assumes bias field is smooth and slowly varying
- May struggle with very low SNR images
- Can amplify noise in homogeneous regions
- Requires sufficient tissue contrast
- May fail with extreme field inhomogeneities
- Assumes multiplicative bias model
- Not designed for correction of gradient distortions

For 4D images specifically:
- Assumes bias field constant across all b-values
- Less accurate if bias varies with diffusion weighting
- Requires good quality b=0 volume

Best Practices:
--------------
1. Always provide a mask when possible (improves accuracy)
2. For 3D: mask can be loose (includes some background)
3. For 4D: use tight brain mask from skull-stripped b=0
4. Run before denoising (bias affects noise distribution)
5. Visual QC of outputs is essential
6. Save bias field for quality assessment
7. Check that tissue histograms are improved

Comparison with Other Methods:
------------------------------
N3 (predecessor):
+ Well-established
+ Similar principle
- Slower convergence
- Less accurate
- Older algorithm

FAST (FSL):
+ Integrated with segmentation
+ Fast processing
- Requires good initial segmentation
- Less flexible for different contrasts

SPM Unified Segmentation:
+ Joint bias correction and segmentation
+ Well-integrated in SPM
- Slower processing
- Less modular

N4ITK (this implementation):
+ State-of-the-art accuracy
+ Fast convergence
+ Works with any contrast
+ Flexible and robust
+ Widely adopted standard

See Also:
--------
- denoise : Noise reduction (run after bias correction)
- bet : Brain extraction (run after bias correction)
- normalize : Intensity normalization (after bias correction)

References:
----------
1. Tustison NJ, Avants BB, Cook PA, et al. N4ITK: Improved N3 Bias Correction.
   IEEE Trans Med Imaging. 2010;29(6):1310-1320. doi:10.1109/TMI.2010.2046908

2. Sled JG, Zijdenbos AP, Evans AC. A nonparametric method for automatic 
   correction of intensity nonuniformity in MRI data. IEEE Trans Med Imaging.
   1998;17(1):87-97. doi:10.1109/42.668698

3. Tustison NJ, Gee JC. N4ITK: Nick's N3 ITK Implementation For MRI Bias Field
   Correction. Insight J. 2009. http://hdl.handle.net/10380/3053

4. Avants BB, Tustison NJ, Song G, et al. A reproducible evaluation of ANTs
   similarity metric performance in brain image registration. Neuroimage.
   2011;54(3):2033-2044. doi:10.1016/j.neuroimage.2010.09.025
"""

import ants
import numpy as np
import argparse
import sys
import os
import tempfile
from colorama import init, Fore, Style

try:
    from dipy.denoise.gibbs import gibbs_removal
    HAS_DIPY = True
except ImportError:
    HAS_DIPY = False

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
    """
    Print comprehensive help message with examples and usage instructions.
    
    This function displays detailed information about N4 bias field correction including:
    - What bias field is and why correction matters
    - How the N4 algorithm works
    - Differences between 3D and 4D processing
    - Command-line options and usage
    - Multiple examples for different scenarios
    - Quality control recommendations
    - Processing details and timing
    
    The help message uses color-coded sections for better readability.
    
    Examples
    --------
    >>> # Display help message
    >>> print_help_message()
    
    >>> # Help is shown automatically with --help, -h, or no arguments
    >>> # micaflow bias_correction --help
    
    Notes
    -----
    - Called automatically when script run with --help, -h, or no arguments
    - Provides more detail than standard argparse help
    - Uses ANSI color codes for visual organization
    - Explains both 3D and 4D processing modes
    """
    
    help_text = f"""
    {CYAN}{BOLD}╔════════════════════════════════════════════════════════════════╗
    ║                    N4 BIAS FIELD CORRECTION                    ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script corrects intensity non-uniformity (bias field) in MR images 
    using the N4ITK algorithm from ANTs. It supports both 3D anatomical images 
    and 4D diffusion-weighted images.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow bias_correction {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--input{RESET}, {YELLOW}-i{RESET}    : Path to the input image (.nii.gz)
      {YELLOW}--output{RESET}, {YELLOW}-o{RESET}   : Path for the output bias-corrected image
    
    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--mask{RESET}, {YELLOW}-m{RESET}     : Path to brain mask (recommended, required for 4D)
      {YELLOW}--mode{RESET}         : Processing mode: 3d, 4d, or auto (default: auto)
      {YELLOW}--b0{RESET}           : b=0 image path (required for 4D mode)
      {YELLOW}--b0-output{RESET}    : Path for corrected b=0 output (4D mode)
      {YELLOW}--shell-dimension{RESET}: Dimension for diffusion volumes (default: 3)
      {YELLOW}--gibbs{RESET}        : Apply Gibbs ringing removal (requires DIPY)
    
    {CYAN}{BOLD}──────────────────── EXAMPLE USAGE ──────────────────────{RESET}
    
    {BLUE}# Example 1: Basic 3D image (auto-detects){RESET}
    micaflow bias_correction \\
      {YELLOW}--input{RESET} T1w.nii.gz \\
      {YELLOW}--output{RESET} T1w_corrected.nii.gz
    
    {BLUE}# Example 2: 3D with mask (recommended){RESET}
    micaflow bias_correction \\
      {YELLOW}--input{RESET} T1w.nii.gz \\
      {YELLOW}--output{RESET} T1w_corrected.nii.gz \\
      {YELLOW}--mask{RESET} brain_mask.nii.gz
    
    {BLUE}# Example 3: 4D diffusion image{RESET}
    micaflow bias_correction \\
      {YELLOW}--input{RESET} DWI.nii.gz \\
      {YELLOW}--output{RESET} DWI_corrected.nii.gz \\
      {YELLOW}--mask{RESET} brain_mask.nii.gz \\
      {YELLOW}--b0{RESET} b0.nii.gz \\
      {YELLOW}--b0-output{RESET} b0_corrected.nii.gz \\
      {YELLOW}--mode{RESET} 4d
    
    {BLUE}# Example 4: Explicit 3D mode{RESET}
    micaflow bias_correction \\
      {YELLOW}--input{RESET} FLAIR.nii.gz \\
      {YELLOW}--output{RESET} FLAIR_corrected.nii.gz \\
      {YELLOW}--mode{RESET} 3d
    
    {CYAN}{BOLD}────────── WHAT IS BIAS FIELD? ────────────────────────{RESET}
    
    {GREEN}Bias field (intensity non-uniformity):{RESET}
    {MAGENTA}•{RESET} Smooth variation in signal intensity across the image
    {MAGENTA}•{RESET} Doesn't reflect true tissue properties
    {MAGENTA}•{RESET} Caused by magnetic field inhomogeneities
    {MAGENTA}•{RESET} Affects segmentation, registration, and quantification
    {MAGENTA}•{RESET} Common artifact in all MRI scans
    
    {GREEN}Why correction is important:{RESET}
    {MAGENTA}•{RESET} Same tissue appears with different intensities
    {MAGENTA}•{RESET} Degrades automated analysis accuracy
    {MAGENTA}•{RESET} Reduces reproducibility across sessions/scanners
    {MAGENTA}•{RESET} Must be corrected before most analyses
    
    {CYAN}{BOLD}────────────── HOW N4 WORKS ────────────────────────────{RESET}
    
    {GREEN}N4ITK Algorithm:{RESET}
    {MAGENTA}1.{RESET} Initialize smooth bias field estimate (B-spline)
    {MAGENTA}2.{RESET} Iteratively refine:
       {MAGENTA}•{RESET} Sharpen histogram by dividing by current estimate
       {MAGENTA}•{RESET} Smooth the sharpened image
       {MAGENTA}•{RESET} Update bias field estimate
    {MAGENTA}3.{RESET} Apply final correction: corrected = original / bias_field
    {MAGENTA}4.{RESET} Converges in 3-5 iterations typically
    
    {GREEN}Multi-resolution processing:{RESET}
    {MAGENTA}•{RESET} Level 1: Coarse estimate (large B-spline knots)
    {MAGENTA}•{RESET} Levels 2-4: Progressively finer estimates
    {MAGENTA}•{RESET} Fast convergence at each level
    
    {CYAN}{BOLD}────────────── 3D vs 4D PROCESSING ─────────────────────{RESET}
    
    {GREEN}3D Mode (Anatomical images):{RESET}
    {MAGENTA}•{RESET} Single volume processing
    {MAGENTA}•{RESET} Mask optional (auto-generated if not provided)
    {MAGENTA}•{RESET} Direct N4 application
    {MAGENTA}•{RESET} Processing time: 30-120 seconds
    {MAGENTA}•{RESET} Best for T1w, T2w, FLAIR, etc.
    
    {GREEN}4D Mode (Diffusion images):{RESET}
    {MAGENTA}•{RESET} Bias estimated from b=0 volume only
    {MAGENTA}•{RESET} Same bias applied to all volumes
    {MAGENTA}•{RESET} Mask required (or auto-generated from b=0)
    {MAGENTA}•{RESET} Processing time: 1-5 minutes
    {MAGENTA}•{RESET} Assumes bias constant across b-values
    {MAGENTA}•{RESET} More efficient than per-volume correction
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    {MAGENTA}•{RESET} Auto mode detects 3D vs 4D automatically
    {MAGENTA}•{RESET} Always provide mask when possible (improves accuracy)
    {MAGENTA}•{RESET} For 4D: b=0 image required for bias estimation
    {MAGENTA}•{RESET} Mask will be automatically resampled if needed
    {MAGENTA}•{RESET} Run as FIRST preprocessing step (before denoising)
    {MAGENTA}•{RESET} Visual QC recommended after correction
    
    {CYAN}{BOLD}─────────────── PIPELINE POSITION ───────────────────────{RESET}
    {BLUE}Structural MRI Pipeline:{RESET}
    {GREEN}1. Bias field correction{RESET} {MAGENTA}← You are here{RESET}
    2. Brain extraction
    3. Tissue segmentation
    4. Registration
    
    {BLUE}Diffusion MRI Pipeline:{RESET}
    {GREEN}1. Bias field correction{RESET} {MAGENTA}← You are here{RESET}
    2. Denoising
    3. Motion correction
    4. Distortion correction
    
    {CYAN}{BOLD}────────────────────── EXIT CODES ───────────────────────{RESET}
    {GREEN}0{RESET} : Success - bias correction completed
    {RED}1{RESET} : Error - invalid inputs, file not found, or processing failure
    
    {CYAN}{BOLD}───────────── QUALITY CONTROL ───────────────────────────{RESET}
    {YELLOW}Visual inspection:{RESET}
    {MAGENTA}1.{RESET} Check that bias field is smooth and plausible
    {MAGENTA}2.{RESET} Verify improved intensity uniformity
    {MAGENTA}3.{RESET} Look for over-correction artifacts
    {MAGENTA}4.{RESET} Ensure tissue contrast is preserved
    {MAGENTA}5.{RESET} Compare before/after histograms
    
    {YELLOW}Expected improvements:{RESET}
    {MAGENTA}•{RESET} More uniform white matter intensities
    {MAGENTA}•{RESET} Sharper tissue boundaries
    {MAGENTA}•{RESET} Better tissue separability
    
    {CYAN}{BOLD}───────────────── COMMON ISSUES ─────────────────────────{RESET}
    {YELLOW}Issue:{RESET} Over-correction or artifacts
    {GREEN}Solution:{RESET} Provide tighter brain mask, check input quality
    
    {YELLOW}Issue:{RESET} Very slow processing
    {GREEN}Solution:{RESET} Normal for high-res images (1-5 min typical)
    
    {YELLOW}Issue:{RESET} 4D mode fails
    {GREEN}Solution:{RESET} Ensure b=0 and mask are provided, check dimensions
    
    {YELLOW}Issue:{RESET} Mask geometry mismatch
    {GREEN}Solution:{RESET} Automatic resampling performed, no action needed
    """
    print(help_text)

    
def bias_field_correction_3d(image_path, output_path, mask_path=None, gibbs=False):
    """
    Perform N4 bias field correction on a 3D anatomical image.
    
    Applies the N4ITK algorithm to correct intensity non-uniformity in
    3D MR images (T1w, T2w, FLAIR, etc.). Uses iterative B-spline
    approximation to estimate and remove smooth bias field.
    
    Parameters
    ----------
    image_path : str
        Path to the input 3D image file (.nii or .nii.gz).
    output_path : str
        Path where the bias-corrected image will be saved.
    mask_path : str, optional
        Path to a brain mask image file. If not provided, a mask will
        be automatically generated using Otsu thresholding.
    gibbs : bool, optional
        If True, apply Gibbs ringing removal before bias correction.
    
    Returns
    -------
    str
        Path to the output corrected image (same as output_path).
    
    Raises
    ------
    FileNotFoundError
        If input image or mask file does not exist.
    RuntimeError
        If N4 correction fails.
    
    Examples
    --------
    >>> # Basic usage without mask
    >>> bias_field_correction_3d(
    ...     "T1w.nii.gz",
    ...     "T1w_corrected.nii.gz"
    ... )
    'T1w_corrected.nii.gz'
    
    >>> # With brain mask
    >>> bias_field_correction_3d(
    ...     "T1w.nii.gz",
    ...     "T1w_corrected.nii.gz",
    ...     "brain_mask.nii.gz"
    ... )
    'T1w_corrected.nii.gz'
    
    Notes
    -----
    - Automatically generates mask if not provided
    - Uses ANTs default N4 parameters (convergence=[50,50,50,50])
    - Processing time: 30-120 seconds for typical volumes
    - Preserves input image geometry (spacing, origin, direction)
    
    See Also
    --------
    bias_field_correction_4d : For 4D diffusion images
    run_bias_field_correction : Main entry point with auto-detection
    """
    print(f"{CYAN}Loading 3D image...{RESET}")
    img = ants.image_read(image_path)

    if gibbs:
        if not HAS_DIPY:
            raise ImportError("DIPY is required for Gibbs removal. Please install dipy.")
        print(f"{CYAN}Running Gibbs ringing removal...{RESET}")
        arr = img.numpy()
        gibbs_removal(arr, slice_axis=2, n_points=3, inplace=True, num_processes=1)
        img = img.new_image_like(arr)

    print(f"  Image shape: {img.shape}")
    print(f"  Spacing: {img.spacing}")
    
    if mask_path:
        print(f"{CYAN}Loading mask...{RESET}")
        mask_img = ants.image_read(mask_path)
    else:
        print(f"{YELLOW}No mask provided. Generating mask automatically...{RESET}")
        mask_img = ants.get_mask(img)
    
    print(f"{CYAN}Running N4 bias field correction...{RESET}")
    corrected_img = ants.n4_bias_field_correction(img, mask=mask_img)
    
    print(f"{CYAN}Saving corrected image...{RESET}")
    ants.image_write(corrected_img, output_path)
    
    print(f"{GREEN}3D bias correction completed{RESET}")
    return output_path


def bias_field_correction_4d(image_path, mask_path=None, output_path=None, 
                             b0_path=None, b0_corrected_path=None, shell_dimension=3, gibbs=False):
    """
    Apply N4 bias field correction to a 4D diffusion image.
    
    Computes the bias field from the b=0 volume (or first volume) and
    applies it to all diffusion-weighted volumes. This is more efficient
    than correcting each volume separately and assumes the bias field
    is constant across all b-values.
    
    Parameters
    ----------
    image_path : str
        Path to the input 4D diffusion image (.nii or .nii.gz).
    mask_path : str, optional
        Path to the brain mask image. If None, a mask will be generated
        automatically from the b=0 volume.
    output_path : str
        Path for the output bias-corrected 4D image.
    b0_path : str, optional
        Path to the b=0 image used for bias field estimation. If None,
        the first volume of the 4D image is used.
    b0_corrected_path : str, optional
        Path to save the corrected b=0 image separately.
    shell_dimension : int, default=3
        Dimension along which diffusion volumes are organized (0-indexed).
        For standard NIfTI: dimension 3 (4th dimension).
    gibbs : bool, optional
        If True, apply Gibbs ringing removal before bias correction.
    
    Returns
    -------
    tuple of str
        (path_to_corrected_4d, path_to_corrected_b0)
        Second element is None if b0_corrected_path not provided.
    
    Raises
    ------
    FileNotFoundError
        If input image, mask, or b0 file does not exist.
    ValueError
        If shell_dimension is invalid for the image shape.
    RuntimeError
        If N4 correction or volume processing fails.
    
    Examples
    --------
    >>> # Basic 4D correction with b=0
    >>> bias_field_correction_4d(
    ...     image_path="DWI.nii.gz",
    ...     output_path="DWI_corrected.nii.gz",
    ...     b0_path="b0.nii.gz",
    ...     b0_corrected_path="b0_corrected.nii.gz"
    ... )
    ('DWI_corrected.nii.gz', 'b0_corrected.nii.gz')
    
    >>> # With mask
    >>> bias_field_correction_4d(
    ...     image_path="DWI.nii.gz",
    ...     mask_path="brain_mask.nii.gz",
    ...     output_path="DWI_corrected.nii.gz",
    ...     b0_path="b0.nii.gz",
    ...     b0_corrected_path="b0_corrected.nii.gz",
    ...     shell_dimension=3
    ... )
    ('DWI_corrected.nii.gz', 'b0_corrected.nii.gz')
    
    Notes
    -----
    - Bias field estimated once from b=0 (best SNR)
    - Same bias applied to all volumes (assumes constant bias)
    - Automatic resampling if b0/mask geometry doesn't match
    - Processing time: 1-5 minutes for typical DWI
    - More efficient than per-volume correction
    - Preserves 4D image geometry
    
    Algorithm:
    1. Extract first volume as reference
    2. Load or generate mask
    3. Load or use first volume as b=0
    4. Resample mask and b=0 if needed
    5. Estimate bias field from b=0
    6. Apply bias field to each volume
    7. Reconstruct 4D image
    
    See Also
    --------
    bias_field_correction_3d : For 3D anatomical images
    run_bias_field_correction : Main entry point with auto-detection
    """
    # Read the input images
    print(f"{CYAN}Loading 4D diffusion image...{RESET}")
    img = ants.image_read(image_path)

    if gibbs:
        if not HAS_DIPY:
            raise ImportError("DIPY is required for Gibbs removal. Please install dipy.")
        print(f"{CYAN}Running Gibbs ringing removal on 4D data...{RESET}")
        arr = img.numpy()
        gibbs_removal(arr, slice_axis=2, n_points=3, inplace=True, num_processes=1)
        img = img.new_image_like(arr)

    img_data = img.numpy()
    print(f"  Image shape: {img_data.shape}")
    print(f"  Shell dimension: {shell_dimension}")
    
    if b0_path:
        print(f"{CYAN}Loading b=0 image...{RESET}")
        b0_img = ants.image_read(b0_path)
        if gibbs:
            print(f"{CYAN}Running Gibbs ringing removal on b=0 image...{RESET}")
            arr_b0 = b0_img.numpy()
            gibbs_removal(arr_b0, slice_axis=2, n_points=3, inplace=True, num_processes=1)
            b0_img = b0_img.new_image_like(arr_b0)
    else:
        b0_img = None
    
    # Create dynamic indexing tuple to access the first volume along specified dimension
    vol0_idx = tuple(slice(None) if i != shell_dimension else 0
                    for i in range(len(img_data.shape)))
    
    # Extract the first volume using dynamic indexing
    first_vol_data = img_data[vol0_idx]
    
    # Create a proper 3D reference image
    first_vol_img = ants.from_numpy(
        first_vol_data,
        spacing=img.spacing[:3],
        origin=img.origin[:3],
        direction=img.direction[:3, :3]
    )
    print(f"  First volume shape: {first_vol_img.shape}")
    
    # Handle the mask - either use provided mask or generate one
    if mask_path:
        print(f"{CYAN}Loading mask...{RESET}")
        mask_img = ants.image_read(mask_path)
    else:
        print(f"{YELLOW}No mask provided. Generating mask automatically...{RESET}")
        # Generate mask from b0 or first volume
        mask_img = ants.get_mask(b0_img if b0_img is not None else first_vol_img)
    
    # Resample mask if needed
    if needs_resampling(mask_img, first_vol_img):
        print(f"{YELLOW}Resampling mask to match input image geometry...{RESET}")
        mask_resampled = ants.resample_image_to_target(
            mask_img,
            first_vol_img,
            interp_type='nearestNeighbor'
        )
    else:
        mask_resampled = mask_img
    
    # Handle b0 image
    if b0_img is None:
        print(f"{YELLOW}No b=0 image provided. Using first volume as b=0.{RESET}")
        b0_resampled = first_vol_img
    else:
        # Resample b0 if needed
        if needs_resampling(b0_img, first_vol_img):
            print(f"{YELLOW}Resampling b=0 to match input image geometry...{RESET}")
            b0_resampled = ants.resample_image_to_target(
                b0_img,
                first_vol_img,
                interp_type='linear'
            )
        else:
            b0_resampled = b0_img
    
    print(f"{CYAN}Estimating bias field from b=0 image...{RESET}")
    
    # Calculate the bias field from b0 with specific parameters
    n4_result = ants.n4_bias_field_correction(
        b0_resampled,
        mask=mask_resampled,
        return_bias_field=True,
    )

    bias_field = n4_result['bias']
    print(f"{GREEN}Bias field estimated{RESET}")

    print(f"{CYAN}Applying bias field to all {img_data.shape[shell_dimension]} DWI volumes...{RESET}")
    
    # Apply the same bias field to each volume of the 4D image
    corrected_vols = []
    num_volumes = img_data.shape[shell_dimension]
    
    for i in range(num_volumes):
        if (i + 1) % 10 == 0 or i == 0 or i == num_volumes - 1:
            print(f"  Processing volume {i+1}/{num_volumes}...")
        
        # Create dynamic indexing tuple to access volumes along the specified dimension
        vol_idx = tuple(slice(None) if j != shell_dimension else i 
                        for j in range(len(img_data.shape)))
        
        # Extract the current volume using dynamic indexing
        vol_data = img_data[vol_idx]
        
        # Create ANTs image from the current volume
        vol_ants = ants.from_numpy(
            vol_data,
            spacing=img.spacing[:3],
            origin=img.origin[:3],
            direction=img.direction[:3, :3],
        )
        
        # Divide by the bias field (same as applying the correction)
        corrected_vol = vol_ants / bias_field
        
        # Add to results
        corrected_vols.append(corrected_vol.numpy())
    
    # Stack corrected volumes into 4D array along the specified shell dimension
    print(f"{CYAN}Reconstructing 4D image...{RESET}")
    corrected_array = np.stack(corrected_vols, axis=shell_dimension)
    
    # Create ANTs image from corrected array
    corrected_img = ants.from_numpy(
        corrected_array, 
        spacing=img.spacing, 
        origin=img.origin, 
        direction=img.direction
    )
    
    # Correct b0 separately
    corrected_b0 = b0_resampled / bias_field
    
    # Save results
    print(f"{CYAN}Saving corrected images...{RESET}")
    if b0_corrected_path:
        ants.image_write(corrected_b0, b0_corrected_path)
        print(f"  Corrected b=0: {b0_corrected_path}")
    
    ants.image_write(corrected_img, output_path)
    print(f"  Corrected 4D: {output_path}")
    
    print(f"{GREEN}4D bias correction completed{RESET}")
    
    return output_path, b0_corrected_path if b0_corrected_path else None


def needs_resampling(img1, img2):
    """
    Check if img1 needs to be resampled to match img2's geometry.
    
    Compares spacing, origin, and direction matrix to determine if
    two images are in the same physical space. Images need resampling
    if any of these differ.
    
    Parameters
    ----------
    img1 : ANTsImage
        First image to compare.
    img2 : ANTsImage
        Second image (reference geometry).
    
    Returns
    -------
    bool
        True if img1 needs resampling to match img2, False otherwise.
    
    Examples
    --------
    >>> img = ants.image_read("image.nii.gz")
    >>> mask = ants.image_read("mask.nii.gz")
    >>> if needs_resampling(mask, img):
    ...     mask = ants.resample_image_to_target(mask, img)
    
    Notes
    -----
    - Uses relative tolerance of 1e-6 for floating point comparisons
    - Checks spacing, origin, and direction matrix
    - Flattens direction matrix for comparison
    """
    return not (np.allclose(img1.spacing, img2.spacing, rtol=1e-6) and
               np.allclose(img1.origin, img2.origin, rtol=1e-6) and
               np.allclose(img1.direction.flatten(), img2.direction.flatten(), rtol=1e-6))


def run_bias_field_correction(image_path, output_path, mask_path=None, mode="auto", 
                              b0_path=None, b0_corrected_path=None, shell_dimension=3, gibbs=False):
    """
    Run bias field correction with automatic dimensionality detection.
    
    Main entry point for N4 bias correction. Automatically detects whether
    the input is a 3D anatomical or 4D diffusion image and applies the
    appropriate correction method. Handles mask resampling automatically.
    
    Parameters
    ----------
    image_path : str
        Path to the input image (.nii or .nii.gz).
    output_path : str
        Path for the output bias-corrected image.
    mask_path : str, optional
        Path to brain mask. If None, mask is auto-generated.
        Required for 4D images, optional for 3D.
    mode : {'auto', '3d', '4d'}, default='auto'
        Processing mode:
        - 'auto': Automatically detect from image dimensions
        - '3d': Force 3D processing (anatomical)
        - '4d': Force 4D processing (diffusion)
    b0_path : str, optional
        Path to b=0 image for 4D processing. If None, first volume is used.
        Required for 4D mode.
    b0_corrected_path : str, optional
        Path to save corrected b=0 image (4D mode only).
    shell_dimension : int, default=3
        Dimension for diffusion volumes (4D mode only).
    gibbs : bool, default=False
        If True, apply Gibbs ringing removal.
    
    Returns
    -------
    str or tuple of str
        For 3D: path to corrected image
        For 4D: (path_to_corrected_4d, path_to_corrected_b0)
    
    Raises
    ------
    FileNotFoundError
        If input image, mask, or b0 file does not exist.
    ValueError
        If 4D mode is used without required b0 parameters.
    RuntimeError
        If bias correction fails.
    
    Examples
    --------
    >>> # Auto-detect 3D vs 4D
    >>> run_bias_field_correction(
    ...     "T1w.nii.gz",
    ...     "T1w_corrected.nii.gz"
    ... )
    'T1w_corrected.nii.gz'
    
    >>> # Explicit 3D with mask
    >>> run_bias_field_correction(
    ...     "T1w.nii.gz",
    ...     "T1w_corrected.nii.gz",
    ...     mask_path="brain_mask.nii.gz",
    ...     mode="3d"
    ... )
    'T1w_corrected.nii.gz'
    
    >>> # 4D diffusion image
    >>> run_bias_field_correction(
    ...     "DWI.nii.gz",
    ...     "DWI_corrected.nii.gz",
    ...     mask_path="brain_mask.nii.gz",
    ...     mode="4d",
    ...     b0_path="b0.nii.gz",
    ...     b0_corrected_path="b0_corrected.nii.gz"
    ... )
    ('DWI_corrected.nii.gz', 'b0_corrected.nii.gz')
    
    Notes
    -----
    - Auto mode checks if dimension 4 exists and has size > 1
    - Automatically resamples mask if geometry doesn't match
    - Temporary resampled masks are cleaned up automatically
    - For 4D: bias estimated from b=0 and applied to all volumes
    
    See Also
    --------
    bias_field_correction_3d : 3D-specific implementation
    bias_field_correction_4d : 4D-specific implementation
    """
    # If auto mode, determine if image is 3D or 4D
    print(f"{CYAN}Detecting image dimensionality...{RESET}")
    img = ants.image_read(image_path)
    
    if mode == "auto":
        dims = img.shape
        mode = "4d" if (len(dims) > 3 and dims[3] > 1) else "3d"
        print(f"  Detected: {mode.upper()} image")
    else:
        print(f"  Mode: {mode.upper()} (explicit)")
    
    # Validate 4D mode requirements
    if mode == "4d" and b0_path is None:
        raise ValueError("4D images require a b=0 image. Please provide --b0 <path>")
    if mode == "4d" and b0_path and b0_corrected_path is None:
        raise ValueError("Please provide --b0-output <path> for the corrected b=0 image")
    
    # Check if mask needs resampling
    temp_mask_path = None
    if mask_path:
        mask_img = ants.image_read(mask_path)
        
        # Check if they're in the same physical space
        same_spacing = np.allclose(img.spacing[:3], mask_img.spacing[:3], rtol=1e-6)
        same_origin = np.allclose(img.origin[:3], mask_img.origin[:3], rtol=1e-6)
        same_direction = np.allclose(img.direction[:3,:3], mask_img.direction[:3,:3], rtol=1e-6)
        
        if not (same_spacing and same_origin and same_direction):
            print(f"{YELLOW}Warning: Mask and input image have different physical properties{RESET}")
            print(f"  Image spacing: {img.spacing[:3]}")
            print(f"  Mask spacing: {mask_img.spacing[:3]}")
            print(f"{CYAN}Resampling mask to match input image...{RESET}")
            
            # Create a temporary file for the resampled mask
            temp_dir = tempfile.gettempdir()
            temp_mask_path = os.path.join(temp_dir, f"resampled_mask_{os.path.basename(mask_path)}")
            
            # Resample mask to match input image
            resampled_mask = ants.resample_image_to_target(
                mask_img, 
                img, 
                interp_type='nearestNeighbor'
            )
            ants.image_write(resampled_mask, temp_mask_path)
            
            # Use the resampled mask instead
            mask_path = temp_mask_path
            print(f"{GREEN}Mask resampled{RESET}")
    
    # Process according to mode
    try:
        if mode == "4d":
            return bias_field_correction_4d(
                image_path, mask_path, output_path, 
                b0_path, b0_corrected_path, shell_dimension, gibbs
            )
        else:  # 3d
            return bias_field_correction_3d(image_path, output_path, mask_path, gibbs)
    finally:
        # Clean up temporary files
        if temp_mask_path and os.path.exists(temp_mask_path):
            os.remove(temp_mask_path)
            print(f"{CYAN}Cleaned up temporary files{RESET}")


if __name__ == "__main__":
    # Check if no arguments provided or help requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="N4 Bias Field Correction for 3D anatomical and 4D diffusion MR images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False  # Use custom help
    )
    
    parser.add_argument(
        "--input", "-i", required=True, 
        help="Path to the input image (NIfTI file)."
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output path for the bias-corrected image."
    )
    parser.add_argument(
        "--mask", "-m",
        help="Path to a mask image (required for 4D images, optional for 3D)."
    )
    parser.add_argument(
        "--mode", choices=["3d", "4d", "auto"], default="auto",
        help="Processing mode: 3d=anatomical, 4d=diffusion, auto=detect (default)"
    )
    parser.add_argument(
        "--b0", help="b=0 image path (required for 4D diffusion images)."
    )
    parser.add_argument(
        "--b0-output", help="Path for the output corrected b=0 image (only for 4D DWI)."
    )
    parser.add_argument(
        "--shell-dimension", type=int, default=3,
        help="Dimension along which diffusion volumes are organized (default: 3)."
    )
    parser.add_argument(
        "--gibbs", action="store_true",
        help="Apply Gibbs ringing removal (requires DIPY)."
    )

    args = parser.parse_args()
    
    try:
        # Validate input file exists
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        
        if args.mask and not os.path.exists(args.mask):
            raise FileNotFoundError(f"Mask file not found: {args.mask}")
        
        if args.b0 and not os.path.exists(args.b0):
            raise FileNotFoundError(f"b=0 file not found: {args.b0}")
        
        print(f"{CYAN}Configuration:{RESET}")
        print(f"  Input: {args.input}")
        print(f"  Output: {args.output}")
        print(f"  Mask: {args.mask if args.mask else 'Auto-generate'}")
        print(f"  Mode: {args.mode}")
        print(f"  Gibbs removal: {'Enabled' if args.gibbs else 'Disabled'}")
        if args.b0:
            print(f"  b=0: {args.b0}")
            print(f"  b=0 output: {args.b0_output if args.b0_output else 'Not specified'}")
        print()
        
        # Run correction
        result = run_bias_field_correction(
            args.input, 
            args.output, 
            args.mask, 
            args.mode,
            args.b0,
            args.b0_output,
            args.shell_dimension,
            args.gibbs
        )
        
        print(f"\n{GREEN}{BOLD}Bias field correction completed successfully!{RESET}")
        if isinstance(result, tuple):
            print(f"  Corrected 4D: {result[0]}")
            if result[1]:
                print(f"  Corrected b=0: {result[1]}")
        else:
            print(f"  Corrected image: {result}")
        print()
        
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\n{RED}{BOLD}File not found:{RESET}")
        print(f"  {str(e)}")
        sys.exit(1)
        
    except ValueError as e:
        print(f"\n{RED}{BOLD}Invalid arguments:{RESET}")
        print(f"  {str(e)}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n{RED}{BOLD}Error during bias correction:{RESET}")
        print(f"  {str(e)}")
        print(f"\n{YELLOW}Run 'micaflow bias_correction --help' for usage information.{RESET}")
        sys.exit(1)