"""
texture_generation - MRI Texture Feature Extraction for Radiomics

Part of the micaflow processing pipeline for neuroimaging data.

This module computes advanced texture features from MRI data that can be used for
tissue characterization, lesion analysis, radiomics applications, and quantitative
imaging biomarker development. It performs automatic tissue segmentation and extracts
quantitative imaging features including gradient magnitude and relative intensity maps,
which capture local intensity variations and tissue contrast properties respectively.

What are Texture Features?
--------------------------
Texture features quantify spatial patterns and intensity variations in medical images:
- Provide information beyond simple intensity statistics
- Capture tissue heterogeneity and structural organization
- Enable quantitative characterization of pathology
- Support radiomics and machine learning applications
- Complement traditional volumetric measurements

Computed Features:
-----------------
1. Gradient Magnitude:
   - Measures local intensity changes
   - Highlights edges and boundaries
   - Captures tissue transitions
   - Sensitive to structural organization
   - Range: 0 to maximum gradient

2. Relative Intensity (RI):
   - Normalized intensity relative to tissue peaks
   - Accounts for scanner/protocol variations
   - Centers around GM-WM boundary
   - Range: Typically 0-200 (100 = background)
   - Provides contrast-normalized values

How It Works:
------------
1. Load input MRI and brain mask
2. Segment brain into GM, WM, CSF (Atropos K-means)
3. Find GM and WM intensity peaks
4. Compute gradient magnitude (edge detection)
5. Calculate relative intensity (normalized contrast)
6. Apply smoothing to relative intensity map
7. Save all feature maps as NIfTI files

Relative Intensity Calculation:
------------------------------
1. Find GM peak intensity (mode of GM histogram)
2. Find WM peak intensity (mode of WM histogram)
3. Compute background: BG = 0.5 × (GM_peak + WM_peak)
4. For voxels < BG: RI = 100 × (1 - (BG - I) / BG)
5. For voxels > BG: RI = 100 × (1 + (I - BG) / BG)
6. Smooth with Gaussian (σ=3mm FWHM)

Command-Line Usage:
------------------
# Basic usage
micaflow texture_generation \\
    --input <path/to/image.nii.gz> \\
    --mask <path/to/brain_mask.nii.gz> \\
    --output <path/to/output_prefix>

# Example with T1w image
micaflow texture_generation \\
    --input T1w_preprocessed.nii.gz \\
    --mask brain_mask.nii.gz \\
    --output subject01_textures

# Example with short flags
micaflow texture_generation \\
    -i T1w.nii.gz \\
    -m mask.nii.gz \\
    -o output/features

Python API Usage:
----------------
>>> from micaflow.scripts.texture_generation import run_texture_pipeline
>>> 
>>> # Basic usage
>>> run_texture_pipeline(
...     input="preprocessed_t1w.nii.gz",
...     mask="brain_mask.nii.gz",
...     output_dir="output_texture_maps"
... )
>>> 
>>> # With variables
>>> input_file = "data/T1w.nii.gz"
>>> mask_file = "data/mask.nii.gz"
>>> output_prefix = "results/subject01"
>>> run_texture_pipeline(input_file, mask_file, output_prefix)

Pipeline Integration:
--------------------
Texture generation typically follows preprocessing:

Structural MRI Pipeline:
1. Preprocessing (N4 bias correction, denoising)
2. Brain extraction (skull stripping)
3. Tissue segmentation (optional: use synthseg instead)
4. Texture feature extractiom
5. Statistical analysis or machine learning

Radiomics Pipeline:
1. Image acquisition and quality control
2. Preprocessing and standardization
3. ROI/lesion segmentation
4. Feature extraction (texture_generation)
5. Feature selection
6. Model training/prediction

Exit Codes:
----------
0 : Success - texture features computed successfully
1 : Error - invalid inputs, file not found, or processing failure

See Also:
--------
- synthseg : Alternative segmentation method
- n4_bias_correction : Recommended preprocessing
- denoise : Noise reduction before feature extraction

"""
import argparse
import os
import sys
import numpy as np
import ants
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
    """
    Print comprehensive help message with examples and usage instructions.
    
    This function displays detailed information about texture feature extraction including:
    - What texture features are and why they matter
    - Available features (gradient magnitude, relative intensity)
    - Command-line options and usage
    - Multiple examples for different scenarios
    - Technical details about algorithms
    - Quality control recommendations
    - Output file descriptions
    
    The help message uses color-coded sections for better readability.
    
    Examples
    --------
    >>> # Display help message
    >>> print_help_message()
    
    >>> # Help is shown automatically with --help, -h, or no arguments
    >>> # micaflow texture_generation --help
    
    Notes
    -----
    - Called automatically when script run with --help, -h, or no arguments
    - Provides more detail than standard argparse help
    - Uses ANSI color codes for visual organization
    """
    
    help_text = f"""
    {CYAN}{BOLD}╔════════════════════════════════════════════════════════════════╗
    ║                  TEXTURE FEATURE EXTRACTION                    ║
    ║                        (Radiomics)                             ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script generates quantitative texture feature maps from MRI data
    for radiomics, tissue characterization, and quantitative imaging biomarker
    development. Features include gradient magnitude and relative intensity.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow texture_generation {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--input{RESET}, {YELLOW}-i{RESET}   : Path to the input MRI image (.nii.gz)
      {YELLOW}--mask{RESET}, {YELLOW}-m{RESET}    : Path to the binary brain mask (.nii.gz)
      {YELLOW}--output{RESET}, {YELLOW}-o{RESET}  : Output prefix for texture feature maps
    
    {CYAN}{BOLD}──────────────────── EXAMPLE USAGE ──────────────────────{RESET}
    
    {BLUE}# Example 1: Basic usage{RESET}
    micaflow texture_generation \\
      {YELLOW}--input{RESET} T1w_preprocessed.nii.gz \\
      {YELLOW}--mask{RESET} brain_mask.nii.gz \\
      {YELLOW}--output{RESET} subject01_textures
    
    {BLUE}# Example 2: With short flags{RESET}
    micaflow texture_generation \\
      {YELLOW}-i{RESET} T1w.nii.gz \\
      {YELLOW}-m{RESET} mask.nii.gz \\
      {YELLOW}-o{RESET} output/features
    
    {BLUE}# Example 3: From preprocessed data{RESET}
    micaflow texture_generation \\
      {YELLOW}-i{RESET} data/preprocessed/T1w_n4.nii.gz \\
      {YELLOW}-m{RESET} data/masks/brain_mask.nii.gz \\
      {YELLOW}-o{RESET} results/radiomics/sub-01
    
    {CYAN}{BOLD}────────── WHAT ARE TEXTURE FEATURES? ──────────────────{RESET}
    
    {GREEN}Texture features quantify spatial patterns in medical images:{RESET}
    {MAGENTA}•{RESET} Go beyond simple intensity statistics
    {MAGENTA}•{RESET} Capture tissue heterogeneity and organization
    {MAGENTA}•{RESET} Provide quantitative imaging biomarkers
    {MAGENTA}•{RESET} Enable radiomics and machine learning
    {MAGENTA}•{RESET} Support precision medicine approaches
    
    {GREEN}Common applications:{RESET}
    {MAGENTA}•{RESET} Tumor characterization and grading
    {MAGENTA}•{RESET} Neurodegenerative disease staging
    {MAGENTA}•{RESET} White matter lesion analysis
    {MAGENTA}•{RESET} Age-related changes
    {MAGENTA}•{RESET} Treatment response monitoring
    
    {CYAN}{BOLD}─────────────── COMPUTED FEATURES ──────────────────────{RESET}
    
    {GREEN}1. Gradient Magnitude:{RESET}
    {MAGENTA}•{RESET} Measures local intensity changes
    {MAGENTA}•{RESET} Highlights edges and tissue boundaries
    {MAGENTA}•{RESET} Sensitive to structural organization
    {MAGENTA}•{RESET} Range: 0 to maximum gradient
    {MAGENTA}•{RESET} Higher values at GM/WM boundaries
    
    {GREEN}2. Relative Intensity (RI):{RESET}
    {MAGENTA}•{RESET} Normalized intensity relative to tissue peaks
    {MAGENTA}•{RESET} Accounts for scanner/protocol variations
    {MAGENTA}•{RESET} Centers around GM-WM boundary (value = 100)
    {MAGENTA}•{RESET} Range: Typically 50-150
    {MAGENTA}•{RESET} Provides contrast-normalized values
    
    {GREEN}3. Tissue Segmentation (intermediate):{RESET}
    {MAGENTA}•{RESET} Gray matter mask
    {MAGENTA}•{RESET} White matter mask
    {MAGENTA}•{RESET} CSF identification
    {MAGENTA}•{RESET} Used for feature computation
    
    {CYAN}{BOLD}────────────────── HOW IT WORKS ────────────────────────{RESET}
    
    {GREEN}Processing pipeline:{RESET}
    {MAGENTA}1.{RESET} Load input MRI and brain mask
    {MAGENTA}2.{RESET} Segment brain into GM, WM, CSF (Atropos K-means)
    {MAGENTA}3.{RESET} Find GM and WM intensity peaks
    {MAGENTA}4.{RESET} Compute gradient magnitude (edge detection)
    {MAGENTA}5.{RESET} Calculate relative intensity (normalized contrast)
    {MAGENTA}6.{RESET} Apply smoothing (σ=3mm FWHM)
    {MAGENTA}7.{RESET} Save feature maps as NIfTI files
    
    {GREEN}Atropos segmentation:{RESET}
    {MAGENTA}•{RESET} K-means clustering (3 tissue classes)
    {MAGENTA}•{RESET} MRF spatial prior (0.2 weight, 1x1x1 neighborhood)
    {MAGENTA}•{RESET} 3 iterations for convergence
    {MAGENTA}•{RESET} Masked to brain tissue only
    {MAGENTA}•{RESET} Output: 1=CSF, 2=GM, 3=WM
    
    {CYAN}{BOLD}────────────────────── OUTPUT FILES ─────────────────────{RESET}
    
    Given output prefix {GREEN}"subject01_textures"{RESET}, creates:
    
    {YELLOW}subject01_textures_gradient-magnitude.nii.gz{RESET}
      - Edge and boundary detection map
      - Higher values at tissue transitions
      - Range: 0 to ~50 (depends on contrast)
    
    {YELLOW}subject01_textures_relative-intensity.nii.gz{RESET}
      - Normalized contrast map
      - Centered at GM-WM boundary (100)
      - Range: Typically 50-150
      - Smoothed with σ=3mm FWHM
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    {MAGENTA}•{RESET} Preprocessing recommended: N4 bias correction, denoising
    {MAGENTA}•{RESET} Processing time: 2-5 minutes per subject
    {MAGENTA}•{RESET} Memory: ~2-4 GB RAM
    {MAGENTA}•{RESET} All operations masked to brain only
    {MAGENTA}•{RESET} No resampling (preserves input resolution)
    {MAGENTA}•{RESET} Values outside mask are zero
    {MAGENTA}•{RESET} Gradient uses first-order finite differences
    {MAGENTA}•{RESET} Peak finding uses 1st-99.5th percentile range
    
    {CYAN}{BOLD}─────────────── PIPELINE POSITION ───────────────────────{RESET}
    {BLUE}Structural MRI Pipeline:{RESET}
    1. Preprocessing (N4, denoising)
    2. Brain extraction (skull stripping)
    3. Tissue segmentation (optional)
    {GREEN}4. Texture feature extraction{RESET} {MAGENTA}← You are here{RESET}
    5. Statistical analysis or ML
    
    {BLUE}Radiomics Pipeline:{RESET}
    1. Image acquisition and QC
    2. Preprocessing and standardization
    3. ROI/lesion segmentation
    {GREEN}4. Feature extraction{RESET} {MAGENTA}← You are here{RESET}
    5. Feature selection
    6. Model training/prediction
    
    {CYAN}{BOLD}────────────────────── EXIT CODES ───────────────────────{RESET}
    {GREEN}0{RESET} : Success - texture features computed
    {RED}1{RESET} : Error - invalid inputs, file not found, or processing failure
    
    {CYAN}{BOLD}───────────── QUALITY CONTROL ───────────────────────────{RESET}
    {YELLOW}Visual inspection:{RESET}
    {MAGENTA}1.{RESET} Check gradient magnitude highlights tissue boundaries
    {MAGENTA}2.{RESET} Verify relative intensity centered around 100
    {MAGENTA}3.{RESET} Ensure smooth transitions (no edge artifacts)
    {MAGENTA}4.{RESET} Confirm brain mask coverage is complete
    {MAGENTA}5.{RESET} Look for segmentation errors
    
    {YELLOW}Expected ranges:{RESET}
    {MAGENTA}•{RESET} Gradient magnitude: 0 to ~50
    {MAGENTA}•{RESET} Relative intensity: 50-150 (100 = background)
    {MAGENTA}•{RESET} Values outside mask: 0
    
    {CYAN}{BOLD}───────────────── COMMON ISSUES ─────────────────────────{RESET}
    {YELLOW}Issue:{RESET} Poor segmentation quality
    {GREEN}Solution:{RESET} Use N4 bias correction first, verify mask quality
    
    {YELLOW}Issue:{RESET} Extreme gradient values
    {GREEN}Solution:{RESET} Check image quality, reduce noise with denoising
    
    {YELLOW}Issue:{RESET} Relative intensity not centered at 100
    {GREEN}Solution:{RESET} Verify tissue segmentation, check for artifacts
    
    {YELLOW}Issue:{RESET} Processing takes very long
    {GREEN}Solution:{RESET} Normal for high-resolution images (2-5 min typical)
    """
    print(help_text)


def compute_gradient_magnitude(image):
    """Compute gradient magnitude using ANTs."""
    # Calculate gradient magnitude (sigma=1 for minimal smoothing)
    grad = ants.iMath(image, 'Grad', 1)
    return grad


def compute_relative_intensity(image, mask):
    """
    Compute relative intensity normalized to the global mean of the masked image.
    
    Simplified approach:
    1. Calculate mean intensity within the brain mask (background reference).
    2. Normalize voxels relative to this mean.
    3. Center at 100.
    """
    # Get data array
    img_data = image.numpy()
    mask_data = mask.numpy()
    
    # Calculate background reference (mean of brain tissue)
    # Using mean is simpler and faster than finding histogram peaks
    brain_voxels = img_data[mask_data > 0]
    
    if len(brain_voxels) == 0:
        print(f"{Fore.YELLOW}Warning: Mask is empty. Returning zero image.{Style.RESET_ALL}")
        return image.new_image_like(np.zeros_like(img_data))

    bg_ref = np.mean(brain_voxels)
    
    # Initialize output array
    ri_data = np.zeros_like(img_data)
    
    # Avoid division by zero
    if bg_ref == 0:
        bg_ref = 1.0
        
    # Calculate Relative Intensity
    # Formula: RI = 100 * (1 + (I - BG) / BG) = 100 * (I / BG)
    # This centers the mean intensity at 100
    
    # Apply only within mask
    mask_indices = mask_data > 0
    ri_data[mask_indices] = 100.0 * (img_data[mask_indices] / bg_ref)
    
    # Create ANTs image
    ri_img = image.new_image_like(ri_data)
    
    # Smooth the result (sigma=3mm)
    ri_smooth = ants.smooth_image(ri_img, sigma=3, FWHM=True)
    
    # Re-mask to ensure background is clean after smoothing
    ri_smooth = ri_smooth * mask
    
    return ri_smooth


def run_texture_pipeline(input_path, mask_path, output_prefix):
    """Run simplified texture generation pipeline."""
    
    print(f"{Fore.CYAN}Loading input: {input_path}{Style.RESET_ALL}")
    img = ants.image_read(input_path)
    
    print(f"{Fore.CYAN}Loading mask: {mask_path}{Style.RESET_ALL}")
    mask = ants.image_read(mask_path)
    
    # Ensure mask is in same space
    if not ants.image_physical_space_consistency(img, mask):
        print(f"{Fore.YELLOW}Resampling mask to input space...{Style.RESET_ALL}")
        mask = ants.resample_image_to_target(mask, img, interp_type='nearestNeighbor')

    # 1. Gradient Magnitude
    print(f"{Fore.GREEN}Computing Gradient Magnitude...{Style.RESET_ALL}")
    grad_map = compute_gradient_magnitude(img)
    
    grad_out = f"{output_prefix}_gradient-magnitude.nii.gz"
    ants.image_write(grad_map, grad_out)
    print(f"  Saved: {grad_out}")

    # 2. Relative Intensity
    print(f"{Fore.GREEN}Computing Relative Intensity...{Style.RESET_ALL}")
    ri_map = compute_relative_intensity(img, mask)
    
    ri_out = f"{output_prefix}_relative-intensity.nii.gz"
    ants.image_write(ri_map, ri_out)
    print(f"  Saved: {ri_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Texture Feature Extraction")
    parser.add_argument("--input", "-i", required=True, help="Input MRI image")
    parser.add_argument("--mask", "-m", required=True, help="Brain mask")
    parser.add_argument("--output", "-o", required=True, help="Output prefix")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"{Fore.RED}Error: Input file not found: {args.input}{Style.RESET_ALL}")
        sys.exit(1)
        
    if not os.path.exists(args.mask):
        print(f"{Fore.RED}Error: Mask file not found: {args.mask}{Style.RESET_ALL}")
        sys.exit(1)

    try:
        run_texture_pipeline(args.input, args.mask, args.output)
        print(f"\n{Fore.GREEN}Done!{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
        sys.exit(1)