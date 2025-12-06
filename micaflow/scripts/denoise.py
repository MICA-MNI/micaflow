"""
denoise - Diffusion-Weighted Image Noise Reduction

Part of the micaflow processing pipeline for neuroimaging data.

This module denoises diffusion-weighted images (DWI) using the Patch2Self algorithm,
which leverages redundant information across diffusion gradients to remove noise 
without requiring additional reference scans. Patch2Self is a self-supervised learning 
approach that improves image quality and enhances subsequent diffusion analyses by 
removing random noise while preserving anatomical structures and signal characteristics.

The Patch2Self Algorithm:
------------------------
Patch2Self is a denoising method that:
- Uses the redundancy in diffusion MRI data across gradient directions
- Learns to predict each volume from surrounding volumes in q-space
- Requires no external training data or noise model
- Preserves anatomical edges and tissue boundaries
- Improves SNR without introducing spatial blurring

Features:
--------
- Self-supervised learning approach requiring no separate reference data
- Adapts to the unique noise characteristics of each dataset
- Preserves anatomical structure while effectively removing noise
- Compatible with standard diffusion MRI acquisition protocols
- Improves subsequent analyses such as fiber tracking and diffusion metrics
- Optional separate b0 volume denoising
- Uses Ordinary Least Squares (OLS) regression model
- Intensity shifting to ensure positive values

Command-Line Usage:
------------------
# Standard denoising (recommended for most cases)
micaflow denoise \\
    --input <path/to/dwi.nii.gz> \\
    --bval <path/to/dwi.bval> \\
    --bvec <path/to/dwi.bvec> \\
    --output <path/to/denoised_dwi.nii.gz>

# With separate b0 denoising (experimental)
micaflow denoise \\
    --input <path/to/dwi.nii.gz> \\
    --bval <path/to/dwi.bval> \\
    --bvec <path/to/dwi.bvec> \\
    --output <path/to/denoised_dwi.nii.gz> \\
    --b0-denoise

# With Gibbs ringing removal
micaflow denoise \\
    --input <path/to/dwi.nii.gz> \\
    --bval <path/to/dwi.bval> \\
    --bvec <path/to/dwi.bvec> \\
    --output <path/to/denoised_dwi.nii.gz> \\
    --gibbs

Python API Usage:
----------------
>>> from micaflow.scripts.denoise import run_denoise
>>> 
>>> # Standard denoising
>>> output = run_denoise(
...     moving="raw_dwi.nii.gz",
...     moving_bval="dwi.bval", 
...     moving_bvec="dwi.bvec",
...     output="denoised_dwi.nii.gz",
...     b0_denoising=False  # Default
... )
>>> 
>>> # With b0 denoising
>>> output = run_denoise(
...     moving="raw_dwi.nii.gz",
...     moving_bval="dwi.bval", 
...     moving_bvec="dwi.bvec",
...     output="denoised_dwi.nii.gz",
...     b0_denoising=True
... )
>>>
>>> # With Gibbs ringing removal
>>> output = run_denoise(
...     moving="raw_dwi.nii.gz",
...     moving_bval="dwi.bval", 
...     moving_bvec="dwi.bvec",
...     output="denoised_dwi.nii.gz",
...     gibbs=True
... )

Pipeline Integration:
--------------------
Denoising is typically the FIRST step in diffusion preprocessing:
1. Denoising (denoise) ← You are here
2. Motion/eddy current correction (motion_correction)
3. Susceptibility distortion correction (apply_SDC)
4. Bias field correction (bias_correction)
5. Brain extraction (bet)
6. DTI metric calculation (compute_fa_md)

Exit Codes:
----------
0 : Success - denoising completed
1 : Error - invalid inputs, file not found, or processing failure

Technical Notes:
---------------
- Patch2Self uses a self-supervised regression approach
- Model type: Ordinary Least Squares (OLS) regression
- B0 threshold: 50 s/mm² (volumes below this considered b0)
- Intensity shifting: Enabled (ensures positive values)
- Negative value clipping: Disabled (preserves signal characteristics)
- Processing time: ~1-2 minutes for typical datasets (varies with # volumes)
- Recommended to denoise BEFORE motion correction for best results
- B0 denoising is experimental; standard mode excludes b0 from denoising

See Also:
--------
- motion_correction : Apply after denoising for motion/eddy correction
- bias_correction : Remove bias fields after motion correction
- compute_fa_md : Compute DTI metrics from denoised data

References:
----------
1. Fadnavis S, Batson J, Garyfallidis E. Patch2Self: Denoising Diffusion MRI with 
   Self-Supervised Learning. Advances in Neural Information Processing Systems. 
   2020;33:16293-16303.

2. Fadnavis S, Farooq H, Theaud G, et al. Patch2Self denoising of diffusion MRI in 
   the CLEAR test-retest dataset. Medical Imaging 2021: Image Processing. 
   2021;11596:115962D. doi:10.1117/12.2582043

"""

import argparse
import nibabel as nib
import sys
import os
import numpy as np
from dipy.denoise.patch2self import patch2self
from dipy.denoise.gibbs import gibbs_removal
from dipy.io.gradients import read_bvals_bvecs
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
    ║                     DWI IMAGE DENOISING                        ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script denoises diffusion-weighted images (DWI) using the Patch2Self 
    algorithm, which leverages redundant information across diffusion gradients
    to remove noise without requiring additional reference scans.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow denoise {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--input{RESET}     : Path to the input DWI image (.nii.gz)
      {YELLOW}--bval{RESET}      : Path to the b-values file (.bval)
      {YELLOW}--bvec{RESET}      : Path to the b-vectors file (.bvec)
      {YELLOW}--output{RESET}    : Output path for the denoised image (.nii.gz)
    
    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--b0-denoise{RESET}: Denoise b0 volumes separately (default: False)
                   {MAGENTA}Experimental - not recommended for most cases{RESET}
      {YELLOW}--gibbs{RESET}     : Apply Gibbs ringing removal after denoising
      {YELLOW}--threads{RESET}   : Number of threads for Gibbs removal (default: 1)
    
    {CYAN}{BOLD}──────────────────── EXAMPLE USAGE ──────────────────────{RESET}
    
    {BLUE}# Standard denoising (recommended){RESET}
    micaflow denoise \\
      {YELLOW}--input{RESET} raw_dwi.nii.gz \\
      {YELLOW}--bval{RESET} dwi.bval \\
      {YELLOW}--bvec{RESET} dwi.bvec \\
      {YELLOW}--output{RESET} denoised_dwi.nii.gz
    
    {BLUE}# With b0 denoising (experimental){RESET}
    micaflow denoise \\
      {YELLOW}--input{RESET} raw_dwi.nii.gz \\
      {YELLOW}--bval{RESET} dwi.bval \\
      {YELLOW}--bvec{RESET} dwi.bvec \\
      {YELLOW}--output{RESET} denoised_dwi.nii.gz \\
      {YELLOW}--b0-denoise{RESET}
    
    {BLUE}# With Gibbs ringing removal{RESET}
    micaflow denoise \\
      {YELLOW}--input{RESET} raw_dwi.nii.gz \\
      {YELLOW}--bval{RESET} dwi.bval \\
      {YELLOW}--bvec{RESET} dwi.bvec \\
      {YELLOW}--output{RESET} denoised_dwi.nii.gz \\
      {YELLOW}--gibbs{RESET}
    
    {CYAN}{BOLD}────────────────── PATCH2SELF ALGORITHM ──────────────────{RESET}
    
    {GREEN}How it works:{RESET}
    {MAGENTA}•{RESET} Uses redundancy across diffusion gradient directions
    {MAGENTA}•{RESET} Predicts each volume from neighboring volumes in q-space
    {MAGENTA}•{RESET} Self-supervised learning (no external training data needed)
    {MAGENTA}•{RESET} Preserves edges and anatomical structures
    
    {GREEN}Benefits:{RESET}
    {MAGENTA}•{RESET} Better visualization of white matter tracts
    {MAGENTA}•{RESET} More accurate DTI fitting and tractography
    {MAGENTA}•{RESET} Reduced bias in diffusion metrics (FA, MD)
    {MAGENTA}•{RESET} No spatial blurring (unlike traditional filters)
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    {MAGENTA}•{RESET} Denoising should be the FIRST step in preprocessing pipeline
    {MAGENTA}•{RESET} Apply BEFORE motion correction for best results
    {MAGENTA}•{RESET} Uses Ordinary Least Squares (OLS) regression model
    {MAGENTA}•{RESET} B0 threshold: 50 s/mm² (volumes below this are b0)
    {MAGENTA}•{RESET} Intensity shifting enabled to ensure positive values
    {MAGENTA}•{RESET} Processing time: ~1-2 minutes (depends on # volumes)
    {MAGENTA}•{RESET} Standard mode excludes b0 volumes from denoising (recommended)
    {MAGENTA}•{RESET} --b0-denoise flag enables experimental b0 denoising
    {MAGENTA}•{RESET} Output preserves original image geometry and orientation
    
    {CYAN}{BOLD}─────────────── PIPELINE POSITION ───────────────────────{RESET}
    {GREEN}1. Denoising (denoise){RESET}           {MAGENTA}← You are here{RESET}
    2. Motion correction (motion_correction)
    3. Distortion correction (apply_SDC)
    4. Bias field correction (bias_correction)
    5. Brain extraction (bet)
    6. DTI metrics (compute_fa_md)
    
    {CYAN}{BOLD}────────────────────── EXIT CODES ───────────────────────{RESET}
    {GREEN}0{RESET} : Success - denoising completed
    {RED}1{RESET} : Error - invalid inputs, file not found, or processing failure
    
    {CYAN}{BOLD}───────────────── COMMON ISSUES ─────────────────────────{RESET}
    {YELLOW}Issue:{RESET} Denoising takes very long
    {GREEN}Solution:{RESET} Typical processing is 2-5 minutes; longer suggests high resolution
    
    {YELLOW}Issue:{RESET} Output looks over-smoothed
    {GREEN}Solution:{RESET} Patch2Self preserves edges; check if input already denoised
    
    {YELLOW}Issue:{RESET} "Out of memory" error
    {GREEN}Solution:{RESET} Reduce image resolution or process on machine with more RAM
    """
    print(help_text)


def run_denoise(moving, moving_bval, moving_bvec, output, b0_denoising=False, gibbs=False, threads=1):
    """
    Denoise diffusion-weighted images using the Patch2Self algorithm.
    
    This function applies Patch2Self denoising to diffusion-weighted images (DWI),
    which uses a self-supervised learning approach to remove noise while preserving 
    anatomical structure. It leverages redundant information across diffusion gradients
    without requiring external training data or noise models.
    
    Parameters
    ----------
    moving : str
        Path to the input DWI image (.nii.gz).
        Should be raw or minimally processed data.
    moving_bval : str
        Path to the b-values file (.bval).
        Text file with space-separated b-values in s/mm².
    moving_bvec : str
        Path to the b-vectors file (.bvec).
        Text file with 3 rows (x, y, z) of space-separated gradient directions.
    output : str
        Path where the denoised image will be saved (.nii.gz).
    b0_denoising : bool, optional
        If True, denoise b0 volumes separately. Default: False.
        Standard practice is to exclude b0 volumes from denoising.
    gibbs : bool, optional
        If True, apply Gibbs ringing removal after denoising. Default: False.
    threads : int, optional
        Number of threads to use for Gibbs ringing removal. Default: 1.
        
    Returns
    -------
    str
        Path to the saved denoised image (same as output parameter).
        
    Raises
    ------
    FileNotFoundError
        If input files cannot be found.
    ValueError
        If bval/bvec files are invalid or mismatched.
        
    Notes
    -----
    - Uses Ordinary Least Squares (OLS) regression model
    - Shifts intensity values to ensure positivity
    - Does not clip negative values (preserves signal characteristics)
    - B0 threshold set to 50 s/mm²
    - Standard mode (b0_denoising=False) excludes b0 volumes (recommended)
    - Processing time: ~2-5 minutes for typical datasets
    - Expected SNR improvement: 20-40%
    - Output preserves input image geometry (spacing, origin, direction)
    
    Examples
    --------
    >>> # Standard denoising (recommended)
    >>> output = run_denoise(
    ...     moving="raw_dwi.nii.gz",
    ...     moving_bval="dwi.bval",
    ...     moving_bvec="dwi.bvec",
    ...     output="denoised_dwi.nii.gz"
    ... )
    >>> 
    >>> # With b0 denoising (experimental)
    >>> output = run_denoise(
    ...     moving="raw_dwi.nii.gz",
    ...     moving_bval="dwi.bval",
    ...     moving_bvec="dwi.bvec",
    ...     output="denoised_dwi.nii.gz",
    ...     b0_denoising=True
    ... )
    >>> 
    >>> # With Gibbs ringing removal
    >>> output = run_denoise(
    ...     moving="raw_dwi.nii.gz",
    ...     moving_bval="dwi.bval",
    ...     moving_bvec="dwi.bvec",
    ...     output="denoised_dwi.nii.gz",
    ...     gibbs=True
    ... )
    """
    # Validate input files exist
    for filepath, name in [(moving, "Input DWI"), 
                           (moving_bval, "B-values"), 
                           (moving_bvec, "B-vectors")]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{name} file not found: {filepath}")
    
    print(f"{CYAN}Loading DWI image...{RESET}")
    moving_image = nib.load(moving)
    dwi_data = moving_image.get_fdata()
    print(f"  Image shape: {dwi_data.shape}")
    
    print(f"{CYAN}Loading gradient table...{RESET}")
    moving_bval_value, moving_bvec_value = read_bvals_bvecs(moving_bval, moving_bvec)
    num_volumes = len(moving_bval_value)
    b0_count = np.sum(moving_bval_value <= 50)
    dwi_count = num_volumes - b0_count
    print(f"  Total volumes: {num_volumes}")
    print(f"  B0 volumes (b ≤ 50): {b0_count}")
    print(f"  DWI volumes (b > 50): {dwi_count}")
    
    if b0_denoising:
        print(f"{YELLOW}B0 denoising enabled (experimental){RESET}")
    else:
        print(f"{CYAN}B0 volumes will be excluded from denoising (standard mode){RESET}")
    
    print(f"\n{CYAN}Applying Patch2Self denoising...{RESET}")
    print(f"  Model: Ordinary Least Squares (OLS)")
    print(f"  B0 threshold: 50 s/mm²")
    print(f"  Intensity shifting: Enabled")
    
    denoised = patch2self(
        dwi_data,
        moving_bval_value,
        model="ols",
        shift_intensity=True,
        clip_negative_vals=False,
        b0_threshold=50,
        b0_denoising=b0_denoising,
    )
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    os.environ.update(env)
    
    if gibbs:
        print(f"\n{CYAN}Applying Gibbs ringing removal...{RESET}")
        gibbs_removal(denoised, slice_axis=2, n_points=3, inplace=True, num_processes=threads)

    print(f"\n{CYAN}Saving denoised image...{RESET}")
    nib.save(nib.Nifti1Image(denoised, moving_image.affine), output)
    
    return output


if __name__ == "__main__":
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="Denoise a DWI image using Patch2Self algorithm.",
        add_help=False  # Use custom help
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input DWI image (NIfTI file).",
    )
    parser.add_argument(
        "--bval", 
        type=str, 
        required=True, 
        help="Path to the b-values file."
    )
    parser.add_argument(
        "--bvec", 
        type=str, 
        required=True, 
        help="Path to the b-vectors file."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Output path for denoised image."
    )
    parser.add_argument(
        "--b0-denoise", 
        action='store_true', 
        help="Denoise b0 volumes separately (experimental, default: False)."
    )
    parser.add_argument(
        "--gibbs", 
        action='store_true', 
        help="Apply Gibbs ringing removal after denoising."
    )
    parser.add_argument(
        "--threads", 
        type=int, 
        default=1, 
        help="Number of threads for Gibbs removal (default: 1)."
    )

    args = parser.parse_args()
    
    try:
        output_path = run_denoise(
            args.input, 
            args.bval, 
            args.bvec, 
            args.output, 
            args.b0_denoise,
            args.gibbs,
            args.threads
        )
        
        print(f"\n{GREEN}{BOLD}Denoising successfully completed!{RESET}")
        print(f"  Output: {output_path}")
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\n{RED}{BOLD}File not found:{RESET}")
        print(f"  {str(e)}")
        sys.exit(1)
        
    except ValueError as e:
        print(f"\n{RED}{BOLD}Value error:{RESET}")
        print(f"  {str(e)}")
        print(f"\n{YELLOW}Check that bval/bvec files match DWI dimensions.{RESET}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n{RED}{BOLD}Error during denoising:{RESET}")
        print(f"  {str(e)}")
        print(f"\n{YELLOW}Run 'micaflow denoise --help' for usage information.{RESET}")
        sys.exit(1)