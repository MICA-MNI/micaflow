"""
compute_fa_md - Diffusion Tensor Imaging Metrics Calculator

Part of the micaflow processing pipeline for neuroimaging data.

This module computes diffusion tensor imaging (DTI) scalar metrics, specifically 
Fractional Anisotropy (FA) and Mean Diffusivity (MD), from preprocessed diffusion-weighted 
images (DWI). FA quantifies the directional preference of water diffusion, serving as a 
marker of white matter integrity, while MD represents the overall magnitude of diffusion. 
These metrics are essential for investigating white matter microstructure and are widely 
used in clinical and research neuroimaging.

DTI Metrics:
-----------
Fractional Anisotropy (FA):
  - Measures directionality of diffusion (0 = isotropic, 1 = highly directional)
  - Sensitive to white matter integrity, myelination, and axonal density
  - Typical values:
    * White matter tracts: 0.4 - 0.8 (corpus callosum ~0.7, internal capsule ~0.6)
    * Gray matter: 0.1 - 0.3
    * CSF: ~0.0 - 0.1

Mean Diffusivity (MD):
  - Average diffusion magnitude across all directions (units: mm²/s × 10⁻³)
  - Reflects overall water content and membrane density
  - Typical values:
    * White matter: 0.6 - 0.9 × 10⁻³ mm²/s
    * Gray matter: 0.7 - 1.0 × 10⁻³ mm²/s
    * CSF: 2.5 - 3.5 × 10⁻³ mm²/s

Features:
--------
- Computes DTI model using robust tensor fitting from DIPY
- Compatible with standard neuroimaging file formats (NIfTI)
- Preserves image header and spatial information in output files

Command-Line Usage:
------------------
# Without b0 volume (already included in input)
micaflow compute_fa_md \\
    --input <path/to/dwi.nii.gz> \\
    --mask <path/to/brain_mask.nii.gz> \\
    --bval <path/to/dwi.bval> \\
    --bvec <path/to/dwi.bvec> \\
    --output-fa <path/to/fa_map.nii.gz> \\
    --output-md <path/to/md_map.nii.gz>

# With separate b0 volume to merge
micaflow compute_fa_md \\
    --input <path/to/dwi.nii.gz> \\
    --mask <path/to/brain_mask.nii.gz> \\
    --bval <path/to/dwi.bval> \\
    --bvec <path/to/dwi.bvec> \\
    --b0-volume <path/to/b0.nii.gz> \\
    --b0-bval <path/to/b0.bval> \\
    --b0-bvec <path/to/b0.bvec> \\
    --b0-index 0 \\
    --output-fa <path/to/fa_map.nii.gz> \\
    --output-md <path/to/md_map.nii.gz>

Python API Usage:
----------------
>>> from micaflow.scripts.compute_fa_md import compute_fa_md
>>> 
>>> # Standard usage
>>> fa_path, md_path = compute_fa_md(
...     bias_corr_path="corrected_dwi.nii.gz",
...     mask_path="brain_mask.nii.gz",
...     moving_bval="dwi.bval",
...     moving_bvec="dwi.bvec",
...     fa_path="fa.nii.gz",
...     md_path="md.nii.gz"
... )
>>> 
>>> # With b0 merging
>>> fa_path, md_path = compute_fa_md(
...     bias_corr_path="corrected_dwi.nii.gz",
...     mask_path="brain_mask.nii.gz",
...     moving_bval="dwi.bval",
...     moving_bvec="dwi.bvec",
...     fa_path="fa.nii.gz",
...     md_path="md.nii.gz",
...     b0_volume="b0.nii.gz",
...     b0_bval="b0.bval",
...     b0_bvec="b0.bvec",
...     b0_index=0
... )

Pipeline Integration:
--------------------
DTI metrics are typically computed after these preprocessing steps:
1. Denoising (denoise)
2. Motion correction (motion_correction)
3. Susceptibility distortion correction (SDC, apply_SDC)
4. Bias field correction (bias_correction)
5. Brain extraction (bet)
6. DTI metric calculation (compute_fa_md) ← You are here

Exit Codes:
----------
0 : Success - FA and MD maps computed and saved
1 : Error - invalid inputs, file not found, dimension mismatch, or processing failure

See Also:
--------
- bias_correction : Recommended before DTI fitting
- denoise : For DWI denoising
- motion_correction : For motion/eddy current correction
- apply_SDC : For distortion correction
- bet : For brain mask generation

Technical Notes:
---------------
- DTI model requires at least 6 unique gradient directions + 1 b0 volume
- More directions improve estimation robustness (30+ recommended)
- Tensor fitting uses weighted least squares (WLS) by default in DIPY
- B-values should be in s/mm² units
- FA values outside [0,1] are clamped (indicates fitting issues)
- Processing time: ~30-60 seconds for typical resolution (depending on # volumes)

References:
----------

4. Garyfallidis E, Brett M, Amirbekian B, et al. Dipy, a library for the analysis of
   diffusion MRI data. Front Neuroinform. 2014;8:8. doi:10.3389/fninf.2014.00008
"""

import argparse
import sys
import os
import numpy as np
from dipy.reconst.dti import TensorModel
from dipy.core.gradients import gradient_table
import nibabel as nib
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
    """Print comprehensive help message with examples and interpretation guidelines."""
    
    help_text = f"""
    {CYAN}{BOLD}╔════════════════════════════════════════════════════════════════╗
    ║                DIFFUSION TENSOR METRICS (FA/MD)                ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script computes Fractional Anisotropy (FA) and Mean Diffusivity (MD)
    maps from diffusion-weighted images using the diffusion tensor model.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow compute_fa_md {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--input{RESET}      : Path to the input DWI image (.nii.gz)
                   {MAGENTA}Should be preprocessed (denoised, corrected, registered){RESET}
      {YELLOW}--mask{RESET}       : Path to the brain mask image (.nii.gz)
      {YELLOW}--bval{RESET}       : Path to the b-values file (.bval)
      {YELLOW}--bvec{RESET}       : Path to the b-vectors file (.bvec)
      {YELLOW}--output-fa{RESET}  : Output path for the FA map (.nii.gz)
      {YELLOW}--output-md{RESET}  : Output path for the MD map (.nii.gz)
    
    {CYAN}{BOLD}──────────────── OPTIONAL B0 MERGING ARGUMENTS ────────────────{RESET}
      {YELLOW}--b0-volume{RESET}  : Path to separate b0 volume to merge (.nii.gz)
      {YELLOW}--b0-bval{RESET}    : Path to b0 b-value file (.bval)
      {YELLOW}--b0-bvec{RESET}    : Path to b0 b-vector file (.bvec)
      {YELLOW}--b0-index{RESET}   : Index at which to insert b0 (default: 0)
                   {MAGENTA}Note: All three b0 arguments required if any is provided{RESET}
    
    {CYAN}{BOLD}──────────────────── EXAMPLE USAGE ──────────────────────{RESET}
    
    {BLUE}# Example 1: DWI already includes b0{RESET}
    micaflow compute_fa_md \\
      {YELLOW}--input{RESET} corrected_dwi.nii.gz \\
      {YELLOW}--mask{RESET} brain_mask.nii.gz \\
      {YELLOW}--bval{RESET} dwi.bval \\
      {YELLOW}--bvec{RESET} dwi.bvec \\
      {YELLOW}--output-fa{RESET} fa.nii.gz \\
      {YELLOW}--output-md{RESET} md.nii.gz
    
    {BLUE}# Example 2: Merge separate b0 volume{RESET}
    micaflow compute_fa_md \\
      {YELLOW}--input{RESET} dwi_no_b0.nii.gz \\
      {YELLOW}--mask{RESET} brain_mask.nii.gz \\
      {YELLOW}--bval{RESET} dwi_no_b0.bval \\
      {YELLOW}--bvec{RESET} dwi_no_b0.bvec \\
      {YELLOW}--b0-volume{RESET} b0.nii.gz \\
      {YELLOW}--b0-bval{RESET} b0.bval \\
      {YELLOW}--b0-bvec{RESET} b0.bvec \\
      {YELLOW}--b0-index{RESET} 0 \\
      {YELLOW}--output-fa{RESET} fa.nii.gz \\
      {YELLOW}--output-md{RESET} md.nii.gz
    
    {CYAN}{BOLD}──────────────── FA/MD INTERPRETATION ─────────────────{RESET}
    
    {GREEN}Fractional Anisotropy (FA){RESET} - Directionality of diffusion:
      {CYAN}0.7 - 0.8{RESET}: Major white matter tracts (corpus callosum)
      {CYAN}0.5 - 0.7{RESET}: Other white matter (internal capsule, corona radiata)
      {YELLOW}0.3 - 0.5{RESET}: Peripheral white matter, crossing fibers
      {MAGENTA}0.1 - 0.3{RESET}: Gray matter
      {BLUE}0.0 - 0.1{RESET}: CSF, free water
    
    {GREEN}Mean Diffusivity (MD){RESET} - Overall diffusion magnitude (×10⁻³ mm²/s):
      {BLUE}2.5 - 3.5{RESET}: CSF (free water diffusion)
      {MAGENTA}0.7 - 1.0{RESET}: Gray matter
      {CYAN}0.6 - 0.9{RESET}: White matter
      {RED}↑ MD{RESET}: Edema, necrosis, inflammation
      {YELLOW}↓ MD{RESET}: Increased cellularity, cytotoxic edema
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    {MAGENTA}•{RESET} FA values range from 0 (isotropic) to 1 (perfectly anisotropic)
    {MAGENTA}•{RESET} MD is reported in mm²/s (typical brain values: 0.6-3.5 × 10⁻³)
    {MAGENTA}•{RESET} Processing requires a brain mask to exclude non-brain regions
    {MAGENTA}•{RESET} DTI fitting requires at least 6 gradient directions + 1 b0 volume
    {MAGENTA}•{RESET} More directions improve robustness (30+ recommended)
    {MAGENTA}•{RESET} If b0 volume is provided, it will be merged with DWI at specified index
    {MAGENTA}•{RESET} B0 gradients (bval/bvec) will be inserted at the same index
    {MAGENTA}•{RESET} Uses weighted least squares (WLS) tensor fitting from DIPY
    {MAGENTA}•{RESET} Typical processing time: 30-60 seconds (varies with # volumes)
    {MAGENTA}•{RESET} FA values are automatically clamped to [0, 1] range
    
    {CYAN}{BOLD}────────────────────── EXIT CODES ───────────────────────{RESET}
    {GREEN}0{RESET} : Success - FA and MD maps computed and saved
    {RED}1{RESET} : Error - invalid inputs, file not found, or processing failure
    
    {CYAN}{BOLD}───────────────── COMMON ISSUES ─────────────────────────{RESET}
    {YELLOW}Issue:{RESET} "Shape mismatch" error
    {GREEN}Solution:{RESET} Ensure DWI, mask, and b0 (if provided) have matching dimensions
    
    {YELLOW}Issue:{RESET} Very low FA values throughout
    {GREEN}Solution:{RESET} Check b-vector normalization and coordinate system
    
    {YELLOW}Issue:{RESET} FA values > 1 or < 0
    {GREEN}Solution:{RESET} Indicates poor tensor fit - check gradient table and SNR
    """
    print(help_text)


def merge_b0_with_dwi(dwi_data, dwi_affine, b0_data, bvals, bvecs, b0_bval, b0_bvec, b0_index=0):
    """
    Merge b0 volume and gradients with DWI data.
    
    This function inserts a b0 volume and its corresponding b-value/b-vector into
    existing DWI data at a specified index position. This is useful when b0 volumes
    are acquired or processed separately from the diffusion-weighted volumes.
    
    Parameters
    ----------
    dwi_data : numpy.ndarray
        4D DWI data array with shape (X, Y, Z, volumes).
    dwi_affine : numpy.ndarray
        4x4 affine transformation matrix from the DWI image.
    b0_data : numpy.ndarray
        3D or 4D b0 volume data. If 4D, only the first volume is used.
    bvals : numpy.ndarray or list
        Array of b-values for DWI volumes (in s/mm²).
    bvecs : numpy.ndarray or list of lists
        Array of b-vectors for DWI (shape: 3 × N or list of 3 lists).
    b0_bval : float
        B-value for the b0 volume (typically 0 or < 50 s/mm²).
    b0_bvec : list or numpy.ndarray
        B-vector for b0 (typically [0, 0, 0] or close to it).
        Should have length 3.
    b0_index : int, optional
        Index at which to insert b0 volume. Default is 0 (beginning).
    
    Returns
    -------
    merged_data : numpy.ndarray
        Merged 4D data array with b0 inserted at specified position.
    merged_bvals : numpy.ndarray
        Merged b-values array.
    merged_bvecs : numpy.ndarray
        Merged b-vectors array (shape: 3 × N).
        
    Examples
    --------
    >>> dwi = np.random.rand(128, 128, 60, 30)  # 30 DWI volumes
    >>> b0 = np.random.rand(128, 128, 60)  # Single b0 volume
    >>> bvals = np.array([1000] * 30)
    >>> bvecs = np.random.rand(3, 30)
    >>> merged, merged_bvals, merged_bvecs = merge_b0_with_dwi(
    ...     dwi, np.eye(4), b0, bvals, bvecs, 0, [0, 0, 0], b0_index=0
    ... )
    >>> merged.shape
    (128, 128, 60, 31)
    >>> len(merged_bvals)
    31
    """
    # Ensure b0 is 3D
    if len(b0_data.shape) == 4:
        print(f"{YELLOW}Warning: b0 volume is 4D, using only first volume{RESET}")
        b0_data = b0_data[..., 0]
    
    # Add volume dimension to b0
    b0_data = b0_data[..., np.newaxis]
    
    # Insert b0 into DWI data at specified index
    merged_data = np.concatenate([
        dwi_data[..., :b0_index],
        b0_data,
        dwi_data[..., b0_index:]
    ], axis=-1)
    
    # Insert b0_bval into bvals
    merged_bvals = np.insert(bvals, b0_index, b0_bval)
    
    # Insert b0_bvec into bvecs (each row separately)
    merged_bvecs = []
    for row_idx in range(3):
        merged_row = np.insert(bvecs[row_idx], b0_index, b0_bvec[row_idx])
        merged_bvecs.append(merged_row)
    merged_bvecs = np.array(merged_bvecs)
    
    return merged_data, merged_bvals, merged_bvecs


def compute_fa_md(bias_corr_path, mask_path, moving_bval, moving_bvec, fa_path, md_path, 
                  b0_volume=None, b0_bval=None, b0_bvec=None, b0_index=0):
    """
    Compute Fractional Anisotropy (FA) and Mean Diffusivity (MD) maps from DWI.
    
    This function takes a preprocessed diffusion-weighted image and computes DTI
    scalar metrics using the diffusion tensor model from DIPY. The calculation
    is restricted to brain tissue using the provided mask.
    
    Parameters
    ----------
    bias_corr_path : str
        Path to the bias-corrected DWI image (.nii.gz).
        Should be fully preprocessed (denoised, motion-corrected, distortion-corrected).
    mask_path : str
        Path to the brain mask image (.nii.gz).
        Binary mask where 1 = brain tissue, 0 = background.
    moving_bval : str
        Path to the b-values file (.bval).
        Text file with space-separated b-values in s/mm².
    moving_bvec : str
        Path to the b-vectors file (.bvec).
        Text file with 3 rows (x, y, z) of space-separated gradient directions.
    fa_path : str
        Output path for the fractional anisotropy (FA) map (.nii.gz).
    md_path : str
        Output path for the mean diffusivity (MD) map (.nii.gz).
        Units: mm²/s
    b0_volume : str, optional
        Path to separate b0 volume to merge with DWI (.nii.gz).
    b0_bval : str, optional
        Path to b0 b-value file (.bval).
    b0_bvec : str, optional
        Path to b0 b-vector file (.bvec).
    b0_index : int, optional
        Index at which to insert b0 volume. Default is 0 (beginning).
        
    Returns
    -------
    tuple of str
        (fa_path, md_path) - Paths to the saved FA and MD NIfTI files.
        
    Raises
    ------
    FileNotFoundError
        If any input file cannot be found.
    ValueError
        If dimensions don't match or gradient table is invalid.
        
    Notes
    -----
    - MD values are in mm²/s (not × 10⁻³ mm²/s)
    - FA values are automatically clamped to [0, 1]
    - Uses weighted least squares (WLS) tensor fitting
    - Requires at least 6 unique gradient directions + 1 b0 volume
    - B-vectors should be normalized for non-zero b-values
    
    Examples
    --------
    >>> # Standard usage
    >>> fa, md = compute_fa_md(
    ...     "corrected_dwi.nii.gz",
    ...     "brain_mask.nii.gz",
    ...     "dwi.bval",
    ...     "dwi.bvec",
    ...     "fa.nii.gz",
    ...     "md.nii.gz"
    ... )
    >>> 
    >>> # With b0 merging
    >>> fa, md = compute_fa_md(
    ...     "dwi_no_b0.nii.gz",
    ...     "brain_mask.nii.gz",
    ...     "dwi.bval",
    ...     "dwi.bvec",
    ...     "fa.nii.gz",
    ...     "md.nii.gz",
    ...     b0_volume="b0.nii.gz",
    ...     b0_bval="b0.bval",
    ...     b0_bvec="b0.bvec"
    ... )
    """
    print(f"{CYAN}Loading DWI data...{RESET}")
    bias_corr = nib.load(bias_corr_path)
    dwi_data = bias_corr.get_fdata()
    dwi_affine = bias_corr.affine
    print(f"  DWI shape: {dwi_data.shape}")
    
    print(f"{CYAN}Loading brain mask...{RESET}")
    mask = nib.load(mask_path)
    mask_data = mask.get_fdata()
    print(f"  Mask shape: {mask_data.shape}")
    
    # Validate dimensions
    if dwi_data.shape[:3] != mask_data.shape[:3]:
        raise ValueError(
            f"Dimension mismatch:\n"
            f"  DWI spatial dimensions: {dwi_data.shape[:3]}\n"
            f"  Mask dimensions: {mask_data.shape[:3]}\n"
            f"Spatial dimensions must match."
        )
    
    print(f"{CYAN}Loading gradient table...{RESET}")
    # Load bvals and bvecs
    with open(moving_bval, 'r') as f:
        bvals = np.array([float(val) for val in f.read().strip().split()])
    
    with open(moving_bvec, 'r') as f:
        bvec_lines = f.readlines()
        bvecs = []
        for line in bvec_lines:
            bvecs.append([float(val) for val in line.strip().split()])
        bvecs = np.array(bvecs)
    
    print(f"  B-values: {len(bvals)} volumes")
    print(f"  B-vectors shape: {bvecs.shape}")
    
    # If b0 volume is provided, merge it with DWI data
    if b0_volume and b0_bval and b0_bvec:
        print(f"{CYAN}Merging b0 volume with DWI data...{RESET}")
        
        # Load b0 volume
        b0_img = nib.load(b0_volume)
        b0_data = b0_img.get_fdata()
        print(f"  B0 shape: {b0_data.shape}")
        
        # Load b0 bval
        with open(b0_bval, 'r') as f:
            b0_bval_value = float(f.read().strip())
        
        # Load b0 bvec
        with open(b0_bvec, 'r') as f:
            b0_bvec_lines = f.readlines()
            b0_bvec_values = [float(line.strip()) for line in b0_bvec_lines]
        
        # Merge b0 with DWI
        dwi_data, bvals, bvecs = merge_b0_with_dwi(
            dwi_data, dwi_affine, b0_data, 
            bvals, bvecs, 
            b0_bval_value, b0_bvec_values, 
            b0_index
        )
        
        print(f"{GREEN}Merged b0 at index {b0_index}{RESET}")
        print(f"  New shape: {dwi_data.shape}")
        print(f"  Total volumes: {len(bvals)}")
    
    # Filter out invalid diffusion volumes (b > 50 but bvec is 0,0,0)
    # bvecs is shaped (3, N) at this point
    bvec_norms = np.linalg.norm(bvecs, axis=0)
    # Identify indices that are in a shell (b>50) but have no direction
    bad_indices = (bvals > 50) & (bvec_norms < 1e-6)
    
    if np.any(bad_indices):
        n_removed = np.sum(bad_indices)
        print(f"{YELLOW}Warning: Found {n_removed} volumes in diffusion shell (b>50) with zero b-vectors (0,0,0).{RESET}")
        print(f"{YELLOW}  Excluding invalid volumes to prevent fitting errors...{RESET}")
        
        # Create mask of volumes to keep
        keep_mask = ~bad_indices
        
        # Apply filtering
        dwi_data = dwi_data[..., keep_mask]
        bvals = bvals[keep_mask]
        bvecs = bvecs[:, keep_mask]
        
        print(f"  New DWI shape after filtering: {dwi_data.shape}")
        print(f"  New B-values count: {len(bvals)}")

    # Count b0 volumes
    b0_count = np.sum(bvals <= 50)
    dwi_count = len(bvals) - b0_count
    print(f"{CYAN}Gradient table summary:{RESET}")
    print(f"  B0 volumes (b ≤ 50): {b0_count}")
    print(f"  DWI volumes (b > 50): {dwi_count}")
    
    if b0_count == 0:
        raise ValueError(
            "No b0 volumes found (b-value ≤ 50). "
            "DTI fitting requires at least one b0 volume."
        )
    
    if dwi_count < 6:
        print(f"{YELLOW}Warning: Only {dwi_count} DWI directions found. "
              f"DTI fitting requires at least 6 for reliable results.{RESET}")
    
    # Apply mask (broadcast to 4D)
    print(f"{CYAN}Applying brain mask...{RESET}")
    masked_data = dwi_data * mask_data[..., None]
    
    # Create gradient table
    print(f"{CYAN}Creating gradient table...{RESET}")
    # DIPY expects bvecs as (N, 3) not (3, N)
    bvecs_transposed = bvecs.T
    gtab = gradient_table(bvals, bvecs_transposed)
    
    # Fit tensor model
    print(f"{CYAN}Fitting diffusion tensor model...{RESET}")
    print(f"  Using weighted least squares (WLS) fitting")
    tensor_model = TensorModel(gtab)
    tensor_fit = tensor_model.fit(masked_data)
    
    # Compute FA and MD
    print(f"{CYAN}Computing FA and MD maps...{RESET}")
    fa = tensor_fit.fa
    md = tensor_fit.md
    
    # Report statistics
    print(f"\n{GREEN}DTI Metrics Summary:{RESET}")
    print(f"  FA range: [{fa.min():.4f}, {fa.max():.4f}]")
    print(f"  FA mean (in mask): {fa[mask_data > 0].mean():.4f}")
    print(f"  MD range: [{md.min():.6f}, {md.max():.6f}] mm²/s")
    print(f"  MD mean (in mask): {md[mask_data > 0].mean():.6f} mm²/s")
    
    # Save FA and MD maps
    print(f"\n{CYAN}Saving output maps...{RESET}")
    nib.save(nib.Nifti1Image(fa, dwi_affine), fa_path)
    nib.save(nib.Nifti1Image(md, dwi_affine), md_path)
    
    return fa_path, md_path


if __name__ == "__main__":
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="Compute FA and MD maps using bias-corrected DWI and a brain mask.",
        add_help=False  # Use custom help
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the bias-corrected DWI image (NIfTI file).")
    parser.add_argument("--mask", type=str, required=True,
                        help="Path to the brain mask image (NIfTI file).")
    parser.add_argument("--bval", type=str, required=True,
                        help="Path to the bvals file.")
    parser.add_argument("--bvec", type=str, required=True,
                        help="Path to the bvecs file.")
    parser.add_argument("--output-fa", type=str, required=True,
                        help="Output path for the FA map.")
    parser.add_argument("--output-md", type=str, required=True,
                        help="Output path for the MD map.")
    parser.add_argument("--b0-volume", type=str,
                        help="Path to the b0 volume to merge with DWI.")
    parser.add_argument("--b0-bval", type=str,
                        help="Path to b0 b-value file.")
    parser.add_argument("--b0-bvec", type=str,
                        help="Path to b0 b-vector file.")
    parser.add_argument("--b0-index", type=int, default=0,
                        help="Index at which to insert b0 volume (default: 0).")
    
    args = parser.parse_args()
    
    # Validate b0 arguments
    b0_args = [args.b0_volume, args.b0_bval, args.b0_bvec]
    if any(b0_args) and not all(b0_args):
        print(f"{RED}Error: If any b0 argument is provided, all three "
              f"(--b0-volume, --b0-bval, --b0-bvec) must be provided.{RESET}")
        sys.exit(1)
    
    try:
        fa_path, md_path = compute_fa_md(
            args.input, args.mask, args.bval, args.bvec, 
            args.output_fa, args.output_md, 
            args.b0_volume, args.b0_bval, args.b0_bvec, args.b0_index
        )
        
        print(f"\n{GREEN}{BOLD}DTI metrics computation complete!{RESET}")
        print(f"  FA map: {fa_path}")
        print(f"  MD map: {md_path}")
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
        print(f"\n{RED}{BOLD}✗ Error during DTI computation:{RESET}")
        print(f"  {str(e)}")
        print(f"\n{YELLOW}Run 'micaflow compute_fa_md --help' for usage information.{RESET}")
        sys.exit(1)