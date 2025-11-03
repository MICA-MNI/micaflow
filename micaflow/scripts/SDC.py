"""
SDC - Susceptibility Distortion Correction for Echo-Planar Imaging

Part of the micaflow processing pipeline for neuroimaging data.

This module corrects geometric distortions in echo-planar imaging (EPI) MRI data 
caused by magnetic field (B0) inhomogeneities. EPI sequences, commonly used for 
functional MRI and diffusion MRI, are highly susceptible to distortions from field 
inhomogeneities that cause compression, stretching, and signal loss in brain regions 
near air-tissue interfaces.

What are Susceptibility Distortions?
------------------------------------
Susceptibility artifacts in EPI occur due to:
- Variations in magnetic susceptibility at tissue boundaries
- Air-tissue interfaces (e.g., frontal sinuses, ear canals)
- Bone-tissue interfaces
- Rapid gradient switching in EPI readout

Effects of distortions:
- Geometric warping (stretching/compression) along phase-encoding direction
- Signal dropout in affected regions (orbitofrontal cortex, temporal poles)
- Misalignment with structural images
- Incorrect spatial localization of brain activity/structure
- Poor registration quality

Common affected regions:
- Orbitofrontal cortex (near frontal sinuses)
- Inferior temporal lobes (near ear canals)
- Brainstem (near skull base)
- Any region near air-tissue interface

HYSCO Algorithm:
---------------
This implementation uses HYSCO (HYperellastic Susceptibility artifact COrrection):
- Estimates B0 field inhomogeneity from blip-up/blip-down image pairs
- Uses opposite phase-encoding acquisitions (e.g., AP and PA)
- Employs hyperelastic regularization for smooth field estimates
- Optimizes using ADMM (Alternating Direction Method of Multipliers)
- Produces both corrected images and displacement fields

How It Works:
1. Acquire two images with opposite phase-encoding directions
2. Register images using ANTs affine transformation
3. Estimate B0 field map using HYSCO optimization
4. Apply displacement field to unwarp distorted images
5. Output corrected image and field map

Features:
--------
- HYSCO-based B0 field estimation with hyperelastic regularization
- GPU acceleration with PyTorch (CUDA support when available)
- Automatic initial alignment using ANTs affine registration
- ADMM optimization for robust field estimation
- Handles both 3D and 4D (multi-volume) datasets
- Outputs corrected images and displacement fields
- Supports all common phase-encoding directions (AP/PA, LR/RL, SI/IS)
- Automatic dimension handling and warp application
- Temporary file management for clean processing

When to Use:
-----------
- Always for EPI-based sequences (fMRI, DWI)
- Before registration to structural images
- After motion correction (for DWI)
- Before group-level analysis

Requirements:
------------
- Two EPI images with opposite phase-encoding directions
- Images should be from the same session/subject
- For DWI: typically use b=0 volumes from AP and PA acquisitions
- GPU recommended for faster processing (10-30x speedup)

Command-Line Usage:
------------------
# Basic AP/PA correction (default)
micaflow SDC \\
    --input <path/to/ap_image.nii.gz> \\
    --reverse-image <path/to/pa_image.nii.gz> \\
    --output <path/to/corrected.nii.gz> \\
    --output-warp <path/to/fieldmap.nii.gz>

# LR/RL phase-encoding
micaflow SDC \\
    --input <path/to/lr_image.nii.gz> \\
    --reverse-image <path/to/rl_image.nii.gz> \\
    --output <path/to/corrected.nii.gz> \\
    --output-warp <path/to/fieldmap.nii.gz> \\
    --phase-encoding lr

# SI/IS phase-encoding
micaflow SDC \\
    --input <path/to/si_image.nii.gz> \\
    --reverse-image <path/to/is_image.nii.gz> \\
    --output <path/to/corrected.nii.gz> \\
    --output-warp <path/to/fieldmap.nii.gz> \\
    --phase-encoding si

Python API Usage:
----------------
>>> from micaflow.scripts.SDC import run
>>> 
>>> # Basic AP/PA correction
>>> run(
...     data_image="ap_b0.nii.gz",
...     reverse_image="pa_b0.nii.gz",
...     output_name="corrected_b0.nii.gz",
...     output_warp="fieldmap.nii.gz"
... )
>>> 
>>> # With specific phase-encoding
>>> run(
...     data_image="lr_epi.nii.gz",
...     reverse_image="rl_epi.nii.gz",
...     output_name="corrected_epi.nii.gz",
...     output_warp="fieldmap.nii.gz",
...     phase_encoding="lr"
... )

Pipeline Integration:
--------------------
Susceptibility distortion correction is typically applied AFTER motion correction
but BEFORE registration to structural images:

Diffusion MRI Pipeline:
1. Denoising (denoise)
2. Motion/eddy correction (motion_correction)
3. Susceptibility correction (SDC)
4. B0 extraction (extract_b0)
5. Registration to T1w (coregister)
6. DTI metrics (compute_fa_md)

Functional MRI Pipeline:
1. Motion correction
2. Susceptibility correction (SDC)
3. Registration to anatomical
4. Smoothing
5. Statistical analysis

Exit Codes:
----------
0 : Success - distortion correction completed
1 : Error - invalid inputs, file not found, or processing failure

Technical Notes:
---------------
- Algorithm: HYSCO (HYperellastic Susceptibility artifact COrrection)
- Optimization: ADMM (Alternating Direction Method of Multipliers)
- Regularization: Hyperelastic regularization with Laplacian operator
- ADMM parameters:
  * max_iter: 500 iterations
  * rho_max: 1e6 (maximum penalty parameter)
  * rho_min: 1e1 (minimum penalty parameter)
  * max_iter_pcg: 20 (preconditioned conjugate gradient iterations)
- Initial alignment: ANTs affine registration
- Interpolation: Linear (order=1) with nearest-neighbor extrapolation
- Phase-encoding dimension mapping:
  * AP/PA → y-axis (dimension 2)
  * LR/RL → x-axis (dimension 1)  
  * SI/IS → z-axis (dimension 3)
- Supports 4D data: applies correction to all volumes
- Field map saved for applying to other images

Phase-Encoding Directions:
-------------------------
- AP (Anterior-Posterior): Front to back, most common
- PA (Posterior-Anterior): Back to front
- LR (Left-Right): Left to right
- RL (Right-Left): Right to left
- SI (Superior-Inferior): Top to bottom
- IS (Inferior-Superior): Bottom to top

Acquisition Requirements:
------------------------
To use this correction, you need:
1. Two EPI images with OPPOSITE phase-encoding directions
2. Same subject, same session
3. Same acquisition parameters (except PE direction)
4. Minimal head motion between acquisitions

Common acquisition protocols:
- DWI: b=0 volumes with AP and PA encoding
- fMRI: Short EPI runs with opposite PE directions
- Separate "blip-up/blip-down" reference scans

See Also:
--------
- motion_correction : Apply before SDC for DWI
- extract_b0 : Extract b=0 volumes for SDC input
- coregister : Register corrected images to structural
- apply_warp : Apply field map to other volumes

References:
----------

Julian, Abigail, and Lars Ruthotto. "PyHySCO: GPU-enabled susceptibility artifact distortion correction in seconds." Frontiers in Neuroscience 18 (2024): 1406821.

"""

import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
from EPI_MRI.EPIMRIDistortionCorrection import DataObject, EPIMRIDistortionCorrection
from optimization.ADMM import myAvg1D, myDiff1D, myLaplacian1D, JacobiCG, ADMM
import torch
import ants
import argparse
import tempfile
import os
import shutil
import sys
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
    ║            SUSCEPTIBILITY DISTORTION CORRECTION                ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script corrects geometric distortions in echo-planar imaging (EPI) 
    MR images caused by magnetic field (B0) inhomogeneities. It uses the 
    HYSCO algorithm with a pair of images acquired with opposite phase-encoding 
    directions to estimate and correct distortions.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow SDC {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--input{RESET}         : Path to the main EPI image (.nii.gz)
      {YELLOW}--reverse-image{RESET} : Path to the reverse phase-encoded image (.nii.gz)
      {YELLOW}--output{RESET}        : Output path for the corrected image (.nii.gz)
      {YELLOW}--output-warp{RESET}   : Output path for the field map (.nii.gz)
    
    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--phase-encoding{RESET}: Phase-encoding direction
                         {MAGENTA}Choices: 'ap', 'pa', 'lr', 'rl', 'si', 'is'{RESET}
                         {MAGENTA}Default: 'ap' (Anterior-Posterior){RESET}
    
    {CYAN}{BOLD}──────────────────── EXAMPLE USAGE ───────────────────────{RESET}
    
    {BLUE}# Example 1: AP/PA correction (most common for DWI){RESET}
    micaflow SDC \\
      {YELLOW}--input{RESET} ap_b0.nii.gz \\
      {YELLOW}--reverse-image{RESET} pa_b0.nii.gz \\
      {YELLOW}--output{RESET} corrected_b0.nii.gz \\
      {YELLOW}--output-warp{RESET} fieldmap.nii.gz
    
    {BLUE}# Example 2: LR/RL phase-encoding{RESET}
    micaflow SDC \\
      {YELLOW}--input{RESET} lr_epi.nii.gz \\
      {YELLOW}--reverse-image{RESET} rl_epi.nii.gz \\
      {YELLOW}--output{RESET} corrected_epi.nii.gz \\
      {YELLOW}--output-warp{RESET} fieldmap.nii.gz \\
      {YELLOW}--phase-encoding{RESET} lr
    
    {BLUE}# Example 3: SI/IS phase-encoding{RESET}
    micaflow SDC \\
      {YELLOW}--input{RESET} si_fmri.nii.gz \\
      {YELLOW}--reverse-image{RESET} is_fmri.nii.gz \\
      {YELLOW}--output{RESET} corrected_fmri.nii.gz \\
      {YELLOW}--output-warp{RESET} fieldmap.nii.gz \\
      {YELLOW}--phase-encoding{RESET} si
    
    {CYAN}{BOLD}─────── WHAT ARE SUSCEPTIBILITY DISTORTIONS? ────────────{RESET}
    
    {GREEN}Definition:{RESET}
    Geometric distortions in EPI images caused by B0 field inhomogeneities
    at tissue boundaries, especially near air-tissue interfaces.
    
    {GREEN}Causes:{RESET}
    {MAGENTA}•{RESET} Magnetic susceptibility differences at boundaries
    {MAGENTA}•{RESET} Air-tissue interfaces (sinuses, ear canals)
    {MAGENTA}•{RESET} Bone-tissue interfaces
    {MAGENTA}•{RESET} Rapid EPI readout susceptibility
    
    {GREEN}Effects:{RESET}
    {MAGENTA}•{RESET} Stretching/compression along phase-encode direction
    {MAGENTA}•{RESET} Signal dropout in affected regions
    {MAGENTA}•{RESET} Misalignment with structural images
    {MAGENTA}•{RESET} Poor registration quality
    
    {GREEN}Commonly Affected Regions:{RESET}
    {MAGENTA}•{RESET} Orbitofrontal cortex (frontal sinuses)
    {MAGENTA}•{RESET} Inferior temporal lobes (ear canals)
    {MAGENTA}•{RESET} Brainstem (skull base)
    
    {CYAN}{BOLD}────────────────── HYSCO ALGORITHM ──────────────────────{RESET}
    
    {GREEN}Method:{RESET}
    {MAGENTA}1.{RESET} Acquire images with opposite phase-encoding (blip-up/down)
    {MAGENTA}2.{RESET} Initial alignment using ANTs affine registration
    {MAGENTA}3.{RESET} Estimate B0 field map using ADMM optimization
    {MAGENTA}4.{RESET} Apply hyperelastic regularization for smoothness
    {MAGENTA}5.{RESET} Generate displacement field and correct images
    
    {GREEN}Optimization:{RESET}
    {MAGENTA}•{RESET} ADMM (Alternating Direction Method of Multipliers)
    {MAGENTA}•{RESET} Hyperelastic regularization
    {MAGENTA}•{RESET} 500 iterations with adaptive penalty parameter
    {MAGENTA}•{RESET} Preconditioned conjugate gradient solver
    
    {CYAN}{BOLD}───────────── PHASE-ENCODING DIRECTIONS ─────────────────{RESET}
    {GREEN}AP{RESET} (Anterior-Posterior): Front → Back {MAGENTA}[Most common for DWI]{RESET}
    {GREEN}PA{RESET} (Posterior-Anterior): Back → Front
    {GREEN}LR{RESET} (Left-Right):         Left → Right
    {GREEN}RL{RESET} (Right-Left):         Right → Left
    {GREEN}SI{RESET} (Superior-Inferior):  Top → Bottom
    {GREEN}IS{RESET} (Inferior-Superior):  Bottom → Top
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    {MAGENTA}•{RESET} Requires TWO images with OPPOSITE phase-encoding directions
    {MAGENTA}•{RESET} Images must be from same subject, same session
    {MAGENTA}•{RESET} For DWI: typically use b=0 volumes (one AP, one PA)
    {MAGENTA}•{RESET} Field map can be applied to other volumes
    {MAGENTA}•{RESET} Supports both 3D and 4D (multi-volume) inputs
    {MAGENTA}•{RESET} Always apply AFTER motion correction (for DWI)
    {MAGENTA}•{RESET} Apply BEFORE registration to structural images
    
    {CYAN}{BOLD}──────────────── ACQUISITION REQUIREMENTS ───────────────{RESET}
    {GREEN}Essential:{RESET}
    {MAGENTA}•{RESET} Two EPI acquisitions with opposite PE directions
    {MAGENTA}•{RESET} Same subject, same session
    {MAGENTA}•{RESET} Same acquisition parameters (except PE direction)
    {MAGENTA}•{RESET} Minimal head motion between acquisitions
    
    {GREEN}Common Protocols:{RESET}
    {MAGENTA}•{RESET} DWI: b=0 volumes with AP and PA
    {MAGENTA}•{RESET} fMRI: Short runs with opposite PE
    {MAGENTA}•{RESET} Separate "fieldmap" scans
    
    {CYAN}{BOLD}─────────────── PIPELINE POSITION ───────────────────────{RESET}
    {BLUE}Diffusion Pipeline:{RESET}
    1. Denoising (denoise)
    2. Motion correction (motion_correction)
    3. Susceptibility correction (SDC)
    4. B0 extraction (extract_b0)
    5. Registration (coregister)
    
    {BLUE}fMRI Pipeline:{RESET}
    1. Motion correction
    2. Susceptibility correction (SDC)
    3. Registration to anatomical
    4. Analysis
    
    {CYAN}{BOLD}────────────────────── EXIT CODES ───────────────────────{RESET}
    {GREEN}0{RESET} : Success - distortion correction completed
    {RED}1{RESET} : Error - invalid inputs, file not found, or processing failure
    
    {CYAN}{BOLD}───────────────── COMMON ISSUES ─────────────────────────{RESET}
    {YELLOW}Issue:{RESET} "Out of memory" error
    {GREEN}Solution:{RESET} Use GPU if available, or reduce image resolution
    
    {YELLOW}Issue:{RESET} Poor correction quality
    {GREEN}Solution:{RESET} Check PE directions match acquisition, ensure minimal motion
    
    {YELLOW}Issue:{RESET} Very slow processing
    {GREEN}Solution:{RESET} Use GPU acceleration (10-30x faster)
    
    {YELLOW}Issue:{RESET} Field map dimensions don't match
    {GREEN}Solution:{RESET} Script auto-crops; check input image dimensions
    """
    print(help_text)


def get_pe_dimension(phase_encoding):
    """
    Get the dimension index for the phase-encoding direction.
    
    Maps phase-encoding direction codes to the corresponding spatial dimension
    index used for applying geometric corrections.
    
    Parameters
    ----------
    phase_encoding : str
        Phase-encoding direction code:
        - 'ap' or 'pa': Anterior-Posterior (y-axis)
        - 'lr' or 'rl': Left-Right (x-axis)
        - 'si' or 'is': Superior-Inferior (z-axis)
        
    Returns
    -------
    int
        The dimension index (1=x, 2=y, 3=z) for the phase-encoding direction.
        Returns 2 (y-axis) as default if direction is unrecognized.
        
    Notes
    -----
    - Most DWI acquisitions use AP/PA (y-axis, dimension 2)
    - Dimension indexing: 1=x, 2=y, 3=z (HYSCO convention)
    - Case-insensitive: 'AP', 'ap', 'Ap' all work
    
    Examples
    --------
    >>> pe_dim = get_pe_dimension('ap')
    >>> print(pe_dim)
    2
    >>> 
    >>> pe_dim = get_pe_dimension('lr')
    >>> print(pe_dim)
    1
    """
    # Convert to lowercase and normalize direction
    pe = phase_encoding.lower()
    
    # Set dimensions based on phase-encoding direction
    if pe in ['lr', 'rl']:  # Left-Right or Right-Left (x-axis)
        return 1
    elif pe in ['ap', 'pa']:  # Anterior-Posterior or Posterior-Anterior (y-axis)
        return 2
    elif pe in ['si', 'is']:  # Superior-Inferior or Inferior-Superior (z-axis)
        return 3
    else:
        print(f"{YELLOW}Warning: Unknown phase-encoding '{phase_encoding}', using default (y-axis){RESET}")
        return 2  # Default to y-axis (AP/PA)

    
def apply_warpfield(image, warpfield, pe_dim=1):
    """
    Apply a displacement field to warp an image along the phase-encoding direction.
    
    This function corrects geometric distortions by applying a displacement field
    that specifies how each voxel should be shifted along the phase-encoding axis.
    Uses linear interpolation to map intensities from the distorted space to the
    corrected space.
    
    Parameters
    ----------
    image : numpy.ndarray
        The input 3D image to be warped. Shape: (nx, ny, nz).
    warpfield : numpy.ndarray
        The displacement field specifying voxel shifts along the phase-encoding
        direction. Same shape as image. Values represent displacement in voxels.
    pe_dim : int, optional
        The dimension along which to apply the warp (0=x, 1=y, 2=z).
        Default is 1 (y-axis, typical for AP/PA encoding).
        
    Returns
    -------
    warped_image : numpy.ndarray
        The corrected image after applying the displacements.
        Same shape as input image.
        
    Notes
    -----
    - Uses linear interpolation (order=1) for smooth results
    - Nearest-neighbor mode for extrapolation at boundaries
    - Displacement is applied by modifying coordinates along PE dimension
    - Positive displacement = shift in positive direction along axis
    - Works on 3D images only; for 4D, apply volume-by-volume
    
    Examples
    --------
    >>> # Load distorted image and field map
    >>> image = nib.load("distorted.nii.gz").get_fdata()
    >>> fieldmap = nib.load("fieldmap.nii.gz").get_fdata()
    >>> 
    >>> # Apply correction along y-axis (AP/PA)
    >>> corrected = apply_warpfield(image, fieldmap, pe_dim=1)
    >>> 
    >>> # Save result
    >>> nib.save(nib.Nifti1Image(corrected, affine), "corrected.nii.gz")
    """
    # Create a grid of coordinates
    coords = np.meshgrid(
        np.arange(image.shape[0]),
        np.arange(image.shape[1]),
        np.arange(image.shape[2]),
        indexing="ij",
    )

    # Convert tuple to list so we can modify it
    warped_coords = list(coords)
    
    # Apply the warpfield to the coordinates along the specified dimension
    warped_coords[pe_dim] = coords[pe_dim] + warpfield

    # Interpolate the image at the warped coordinates
    warped_image = map_coordinates(
        image, warped_coords, order=1, mode="nearest"
    )

    return warped_image


def run(data_image, reverse_image, output_name, output_warp, phase_encoding='ap', shell_channel=0):
    """
    Perform EPI distortion correction using phase-encoding reversed images.
    
    This function implements the HYSCO (HYperellastic Susceptibility artifact 
    COrrection) algorithm for correcting geometric distortions in echo-planar 
    imaging (EPI) MRI data. It uses a pair of images acquired with opposite 
    phase-encoding directions to estimate and correct susceptibility-induced 
    distortions via ADMM optimization.
    
    Parameters
    ----------
    data_image : str
        Path to the main EPI image (NIfTI file).
        Can be 3D or 4D. For DWI, typically a b=0 volume.
    reverse_image : str
        Path to the reverse phase-encoded EPI image (NIfTI file).
        Must be acquired with opposite PE direction from data_image.
        Should be same subject, session, and have minimal motion.
    output_name : str
        Path where the distortion-corrected image will be saved.
        Output will have same dimensions as data_image.
    output_warp : str
        Path where the estimated displacement field will be saved.
        Field map shows voxel displacements along PE direction.
    phase_encoding : str, optional
        Phase-encoding direction of data_image:
        - 'ap' or 'pa': Anterior-Posterior (y-axis) [default]
        - 'lr' or 'rl': Left-Right (x-axis)
        - 'si' or 'is': Superior-Inferior (z-axis)
    shell_channel : int, optional
        For 4D data, index of volume to use for field estimation.
        Default: 0 (first volume, typically b=0 for DWI).
        
    Returns
    -------
    None
        Saves corrected image to output_name and field map to output_warp.
        
    Raises
    ------
    FileNotFoundError
        If input files cannot be found.
    ValueError
        If image dimensions are incompatible.
        
    Notes
    -----
    - HYSCO algorithm: Hyperelastic regularized field estimation
    - ADMM optimization: 500 iterations with adaptive penalty
    - GPU acceleration: Automatic if CUDA available (10-30x faster)
    - Initial alignment: ANTs affine registration before field estimation
    - Processing time: 1-5 min (GPU) or 5-30 min (CPU)
    - Memory: ~8-16 GB RAM for typical datasets
    - Field map is in units of voxel displacement
    - For 4D inputs, applies correction to all volumes
    - Temporary files cleaned up automatically
    
    Algorithm Steps:
    1. Load input images and extract specified volume (if 4D)
    2. Register reverse image to data image (ANTs affine)
    3. Initialize HYSCO optimization with registered pair
    4. Estimate B0 field map using ADMM (500 iterations)
    5. Apply displacement field to correct distortions
    6. Save corrected image and field map
    
    Examples
    --------
    >>> # Basic AP/PA correction for DWI b=0
    >>> run(
    ...     data_image="ap_b0.nii.gz",
    ...     reverse_image="pa_b0.nii.gz",
    ...     output_name="corrected_b0.nii.gz",
    ...     output_warp="fieldmap.nii.gz"
    ... )
    >>> 
    >>> # LR/RL correction for fMRI
    >>> run(
    ...     data_image="lr_epi.nii.gz",
    ...     reverse_image="rl_epi.nii.gz",
    ...     output_name="corrected_epi.nii.gz",
    ...     output_warp="fieldmap.nii.gz",
    ...     phase_encoding="lr"
    ... )
    >>> 
    >>> # 4D DWI with specific volume
    >>> run(
    ...     data_image="dwi_ap.nii.gz",
    ...     reverse_image="b0_pa.nii.gz",
    ...     output_name="dwi_corrected.nii.gz",
    ...     output_warp="fieldmap.nii.gz",
    ...     shell_channel=0
    ... )
    """

    # Validate input files
    for filepath, name in [(data_image, "Data image"), (reverse_image, "Reverse image")]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{name} not found: {filepath}")
    
    # Convert phase-encoding direction to dimension index
    pe_dim = get_pe_dimension(phase_encoding)
    print(f"{CYAN}Phase-encoding direction:{RESET} {phase_encoding.upper()} (dimension: {pe_dim})")
    
    # Check GPU availability
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda:0":
        print(f"{GREEN}GPU acceleration: ENABLED (CUDA){RESET}")
    else:
        print(f"{YELLOW}GPU acceleration: DISABLED (using CPU){RESET}")
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\n{CYAN}Loading images...{RESET}")
        
        # Load images
        im1_nii = nib.load(data_image)
        affine = im1_nii.affine
        im1_data = im1_nii.get_fdata()
        print(f"  Data image: {im1_data.shape}")
        
        im2_data = nib.load(reverse_image).get_fdata()
        print(f"  Reverse image: {im2_data.shape}")

        # Convert images to ANTsImage
        print(f"\n{CYAN}Performing initial alignment...{RESET}")
        print(f"  Registration type: ANTs Affine")
        ants_im1 = ants.from_numpy(im1_data)
        ants_im2 = ants.from_numpy(im2_data)

        # Perform affine registration
        registration = ants.registration(
            fixed=ants_im1, moving=ants_im2, type_of_transform="Affine"
        )
        
        print(f"{GREEN}Initial alignment completed{RESET}")

        # Get the registered image
        registered_im2 = registration["warpedmovout"].numpy()

        # Save the registered images in the temporary directory
        print(f"\n{CYAN}Preparing for HYSCO optimization...{RESET}")
        registered_im2_path = os.path.join(temp_dir, "registered_im2.nii.gz")
        registered_im1_path = os.path.join(temp_dir, "registered_im1.nii.gz")
        registered_im2_nifti = nib.Nifti1Image(registered_im2, affine)
        nib.save(registered_im2_nifti, registered_im2_path)
        nib.save(nib.Nifti1Image(im1_data, affine), registered_im1_path)

        # Load the image and domain information
        print(f"  Creating data object...")
        data = DataObject(
            registered_im1_path,
            registered_im2_path,
            pe_dim,
            device=device,
        )
        
        # Set up the objective function
        print(f"  Setting up HYSCO objective function...")
        print(f"    Regularizer: Hyperelastic (Laplacian)")
        print(f"    Penalty parameter (rho): 1e3")
        loss_func = EPIMRIDistortionCorrection(
            data,
            300,
            1e-4,
            averaging_operator=myAvg1D,
            derivative_operator=myDiff1D,
            regularizer=myLaplacian1D,
            rho=1e3,
            PC=JacobiCG,
        )
        
        # Initialize the field map
        print(f"  Initializing B0 field map...")
        B0 = loss_func.initialize(blur_result=True)
        print(f"{GREEN}Initialization completed{RESET}")
        
        # Set up the optimizer
        print(f"\n{CYAN}Starting ADMM optimization...{RESET}")
        print(f"  Maximum iterations: 500")
        print(f"  Penalty range: [1e1, 1e6]")
        print(f"  PCG iterations: 20")
        print(f"  Gauss-Newton iterations: 1")
        resultspath = os.path.join(temp_dir, "hysco_result")  # Now inside temp_dir
        opt = ADMM(
            loss_func,
            max_iter=500,
            rho_max=1e6,
            rho_min=1e1,
            max_iter_gn=1,
            max_iter_pcg=20,
            verbose=True,
            path=resultspath,
        )
        
        # Optimize!
        opt.run_correction(B0)
        print(f"\n{GREEN}Optimization completed{RESET}")
        
        # Save field map and corrected images
        print(f"\n{CYAN}Applying correction and saving results...{RESET}")
        corr, _ = opt.apply_correction()
        
        # Move field map to output location
        shutil.move(resultspath + "-EstFieldMap.nii.gz", output_warp)
        print(f"{GREEN}Field map saved:{RESET} {output_warp}")
        
        # Load the field map
        fieldmap = nib.load(output_warp).get_fdata()

        # Ensure the warpfield has the same dimensions as the image
        if fieldmap.shape != im1_data.shape:
            print(f"{YELLOW}Warning: Fieldmap shape {fieldmap.shape} doesn't match image shape {im1_data.shape}{RESET}")
            print(f"  Auto-cropping fieldmap to match...")
            
            # Calculate center crop indices for each dimension
            crop_slices = []
            for dim in range(3):
                if fieldmap.shape[dim] > im1_data.shape[dim]:
                    # Center crop
                    diff = fieldmap.shape[dim] - im1_data.shape[dim]
                    start = diff // 2
                    end = start + im1_data.shape[dim]
                    crop_slices.append(slice(start, end))
                elif fieldmap.shape[dim] < im1_data.shape[dim]:
                    print(f"{RED}ERROR: Fieldmap dimension {dim} is smaller than image!{RESET}")
                    crop_slices.append(slice(None))
                else:
                    crop_slices.append(slice(None))
            
            fieldmap = fieldmap[tuple(crop_slices)]
            print(f"  Cropped shape: {fieldmap.shape}")
            
            cropped_fieldmap_nii = nib.Nifti1Image(fieldmap, affine)
            nib.save(cropped_fieldmap_nii, output_warp)

        # Apply the warpfield along the specified phase-encoding dimension
        print(f"\n{CYAN}Applying displacement field...{RESET}")
        warped_im = apply_warpfield(im1_data, fieldmap, pe_dim-1)

        # Handle multi-volume case if input was 4D
        if len(im1_nii.shape) > 3:
            num_volumes = im1_nii.shape[3]
            print(f"  Detected 4D input with {num_volumes} volumes")
            print(f"  Applying correction to all volumes...")
            
            corrected_volumes = []
            for vol_idx in range(num_volumes):
                if vol_idx == shell_channel:
                    # Use the already corrected volume
                    corrected_volumes.append(warped_im)
                else:
                    # Apply same warp to other volumes
                    vol_data = im1_data[:,:,:,vol_idx]
                    corrected_vol = apply_warpfield(vol_data, fieldmap, pe_dim-1)
                    corrected_volumes.append(corrected_vol)
            
            # Combine volumes back into a 4D image
            corrected_4d = np.stack(corrected_volumes, axis=3)
            output_nii = nib.Nifti1Image(corrected_4d, affine, im1_nii.header)
            nib.save(output_nii, output_name)
            print(f"{GREEN}Corrected 4D image saved:{RESET} {output_name}")
        else:
            # For 3D case, just save the single warped volume
            warped_nifti = nib.Nifti1Image(warped_im, affine)
            nib.save(warped_nifti, output_name)
            print(f"{GREEN}Corrected 3D image saved:{RESET} {output_name}")
                # Clean up temporary transform files generated by ANTs
        if 'fwdtransforms' in registration:
            for transform_file in registration['fwdtransforms']:
                if os.path.exists(transform_file):
                    try:
                        os.remove(transform_file)
                    except Exception as e:
                        print(f"{YELLOW}Warning: Could not remove {transform_file}: {e}{RESET}")
        
        if 'invtransforms' in registration:
            for transform_file in registration['invtransforms']:
                if os.path.exists(transform_file):
                    try:
                        os.remove(transform_file)
                    except Exception as e:
                        print(f"{YELLOW}Warning: Could not remove {transform_file}: {e}{RESET}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if os.path.exists(temp_dir + 'GN-'):
            shutil.rmtree(temp_dir + 'GN-')
        if os.path.exists(registered_im2_path):
            os.remove(registered_im2_path)
        if os.path.exists(registered_im1_path):
            os.remove(registered_im1_path)
    
    print(f"\n{GREEN}{BOLD}Susceptibility distortion correction completed successfully!{RESET}")
    print(f"  Input: {data_image}")
    print(f"  Reverse: {reverse_image}")
    print(f"  Output: {output_name}")
    print(f"  Field map: {output_warp}")
    print(f"  Phase-encoding: {phase_encoding.upper()}\n")

if __name__ == "__main__":
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="Correct susceptibility distortions in EPI images using HYSCO.",
        add_help=False  # Use custom help
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the data image (NIfTI file).",
    )
    parser.add_argument(
        "--reverse-image",
        type=str,
        required=True,
        help="Path to the reverse phase-encoded image (NIfTI file).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the corrected image (NIfTI file).",
    )
    parser.add_argument(
        "--output-warp", 
        type=str, 
        required=True, 
        help="Output path for the displacement field (NIfTI file)."
    )
    parser.add_argument(
        "--phase-encoding",
        type=str,
        default="ap",
        choices=["ap", "pa", "lr", "rl", "si", "is"],
        help="Phase-encoding direction (default: ap)"
    )

    args = parser.parse_args()

    try:
        run(
            args.input, 
            args.reverse_image, 
            args.output, 
            args.output_warp,
            phase_encoding=args.phase_encoding,
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
        print(f"\n{RED}{BOLD}Error during distortion correction:{RESET}")
        print(f"  {str(e)}")
        print(f"\n{YELLOW}Run 'micaflow SDC --help' for usage information.{RESET}")
        sys.exit(1)
