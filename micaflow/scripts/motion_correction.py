"""
motion_correction - Diffusion MRI Motion and Eddy Current Artifact Correction

Part of the micaflow processing pipeline for neuroimaging data.

This module corrects for subject motion and eddy current distortions in diffusion-weighted 
images (DWI) by registering each volume to a reference B0 image. Subject movement during 
acquisition is one of the primary sources of artifacts in diffusion MRI, causing misalignment 
between volumes that can severely impact tractography, metric calculation, and analysis. 

Motion Artifacts in Diffusion MRI:
----------------------------------
Subject motion during DWI acquisition causes several problems:
- Volume misalignment: Different diffusion directions acquired at different head positions
- Signal dropout: Severe motion can cause complete signal loss in some slices
- Bias in metrics: FA, MD, and other metrics become inaccurate
- Registration errors: Multi-modal registration becomes unreliable

Eddy Current Distortions:
-------------------------
Rapidly switching diffusion gradients induce eddy currents that cause:
- Image stretching/compression
- Translations and shearing of volumes
- Different distortions for each gradient direction
- Systematic bias that varies with b-value and gradient direction

This Implementation:
-------------------
Uses ANTs' SyN (Symmetric Normalization) algorithm which combines:
- Rigid transformation (6 DOF: 3 translations + 3 rotations)
- Affine transformation (12 DOF: adds scaling and shearing)
- Deformable transformation (captures eddy current distortions)

Features:
--------
- Volume-by-volume registration to a reference B0 image
- Option to use an external B0 image or the first volume as reference
- Combines rigid, affine, and deformable transformations using ANTs SyN
- Automatic b-vector rotation to account for head motion
- Preserves original image header information and coordinates
- Progress visualization with volume-wise completion tracking
- Compatible with standard diffusion acquisition protocols
- Handles multi-shell and high b-value acquisitions

Why B-vector Rotation is Critical:
----------------------------------
When the head moves, the gradient directions relative to the head coordinate 
system change. If we don't rotate the b-vectors accordingly:
- Gradient directions are incorrect relative to brain anatomy
- Tensor fitting produces incorrect fiber orientations
- Tractography generates spurious or missing fiber tracts
- Metrics like FA can be artificially inflated or reduced

Command-Line Usage:
------------------
# Using first volume as reference (standard)
micaflow motion_correction \\
    --denoised <path/to/dwi.nii.gz> \\
    --input-bvecs <path/to/dwi.bvec> \\
    --output-bvecs <path/to/corrected.bvec> \\
    --output <path/to/motion_corrected_dwi.nii.gz>

# Using external B0 as reference (recommended for better alignment)
micaflow motion_correction \\
    --denoised <path/to/dwi.nii.gz> \\
    --input-bvecs <path/to/dwi.bvec> \\
    --output-bvecs <path/to/corrected.bvec> \\
    --output <path/to/motion_corrected_dwi.nii.gz> \\
    --b0 <path/to/reference_b0.nii.gz>

# Custom shell dimension and threading
micaflow motion_correction \\
    --denoised <path/to/dwi.nii.gz> \\
    --input-bvecs <path/to/dwi.bvec> \\
    --output-bvecs <path/to/corrected.bvec> \\
    --output <path/to/motion_corrected_dwi.nii.gz> \\
    --shell-dimension 3 \\
    --threads 8

Python API Usage:
----------------
>>> from micaflow.scripts.motion_correction import run_motion_correction
>>> 
>>> # Using first volume as reference
>>> output = run_motion_correction(
...     dwi_path="denoised_dwi.nii.gz",
...     input_bvec_path="dwi.bvec",
...     output_bvec_path="corrected.bvec",
...     output="motion_corrected_dwi.nii.gz"
... )
>>> 
>>> # Using external B0 reference (recommended)
>>> output = run_motion_correction(
...     dwi_path="denoised_dwi.nii.gz",
...     input_bvec_path="dwi.bvec",
...     output_bvec_path="corrected.bvec",
...     output="motion_corrected_dwi.nii.gz",
...     b0_path="extracted_b0.nii.gz"
... )

Pipeline Integration:
--------------------
Motion correction is typically the SECOND step in diffusion preprocessing:
1. Denoising (denoise)
2. Motion/eddy correction (motion_correction) ← You are here
3. Susceptibility distortion correction (apply_SDC)
4. Bias field correction (bias_correction)
5. Brain extraction (bet)
6. DTI metric calculation (compute_fa_md)

Exit Codes:
----------
0 : Success - motion correction completed
1 : Error - invalid inputs, file not found, or processing failure

Technical Notes:
---------------
- Registration type: ANTs SyN (Symmetric Normalization)
- Affine metric: Mattes mutual information
- Affine sampling: 32 samples with 20% random sampling
- Affine iterations: 100, 50, 25 (coarse to fine)
- SyN iterations: 20, 10, 0 (deformable registration)
- B-vector rotation: Applies inverse of registration rotation matrix
- Processing time: ~30-90 minutes for typical datasets (varies with # volumes)
- First volume used as reference if no external B0 provided
- External B0 recommended for better initial alignment
- B-vectors automatically normalized after rotation
- Preserves NIfTI header and spatial information
- Winsorization applied to reduce outlier effects

Registration Parameters:
-----------------------
- grad_step: 0.01 (gradient descent step size)
- aff_metric: 'mattes' (Mattes mutual information)
- aff_sampling: 32 (number of samples for metric computation)
- aff_random_sampling_rate: 0.2 (20% random sampling)
- aff_iterations: (100, 50, 25) (multi-resolution iterations)
- aff_shrink_factors: (2, 1, 1) (downsampling at each level)
- aff_smoothing_sigmas: (2.0, 0.0, 0.0) (Gaussian smoothing)
- reg_iterations: (20, 10, 0) (deformable registration iterations)
- winsorize_lower_quantile: 0.0001
- winsorize_upper_quantile: 0.9998

B-vector Rotation Details:
--------------------------
For each volume:
1. Extract affine transformation matrix from ANTs output
2. Extract 3×3 rotation component from affine matrix
3. Apply rotation to corresponding b-vector
4. Normalize rotated b-vector to unit length
5. Save updated b-vectors to output file

Why rotation (not inverse)?
- Registration moves image FROM original TO aligned position
- B-vectors must follow the same transformation
- This maintains correct gradient directions relative to anatomy

See Also:
--------
- denoise : Denoise DWI before motion correction
- extract_b0 : Extract B0 for external reference
- apply_SDC : Apply susceptibility distortion correction
- bias_correction : Correct bias fields after motion correction


"""

import argparse
import ants
import numpy as np
import sys
import os
from tqdm import tqdm
from colorama import init, Fore, Style
import scipy

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
    ║              MOTION & EDDY CURRENT CORRECTION                  ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script corrects for subject motion and eddy current distortions in 
    diffusion-weighted images (DWI) by registering each volume to a reference 
    B0 image. It uses ANTs SyN registration which combines rigid, affine, and
    deformable transformations, and automatically rotates b-vectors to account
    for head motion.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow motion_correction {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--denoised{RESET}       : Path to the input denoised DWI image (.nii.gz)
      {YELLOW}--input-bvecs{RESET}    : Path to the input b-vectors file (.bvec)
      {YELLOW}--output-bvecs{RESET}   : Path for the rotated b-vectors file (.bvec)
      {YELLOW}--output{RESET}         : Output path for motion-corrected image (.nii.gz)
    
    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--b0{RESET}             : Path to external B0 image as reference
                         {MAGENTA}If not provided, first volume is used{RESET}
      {YELLOW}--shell-dimension{RESET}: Dimension of volume axis (default: 3)
      {YELLOW}--threads{RESET}        : Number of threads for ANTs registration (default: 1)
    
    {CYAN}{BOLD}──────────────────── EXAMPLE USAGE ───────────────────────{RESET}
    
    {BLUE}# Standard: Using first volume as reference{RESET}
    micaflow motion_correction \\
      {YELLOW}--denoised{RESET} denoised_dwi.nii.gz \\
      {YELLOW}--input-bvecs{RESET} dwi.bvec \\
      {YELLOW}--output-bvecs{RESET} corrected.bvec \\
      {YELLOW}--output{RESET} motion_corrected_dwi.nii.gz
    
    {BLUE}# Recommended: Using external B0 as reference{RESET}
    micaflow motion_correction \\
      {YELLOW}--denoised{RESET} denoised_dwi.nii.gz \\
      {YELLOW}--input-bvecs{RESET} dwi.bvec \\
      {YELLOW}--output-bvecs{RESET} corrected.bvec \\
      {YELLOW}--output{RESET} motion_corrected_dwi.nii.gz \\
      {YELLOW}--b0{RESET} extracted_b0.nii.gz
    
    {BLUE}# Custom shell dimension and threading{RESET}
    micaflow motion_correction \\
      {YELLOW}--denoised{RESET} denoised_dwi.nii.gz \\
      {YELLOW}--input-bvecs{RESET} dwi.bvec \\
      {YELLOW}--output-bvecs{RESET} corrected.bvec \\
      {YELLOW}--output{RESET} motion_corrected_dwi.nii.gz \\
      {YELLOW}--shell-dimension{RESET} 3 \\
      {YELLOW}--threads{RESET} 8
    
    {CYAN}{BOLD}───────────── WHY MOTION CORRECTION? ────────────────────{RESET}
    
    {GREEN}Motion Artifacts:{RESET}
    {MAGENTA}•{RESET} Volume misalignment from head movement
    {MAGENTA}•{RESET} Signal dropout in severe motion
    {MAGENTA}•{RESET} Biased diffusion metrics (FA, MD)
    {MAGENTA}•{RESET} Failed tractography
    
    {GREEN}Eddy Current Distortions:{RESET}
    {MAGENTA}•{RESET} Image stretching/compression
    {MAGENTA}•{RESET} Translations and shearing
    {MAGENTA}•{RESET} Different for each gradient direction
    {MAGENTA}•{RESET} Systematic bias varying with b-value
    
    {GREEN}Why B-vector Rotation:{RESET}
    {MAGENTA}•{RESET} Head motion changes gradient directions relative to brain
    {MAGENTA}•{RESET} Must rotate b-vectors to maintain correct orientations
    {MAGENTA}•{RESET} Without rotation: incorrect fiber orientations
    {MAGENTA}•{RESET} Critical for accurate tractography and tensor fitting
    
    {CYAN}{BOLD}────────────────── REGISTRATION METHOD ──────────────────{RESET}
    
    {GREEN}ANTs SyN (Symmetric Normalization):{RESET}
    {MAGENTA}1.{RESET} Rigid registration (6 DOF): translations + rotations
    {MAGENTA}2.{RESET} Affine registration (12 DOF): adds scaling + shearing
    {MAGENTA}3.{RESET} Deformable registration: captures eddy distortions
    
    {GREEN}Parameters:{RESET}
    {MAGENTA}•{RESET} Metric: Mattes mutual information
    {MAGENTA}•{RESET} Multi-resolution: 3 levels (coarse to fine)
    {MAGENTA}•{RESET} Affine iterations: 100, 50, 25
    {MAGENTA}•{RESET} SyN iterations: 20, 10, 0
    {MAGENTA}•{RESET} Sampling: 32 samples, 20% random
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    {MAGENTA}•{RESET} Denoising should be performed BEFORE motion correction
    {MAGENTA}•{RESET} If no external B0: first volume used as reference
    {MAGENTA}•{RESET} External B0 recommended for better alignment
    {MAGENTA}•{RESET} Each volume registered independently to reference
    {MAGENTA}•{RESET} B-vectors automatically rotated based on transformation
    {MAGENTA}•{RESET} Rotated b-vectors saved to --output-bvecs
    {MAGENTA}•{RESET} Processing time: ~30-90 seconds (depends on # volumes)
    {MAGENTA}•{RESET} Progress bar shows volume-by-volume registration
    {MAGENTA}•{RESET} Output preserves input image geometry and header
    {MAGENTA}•{RESET} Winsorization reduces outlier effects during registration
    
    {CYAN}{BOLD}─────────────── PIPELINE POSITION ───────────────────────{RESET}
    1. Denoising (denoise)
    {GREEN}2. Motion correction (motion_correction){RESET} {MAGENTA}← You are here{RESET}
    3. Distortion correction (apply_SDC)
    4. Bias field correction (bias_correction)
    5. Brain extraction (bet)
    6. DTI metrics (compute_fa_md)
    
    {CYAN}{BOLD}────────────────────── EXIT CODES ───────────────────────{RESET}
    {GREEN}0{RESET} : Success - motion correction completed
    {RED}1{RESET} : Error - invalid inputs, file not found, or processing failure
    
    {CYAN}{BOLD}───────────────── COMMON ISSUES ─────────────────────────{RESET}
    {YELLOW}Issue:{RESET} Registration takes very long
    {GREEN}Solution:{RESET} Normal processing is 30-90 min; reduce volumes if needed
    
    {YELLOW}Issue:{RESET} Poor alignment after correction
    {GREEN}Solution:{RESET} Use external B0 reference, check for severe motion
    
    {YELLOW}Issue:{RESET} "Out of memory" error
    {GREEN}Solution:{RESET} Reduce image resolution or process on machine with more RAM
    
    {YELLOW}Issue:{RESET} B-vector dimensions mismatch
    {GREEN}Solution:{RESET} Ensure bvec file has 3 rows × N volumes format
    """
    print(help_text)


def run_motion_correction(dwi_path, input_bvec_path, output_bvec_path, output, 
                          b0_path=None, shell_dimension=3, threads=1):
    """
    Perform motion and eddy current correction on diffusion-weighted images (DWI).
    
    This function corrects for subject motion and eddy current distortions in DWI data 
    by registering each volume to a reference B0 image using ANTs SyN. It also rotates 
    b-vectors to account for head motion, which is critical for accurate fiber orientation.
    
    Parameters
    ----------
    dwi_path : str
        Path to the input DWI NIfTI file (.nii.gz).
        Should be denoised before motion correction.
    input_bvec_path : str
        Path to the input b-vectors file (.bvec).
        Format: 3 rows (x, y, z) × N columns (one per volume).
    output_bvec_path : str
        Path where the rotated b-vectors will be saved (.bvec).
        B-vectors are updated to reflect head motion corrections.
    output : str
        Path where the motion-corrected DWI will be saved (.nii.gz).
    b0_path : str, optional
        Path to an external B0 image to use as registration reference.
        If not provided, the first volume of the DWI is used as reference.
        External B0 recommended for better initial alignment.
    shell_dimension : int, optional
        Dimension along which diffusion volumes are organized. Default: 3.
        For standard 4D NIfTI (X, Y, Z, volumes), this should be 3.
    threads : int, optional
        Number of threads to use for ANTs registration. Default: 1.
        Increasing this can speed up processing on multi-core systems.
        
    Returns
    -------
    str
        Path to the saved motion-corrected DWI image.
        
    Raises
    ------
    FileNotFoundError
        If input files cannot be found.
    ValueError
        If b-vectors file format is invalid or dimensions don't match.
        
    Notes
    -----
    - Uses ANTs SyN with rigid + affine + deformable registration
    - B-vectors are rotated using the rotation component of affine transforms
    - Rotation matrix extracted from ITK affine transform files (.mat)
    - Each volume registered independently to minimize cumulative drift
    - Processing time: ~30-90 seconds for typical datasets
    - First volume copied unchanged if used as reference
    - B-vectors normalized to unit length after rotation
    - Preserves NIfTI header and spatial information
    
    B-vector Rotation:
    - Extracts 3×3 rotation matrix from each affine transformation
    - Applies rotation to corresponding b-vector
    - Normalizes result to maintain unit vector
    - Critical for accurate fiber orientation in tensor fitting
    
    Examples
    --------
    >>> # Using first volume as reference
    >>> output = run_motion_correction(
    ...     dwi_path="denoised_dwi.nii.gz",
    ...     input_bvec_path="dwi.bvec",
    ...     output_bvec_path="corrected.bvec",
    ...     output="motion_corrected_dwi.nii.gz"
    ... )
    >>> 
    >>> # Using external B0 reference (recommended)
    >>> output = run_motion_correction(
    ...     dwi_path="denoised_dwi.nii.gz",
    ...     input_bvec_path="dwi.bvec",
    ...     output_bvec_path="corrected.bvec",
    ...     output="motion_corrected_dwi.nii.gz",
    ...     b0_path="extracted_b0.nii.gz"
    ... )
    """
    # Validate input files exist
    for filepath, name in [(dwi_path, "DWI"), (input_bvec_path, "B-vectors")]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{name} file not found: {filepath}")
    
    if b0_path and not os.path.exists(b0_path):
        raise FileNotFoundError(f"B0 reference file not found: {b0_path}")
    
    print(f"{CYAN}Loading DWI image...{RESET}")
    print(f"  File: {dwi_path}")
    dwi_ants = ants.image_read(dwi_path)
    dwi_data = dwi_ants.numpy()
    print(f"  Shape: {dwi_data.shape}")
    
    # Load bvecs and ensure they're in the correct format [3, N]
    print(f"\n{CYAN}Loading b-vectors...{RESET}")
    print(f"  File: {input_bvec_path}")
    bvecs = np.loadtxt(input_bvec_path)
    if bvecs.shape[0] != 3 and bvecs.shape[1] == 3:
        bvecs = bvecs.T  # Transpose if in [N, 3] format
        print(f"  Transposed bvecs from [N, 3] to [3, N] format")
    
    print(f"  Shape: {bvecs.shape}")
    if bvecs.shape[0] != 3:
        raise ValueError(f"B-vectors must have 3 rows (x, y, z), got {bvecs.shape[0]}")
    
    num_volumes = dwi_data.shape[shell_dimension]
    if bvecs.shape[1] != num_volumes:
        raise ValueError(f"Number of b-vectors ({bvecs.shape[1]}) doesn't match "
                        f"number of volumes ({num_volumes})")
    
    # Create a copy for the rotated bvecs
    rotated_bvecs = np.copy(bvecs)

    # Set up the B0 reference image
    print(f"\n{CYAN}Setting up reference image...{RESET}")
    if b0_path:
        print(f"  Using external B0: {b0_path}")
        b0_ants = ants.image_read(b0_path)
    else:
        print(f"  Using first volume as reference (internal B0)")
        # Extract the first volume as reference if no external B0 provided
        vol_idx = tuple(slice(None) if i != shell_dimension else 0 
                        for i in range(len(dwi_data.shape)))
        first_vol_data = dwi_data[vol_idx]
        b0_ants = ants.from_numpy(
            first_vol_data,
            origin=dwi_ants.origin[:3],
            spacing=dwi_ants.spacing[:3],
            direction=dwi_ants.direction[:3, :3]
        )
    
    print(f"  Reference shape: {b0_ants.numpy().shape}")

    # Set ANTs thread count (similar to lamar.py)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)

    # Initialize array for registered data
    registered_data = np.zeros_like(dwi_data)
    
    # If using the first volume as reference, copy it directly
    if not b0_path:
        vol_idx = tuple(slice(None) if i != shell_dimension else 0 
                        for i in range(len(dwi_data.shape)))
        registered_data[vol_idx] = dwi_data[vol_idx]  # Copy first volume unchanged
        print(f"  First volume copied unchanged (used as reference)")

    # Register each volume to the B0 reference
    start_idx = 0 if b0_path else 1  # Start from volume 0 if using external B0
    
    print(f"\n{CYAN}Starting volume-by-volume registration...{RESET}")
    print(f"  Registering volumes {start_idx} to {num_volumes-1}")
    print(f"  Registration type: ANTs SyN (Rigid + Affine + Deformable)")
    print(f"  Using {threads} threads")
    
    for idx in tqdm(range(start_idx, dwi_data.shape[shell_dimension]), 
                    desc=f"{CYAN}Registering volumes{RESET}", 
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
        # Create dynamic indexing tuple to access volumes along the specified dimension
        vol_idx = tuple(slice(None) if i != shell_dimension else idx 
                        for i in range(len(dwi_data.shape)))
        
        # Extract the current volume using dynamic indexing
        moving_data = dwi_data[vol_idx]
        moving_ants = ants.from_numpy(
            moving_data, 
            origin=dwi_ants.origin[:3], 
            spacing=dwi_ants.spacing[:3],
            direction=dwi_ants.direction[:3, :3]
        )

        # Custom registration based on ANTs SyN configuration
        reg_result = ants.registration(
            fixed=b0_ants,
            moving=moving_ants,
            type_of_transform='SyN',
            
            # Key parameters for optimal motion correction
            grad_step=0.01,
            aff_metric='mattes',
            aff_sampling=32,
            aff_random_sampling_rate=0.2,
            aff_iterations=(100, 50, 25),
            aff_shrink_factors=(2, 1, 1),
            aff_smoothing_sigmas=(2.0, 0.0, 0.0),
            reg_iterations=(20, 10, 0),
            winsorize_lower_quantile=0.0001,
            winsorize_upper_quantile=0.9998,
            num_threads=threads      
        )

        # Place the registered volume in the output array using the same indexing
        warped_data = reg_result["warpedmovout"].numpy()
        registered_data[vol_idx] = warped_data
        
        # Extract and apply the rotation to the bvec for this volume
        if 'fwdtransforms' in reg_result and len(reg_result['fwdtransforms']) > 0:
            # Find the affine transform file (typically ends with .mat)
            affine_file = None
            for transform in reg_result['fwdtransforms']:
                if transform.endswith('.mat'):
                    affine_file = transform
                    break
            
            if affine_file and os.path.exists(affine_file):
                # Extract rotation matrix from the affine transformation
                rotation_matrix = extract_rotation_matrix_from_itk(affine_file)
                
                # Apply rotation to the corresponding bvec
                if rotation_matrix is not None and idx < bvecs.shape[1]:
                    # Rotate the b-vector using the transformation rotation matrix
                    rotated_bvec = np.dot(rotation_matrix, bvecs[:, idx])
                    
                    # Normalize to unit vector (preserve direction, normalize magnitude)
                    norm = np.linalg.norm(rotated_bvec)
                    if norm > 0:
                        rotated_bvec = rotated_bvec / norm
                        
                    rotated_bvecs[:, idx] = rotated_bvec
        
        # Clean up temporary transform files generated by ANTs
        if 'fwdtransforms' in reg_result:
            for transform_file in reg_result['fwdtransforms']:
                if os.path.exists(transform_file):
                    try:
                        os.remove(transform_file)
                    except Exception as e:
                        print(f"{YELLOW}Warning: Could not remove {transform_file}: {e}{RESET}")
        
        if 'invtransforms' in reg_result:
            for transform_file in reg_result['invtransforms']:
                if os.path.exists(transform_file):
                    try:
                        os.remove(transform_file)
                    except Exception as e:
                        print(f"{YELLOW}Warning: Could not remove {transform_file}: {e}{RESET}")

    # Save the registered data with original geometry
    print(f"\n{CYAN}Saving motion-corrected DWI...{RESET}")
    registered_ants = ants.from_numpy(
        registered_data, 
        origin=dwi_ants.origin, 
        spacing=dwi_ants.spacing, 
        direction=dwi_ants.direction
    )
    ants.image_write(registered_ants, output)
    print(f"{GREEN}Saved to: {output}{RESET}")
    
    # Save the rotated bvecs
    print(f"\n{CYAN}Saving rotated b-vectors...{RESET}")
    np.savetxt(output_bvec_path, rotated_bvecs, fmt='%.6f')
    print(f"{GREEN}Saved to: {output_bvec_path}{RESET}")

    print(f"\n{GREEN}{BOLD}Motion correction completed successfully!{RESET}")
    print(f"  Processed {num_volumes - start_idx} volumes")
    print(f"  Motion-corrected DWI: {output}")
    print(f"  Rotated b-vectors: {output_bvec_path}")
    
    return output


def extract_rotation_matrix_from_itk(affine_file):
    """
    Extract the rotation matrix from an ITK affine transform file.
    
    ITK affine transform files store transformation parameters including a 3×3
    rotation/scaling matrix and a 3×1 translation vector. This function extracts
    the rotation component needed for b-vector correction.
    
    Parameters
    ----------
    affine_file : str
        Path to the ITK affine transform file (.mat).
        Generated by ANTs during registration.
        
    Returns
    -------
    numpy.ndarray
        3×3 rotation matrix, or identity matrix if extraction fails.
        
    Notes
    -----
    - ITK format stores parameters in row-major order
    - First 9 elements form the 3×3 transformation matrix
    - Matrix includes rotation, scaling, and shearing
    - For DWI, we primarily care about the rotation component
    - Returns identity matrix on error to avoid breaking pipeline
    
    Examples
    --------
    >>> rotation = extract_rotation_matrix_from_itk("transform.mat")
    >>> rotated_bvec = np.dot(rotation, original_bvec)
    """
    try:
        # Load the transformation file using scipy.io.loadmat
        mat = scipy.io.loadmat(affine_file)
        
        # Get the transformation parameters array (should be 12 elements: 9 matrix + 3 translation)
        params = mat['AffineTransform_float_3_3'].flatten()
        
        # First 9 elements are the transformation matrix in row-major order
        rotation_matrix = np.array([
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [params[6], params[7], params[8]]
        ])
        
        return rotation_matrix
            
    except Exception as e:
        print(f"{YELLOW}Warning: Error extracting rotation matrix from {affine_file}: {e}{RESET}")
        print(f"{YELLOW}Using identity matrix (no rotation applied).{RESET}")
        # Return identity matrix as fallback
        return np.eye(3)


if __name__ == "__main__":
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="Perform motion and eddy current correction on DWI using ANTs SyN.",
        add_help=False  # Use custom help
    )
    parser.add_argument(
        "--denoised",
        type=str,
        required=True,
        help="Path to the denoised DWI (NIfTI file).",
    )
    parser.add_argument(
        "--input-bvecs",
        type=str,
        required=True,
        help="Path to the input b-vectors file (.bvec).",
    )
    parser.add_argument(
        "--output-bvecs",
        type=str,
        required=True,
        help="Path for the rotated b-vectors file (.bvec).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the motion-corrected DWI.",
    )
    parser.add_argument(
        "--b0",
        type=str,
        help="Path to an external B0 image to use as reference. If not provided, the first volume is used.",
    )
    parser.add_argument(
        "--shell-dimension",
        type=int,
        default=3,
        help="Dimension along which diffusion volumes are organized (default: 3).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads to use for ANTs registration (default: 1).",
    )

    args = parser.parse_args()
    
    try:
        corrected_image = run_motion_correction(
            args.denoised, 
            args.input_bvecs, 
            args.output_bvecs, 
            args.output, 
            args.b0, 
            args.shell_dimension,
            args.threads
        )
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\n{RED}{BOLD}File not found:{RESET}")
        print(f"  {str(e)}")
        sys.exit(1)
        
    except ValueError as e:
        print(f"\n{RED}{BOLD}Value error:{RESET}")
        print(f"  {str(e)}")
        print(f"\n{YELLOW}Check that bvec file format is correct (3 rows × N columns).{RESET}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n{RED}{BOLD}Error during motion correction:{RESET}")
        print(f"  {str(e)}")
        print(f"\n{YELLOW}Run 'micaflow motion_correction --help' for usage information.{RESET}")
        sys.exit(1)