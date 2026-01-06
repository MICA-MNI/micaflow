#!/usr/bin/env python3
"""
extract_b0 - B0 Volume Extraction and Dataset Splitting Tool

Part of the micaflow processing pipeline for neuroimaging data.

This module extracts b=0 (non-diffusion-weighted) volumes from diffusion-weighted 
images (DWI) and separates them from the diffusion-weighted volumes. B0 volumes are 
acquired without diffusion weighting (b-value ≈ 0 s/mm²) and serve as reference images 
for various DWI processing steps including motion correction, distortion correction, 
and registration to structural images.

What are B0 Volumes?
-------------------
B0 volumes (also called "b=0" or "non-diffusion-weighted") are MRI volumes acquired 
without any diffusion-weighting gradients applied. They have:
- B-value = 0 s/mm² (or very close to 0, typically < 50 s/mm²)
- Higher SNR compared to diffusion-weighted volumes
- T2-weighted contrast showing anatomy without diffusion information
- Serve as anatomical reference for DWI processing

Common Use Cases:
----------------
1. Separate Processing: Some tools require b0 and DWI volumes to be processed separately
2. Registration Reference: Use b0 as reference for registering DWI to structural scans
3. Distortion Correction: b0 volumes are used to estimate and correct distortions
4. Motion Correction: b0 serves as reference for aligning diffusion volumes
5. Quality Assessment: b0 provides anatomical reference for checking alignment
6. Brain Extraction: Create masks from high-SNR b0 volumes

Features:
--------
- Automatically identifies b0 volumes based on b-value threshold
- Extracts b0 volume and saves it separately
- Optionally saves remaining DWI volumes without b0
- Updates b-value and b-vector files to match extracted volumes
- Can save b0-specific gradient files (bval/bvec)
- Supports manual volume index specification
- Handles flexible dimension ordering
- Preserves image geometry and header information

Command-Line Usage:
------------------
# Basic extraction (b0 only)
micaflow extract_b0 \\
    --input <path/to/dwi.nii.gz> \\
    --bvals <path/to/dwi.bval> \\
    --output <path/to/b0.nii.gz>

# Full split (b0 + DWI + updated gradient files)
micaflow extract_b0 \\
    --input <path/to/dwi.nii.gz> \\
    --bvals <path/to/dwi.bval> \\
    --bvecs <path/to/dwi.bvec> \\
    --output <path/to/b0.nii.gz> \\
    --output-dwi <path/to/dwi_no_b0.nii.gz> \\
    --output-bvals <path/to/dwi_no_b0.bval> \\
    --output-bvecs <path/to/dwi_no_b0.bvec>

# With b0 gradient files
micaflow extract_b0 \\
    --input <path/to/dwi.nii.gz> \\
    --bvals <path/to/dwi.bval> \\
    --bvecs <path/to/dwi.bvec> \\
    --output <path/to/b0.nii.gz> \\
    --b0-bval <path/to/b0.bval> \\
    --b0-bvec <path/to/b0.bvec>

# Custom threshold
micaflow extract_b0 \\
    --input <path/to/dwi.nii.gz> \\
    --bvals <path/to/dwi.bval> \\
    --output <path/to/b0.nii.gz> \\
    --threshold 100

# Manual index specification
micaflow extract_b0 \\
    --input <path/to/dwi.nii.gz> \\
    --output <path/to/specific_volume.nii.gz> \\
    --index 0

Python API Usage:
----------------
>>> from micaflow.scripts.extract_b0 import extract_b0
>>> 
>>> # Basic extraction
>>> b0_img = extract_b0(
...     dwi_path="dwi.nii.gz",
...     bvals_path="dwi.bval",
...     output_path="b0.nii.gz"
... )
>>> 
>>> # Full split with gradient updates
>>> b0_img = extract_b0(
...     dwi_path="dwi.nii.gz",
...     bvals_path="dwi.bval",
...     bvecs_path="dwi.bvec",
...     output_path="b0.nii.gz",
...     output_dwi="dwi_no_b0.nii.gz",
...     output_bvals="dwi_no_b0.bval",
...     output_bvecs="dwi_no_b0.bvec"
... )
>>> 
>>> # With b0-specific gradient files
>>> b0_img = extract_b0(
...     dwi_path="dwi.nii.gz",
...     bvals_path="dwi.bval",
...     bvecs_path="dwi.bvec",
...     output_path="b0.nii.gz",
...     b0_bval="b0.bval",
...     b0_bvec="b0.bvec"
... )

Pipeline Integration:
--------------------
B0 extraction is typically performed when:
1. Tools require separate b0 and DWI processing
2. Registration to structural images (use b0 as moving image)
3. Creating brain masks from high-SNR b0
4. Multi-step registration pipelines (DWI→b0→T1w→MNI)
5. Quality control and visual inspection

Workflow example:
1. Extract b0 (extract_b0) 
2. Register b0 to T1w (coregister)
3. Apply transform to DWI (apply_warp)
4. Continue DWI processing...

Exit Codes:
----------
0 : Success - b0 extraction completed
1 : Error - invalid inputs, file not found, or processing failure

Technical Notes:
---------------
- Default b0 threshold: 50 s/mm² (volumes with b-value ≤ 50 considered b0)
- Extracts the FIRST b0 volume found (lowest index)
- Multiple b0 volumes: only first is extracted (can specify --index for others)
- Gradient files updated to match extracted volumes
- B0-specific gradient files contain single value/vector
- Non-b0 gradient files exclude the extracted b0 entry
- Shell dimension default is 3 (last dimension, standard 4D format)
- Manual index specification overrides b-value threshold
- Negative indices supported (e.g., -1 for last volume)
- Preserves NIfTI header and spatial information

File Format Notes:
-----------------
B-values (.bval):
  - Space-separated values, one per volume
  - Units: s/mm²
  - Example: "0 1000 1000 2000 2000"

B-vectors (.bvec):
  - 3 rows (x, y, z) × N columns (one per volume)
  - Normalized direction vectors
  - Example:
    0.0 0.707 -0.707 0.577 -0.577
    0.0 0.707  0.707 0.577 -0.577
    0.0 0.0    0.0   0.577  0.577

See Also:
--------
- coregister : Register b0 to structural images
- apply_warp : Apply transformations to DWI volumes
- bet : Extract brain from b0 volumes
- denoise : Denoise DWI (typically before splitting)

"""

import ants
import nibabel as nib
import numpy as np
import argparse
import sys
import shutil
import os
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
    ║                       B0 VOLUME EXTRACTION                     ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script extracts b=0 (non-diffusion-weighted) volumes from a diffusion-
    weighted image (DWI), saves the remaining DWI volumes separately, and updates 
    gradient files to match the extracted data.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow extract_b0 {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--input{RESET}       : Path to input DWI image (.nii.gz)
      {YELLOW}--output{RESET}      : Path for output b0 volume (.nii.gz)
    
    {CYAN}{BOLD}────────────── IDENTIFICATION ARGUMENTS ──────────────────{RESET}
    {MAGENTA}One of these must be provided to identify b0:{RESET}
      {YELLOW}--bvals{RESET}       : Path to b-values file (.bval)
                     {MAGENTA}Auto-detects b0 using threshold{RESET}
      {YELLOW}--index{RESET}       : Directly specify volume index to extract
                     {MAGENTA}Overrides bvals-based detection{RESET}
    
    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--bvecs{RESET}       : Path to input b-vectors file (.bvec)
      {YELLOW}--output-dwi{RESET}  : Path for output non-b0 volumes (.nii.gz)
      {YELLOW}--output-bvals{RESET}: Path for output b-values (non-b0 only)
      {YELLOW}--output-bvecs{RESET}: Path for output b-vectors (non-b0 only)
      {YELLOW}--b0-bval{RESET}     : Path for b0-only b-value file
      {YELLOW}--b0-bvec{RESET}     : Path for b0-only b-vector file
      {YELLOW}--threshold{RESET}   : Max b-value to consider as b0 (default: 50 s/mm²)
      {YELLOW}--shell-dimension{RESET}: Dimension of volume axis (default: 3)
    
    {CYAN}{BOLD}──────────────────── EXAMPLE USAGE ───────────────────────{RESET}
    
    {BLUE}# Example 1: Basic b0 extraction{RESET}
    micaflow extract_b0 \\
      {YELLOW}--input{RESET} dwi.nii.gz \\
      {YELLOW}--bvals{RESET} dwi.bval \\
      {YELLOW}--output{RESET} b0.nii.gz
    
    {BLUE}# Example 2: Full dataset split{RESET}
    micaflow extract_b0 \\
      {YELLOW}--input{RESET} dwi.nii.gz \\
      {YELLOW}--bvals{RESET} dwi.bval \\
      {YELLOW}--bvecs{RESET} dwi.bvec \\
      {YELLOW}--output{RESET} b0.nii.gz \\
      {YELLOW}--output-dwi{RESET} dwi_no_b0.nii.gz \\
      {YELLOW}--output-bvals{RESET} dwi_no_b0.bval \\
      {YELLOW}--output-bvecs{RESET} dwi_no_b0.bvec
    
    {BLUE}# Example 3: With b0 gradient files{RESET}
    micaflow extract_b0 \\
      {YELLOW}--input{RESET} dwi.nii.gz \\
      {YELLOW}--bvals{RESET} dwi.bval \\
      {YELLOW}--bvecs{RESET} dwi.bvec \\
      {YELLOW}--output{RESET} b0.nii.gz \\
      {YELLOW}--b0-bval{RESET} b0.bval \\
      {YELLOW}--b0-bvec{RESET} b0.bvec
    
    {BLUE}# Example 4: Custom threshold{RESET}
    micaflow extract_b0 \\
      {YELLOW}--input{RESET} dwi.nii.gz \\
      {YELLOW}--bvals{RESET} dwi.bval \\
      {YELLOW}--output{RESET} b0.nii.gz \\
      {YELLOW}--threshold{RESET} 100
    
    {BLUE}# Example 5: Manual index specification{RESET}
    micaflow extract_b0 \\
      {YELLOW}--input{RESET} dwi.nii.gz \\
      {YELLOW}--output{RESET} volume_0.nii.gz \\
      {YELLOW}--index{RESET} 0
    
    {CYAN}{BOLD}──────────────────── WHAT ARE B0 VOLUMES? ────────────────{RESET}
    
    {GREEN}Definition:{RESET}
    B0 volumes (b=0) are MRI volumes acquired WITHOUT diffusion weighting.
    
    {GREEN}Characteristics:{RESET}
    {MAGENTA}•{RESET} B-value ≈ 0 s/mm² (no diffusion gradients applied)
    {MAGENTA}•{RESET} Higher SNR than diffusion-weighted volumes
    {MAGENTA}•{RESET} T2-weighted contrast showing anatomy
    {MAGENTA}•{RESET} No directional diffusion information
    
    {GREEN}Common Uses:{RESET}
    {MAGENTA}•{RESET} Reference for motion/distortion correction
    {MAGENTA}•{RESET} Registration to structural images
    {MAGENTA}•{RESET} Brain mask creation (high SNR)
    {MAGENTA}•{RESET} Quality control and visualization
    {MAGENTA}•{RESET} Multi-step registration pipelines
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    {MAGENTA}•{RESET} Default b0 threshold: 50 s/mm² (volumes with b ≤ 50 are b0)
    {MAGENTA}•{RESET} Extracts the FIRST b0 volume found (lowest index)
    {MAGENTA}•{RESET} Multiple b0s: use --index to extract specific volume
    {MAGENTA}•{RESET} Negative indices supported (e.g., -1 for last volume)
    {MAGENTA}•{RESET} Manual --index overrides automatic b-value detection
    {MAGENTA}•{RESET} Output preserves input image geometry and header
    {MAGENTA}•{RESET} Gradient files (.bval/.bvec) automatically updated
    {MAGENTA}•{RESET} B0 gradient files contain single entry for extracted volume
    {MAGENTA}•{RESET} Non-b0 gradient files exclude the extracted b0 entry
    
    {CYAN}{BOLD}─────────────── OUTPUT FILE DESCRIPTIONS ────────────────{RESET}
    {GREEN}--output{RESET}         : Extracted b0 volume (3D NIfTI)
    {GREEN}--output-dwi{RESET}     : Remaining DWI volumes without b0 (4D NIfTI)
    {GREEN}--output-bvals{RESET}   : B-values for non-b0 volumes only
    {GREEN}--output-bvecs{RESET}   : B-vectors for non-b0 volumes only
    {GREEN}--b0-bval{RESET}        : B-value for extracted b0 (single value)
    {GREEN}--b0-bvec{RESET}        : B-vector for extracted b0 (3 values)
    
    {CYAN}{BOLD}────────────────────── EXIT CODES ───────────────────────{RESET}
    {GREEN}0{RESET} : Success - b0 extraction completed
    {RED}1{RESET} : Error - invalid inputs, file not found, or processing failure
    
    {CYAN}{BOLD}───────────────── COMMON ISSUES ─────────────────────────{RESET}
    {YELLOW}Issue:{RESET} "No b-value below threshold found"
    {GREEN}Solution:{RESET} Increase --threshold or use --index to specify manually
    
    {YELLOW}Issue:{RESET} "Number of bvals doesn't match volumes"
    {GREEN}Solution:{RESET} Check bvals file format or use --index instead
    
    {YELLOW}Issue:{RESET} Need to extract a specific b0 (not first one)
    {GREEN}Solution:{RESET} Use --index to specify the exact volume index
    """
    print(help_text)


def extract_b0(dwi_path, bvals_path=None, bvecs_path=None, output_path=None, 
               output_dwi=None, output_bvals=None, output_bvecs=None, 
               threshold=50, shell_index=None, shell_dimension=3, b0_bval=None, b0_bvec=None):
    """
    Extract b=0 volume(s) from a DWI image, save non-b0 volumes, and update gradient files.
    
    This function extracts b0 (non-diffusion-weighted) volumes from diffusion-weighted 
    images. B0 volumes can be identified automatically using b-values and a threshold, 
    or manually specified by index. The function can also split the dataset into b0 
    and non-b0 components with updated gradient files.
    
    Parameters
    ----------
    dwi_path : str
        Path to the input DWI image file (.nii.gz).
        Must be a 4D volume.
    bvals_path : str, optional
        Path to the b-values file (.bval).
        Required if shell_index is not specified.
    bvecs_path : str, optional
        Path to the b-vectors file (.bvec).
        Required if updating b-vectors.
    output_path : str, optional
        Path where to save the extracted b0 volume (.nii.gz).
        Output will be 3D.
    output_dwi : str, optional
        Path where to save the non-b0 volumes (.nii.gz).
        Output will be 4D with b0 volume removed.
    output_bvals : str, optional
        Path where to save the updated bvals file (non-b0 only).
        Requires bvals_path to be provided.
    output_bvecs : str, optional
        Path where to save the updated bvecs file (non-b0 only).
        Requires bvecs_path to be provided.
    threshold : int or float, optional
        Maximum b-value to consider as b0. Default: 50 s/mm².
        Volumes with b-value ≤ threshold are considered b0.
    shell_index : int, optional
        Directly specify the volume index to extract.
        Overrides automatic b-value-based detection.
        Supports negative indexing (e.g., -1 for last volume).
    shell_dimension : int, optional
        Dimension along which volumes are stacked. Default: 3 (last dimension).
        For standard 4D NIfTI (X, Y, Z, volumes), this should be 3.
    b0_bval : str, optional
        Path where to save the b0-only bval file.
        Contains single b-value for extracted b0.
    b0_bvec : str, optional
        Path where to save the b0-only bvec file.
        Contains single b-vector (3 values) for extracted b0.
    
    Returns
    -------
    None
        Files are saved to specified output paths.
        
    Raises
    ------
    FileNotFoundError
        If input files cannot be found.
    ValueError
        If input is not 4D or dimensions don't match.
        
    Notes
    -----
    - Extracts the FIRST b0 volume found (unless shell_index specified)
    - B0 threshold default is 50 s/mm² (standard for DWI)
    - Gradient files must have entries matching number of volumes
    - B-vectors file format: 3 rows × N columns
    - B-values file format: space-separated values
    - Preserves NIfTI header and spatial information
    
    Examples
    --------
    >>> # Basic b0 extraction
    >>> extract_b0(
    ...     dwi_path="dwi.nii.gz",
    ...     bvals_path="dwi.bval",
    ...     output_path="b0.nii.gz"
    ... )
    >>> 
    >>> # Full dataset split
    >>> extract_b0(
    ...     dwi_path="dwi.nii.gz",
    ...     bvals_path="dwi.bval",
    ...     bvecs_path="dwi.bvec",
    ...     output_path="b0.nii.gz",
    ...     output_dwi="dwi_no_b0.nii.gz",
    ...     output_bvals="dwi_no_b0.bval",
    ...     output_bvecs="dwi_no_b0.bvec"
    ... )
    >>> 
    >>> # With b0-specific gradient files
    >>> extract_b0(
    ...     dwi_path="dwi.nii.gz",
    ...     bvals_path="dwi.bval",
    ...     bvecs_path="dwi.bvec",
    ...     output_path="b0.nii.gz",
    ...     b0_bval="b0.bval",
    ...     b0_bvec="b0.bvec"
    ... )
    >>> 
    >>> # Manual index specification
    >>> extract_b0(
    ...     dwi_path="dwi.nii.gz",
    ...     output_path="volume_5.nii.gz",
    ...     shell_index=5
    ... )
    """
    # Load the DWI image
    print(f"{CYAN}Loading DWI image...{RESET}")
    print(f"  File: {dwi_path}")
    try:
        img_nib = nib.load(dwi_path)
        img_data = img_nib.get_fdata()
        img_shape = img_data.shape
        img_affine = img_nib.affine
        print(f"  Shape: {img_shape}")
    except Exception as e:
        print(f"{RED}Error loading DWI image: {e}{RESET}")
        return None
    
    # Check if input is 4D
    if len(img_shape) == 3:
        print(f"{YELLOW}Input image is 3D ({img_shape}). Checking if it is a valid b0...{RESET}")
        
        # If 3D, we effectively have 1 volume
        num_volumes = 1
        bvals = None  # Initialize bvals variable for this scope

        if bvals_path:
            try:
                with open(bvals_path, 'r') as f:
                    # Read all values, likely just one for a 3D image usually, 
                    # but maybe the user provided a full bval string?
                    # If it's a single volume extraction, bvals might still contain just 1 number.
                    vals = [float(val) for val in f.read().strip().split()]
                    bvals = vals  # Assign to the variable name used later
                    
                # If there's exactly 1 b-value, check it
                if len(vals) == 1:
                    b_val = vals[0]
                    if b_val > threshold:
                        print(f"{RED}Error: Input 3D image has b-value {b_val} > threshold {threshold}. Not a b0.{RESET}")
                        return None
                    else:
                        print(f"{GREEN}Input is a valid b0 (b={b_val}).{RESET}")
                
                # If there are multiple b-values but input is 3D, it implies a mismatch 
                # OR the user gave a full bvals file for a single extracted volume.
                # In the 'single 3D file' context, we assume the single volume corresponds to the first bval or 'a' bval is risky
                # without more info, but commonly 3D means "this is the volume".
                # Let's fallback to checking if the *first* value is b0 if no index.
                elif shell_index is not None and shell_index < len(vals):
                     if vals[shell_index] > threshold:
                        print(f"{RED}Error: Selected index {shell_index} has b-value {vals[shell_index]} > threshold {threshold}.{RESET}")
                        return None
                else:
                    # Ambiguous case: 3D image but multiple bvals and no index. 
                    # Assuming the single volume corresponds to the first bval or 'a' bval is risky
                    # without more info, but commonly 3D means "this is the volume".
                    # Let's fallback to checking if the *first* value is b0 if no index.
                    if vals[0] > threshold:
                         print(f"{RED}Error: First b-value {vals[0]} > threshold {threshold}.{RESET}")
                         return None

            except Exception as e:
                print(f"{RED}Error reading bvals for 3D image: {str(e)}{RESET}")
                return None
        
        # If we got here, it's 3D and either valid b0 or no bvals provided (assumed b0/user knows)
        if output_path:
            shutil.copyfile(dwi_path, output_path)
            print(f"{GREEN}Copied 3D b0 volume to: {output_path}{RESET}")
            
        # Handle gradient files if requested (create single-entry files)
        # Handle bvals
        if bvals and b0_bval:
             with open(b0_bval, 'w') as f:
                 # Use the detected b-value or 0 if unknown
                 val = bvals[shell_index if shell_index is not None else 0] if bvals else 0
                 f.write(str(int(val)))
        
        # Handle bvecs
        if bvecs_path and b0_bvec:
             try:
                with open(bvecs_path, 'r') as f:
                    lines = f.readlines()
                
                # Default to zero vector
                b0_vec = [0.0, 0.0, 0.0]
                
                if len(lines) == 3:
                     # Parse bvecs (3 rows)
                     vecs = [[float(v) for v in l.strip().split()] for l in lines]
                     idx = shell_index if shell_index is not None else 0
                     
                     # Extract vector at index
                     current_vec = []
                     for dim_row in vecs:
                        if idx < len(dim_row):
                            current_vec.append(dim_row[idx])
                        else:
                            current_vec.append(0.0)
                     b0_vec = current_vec
                
                # Write b0 bvec file (3 rows, 1 col typically expected for single vol, or 3 lines)
                # Matches format used in main function: writes one value per line
                with open(b0_bvec, 'w') as f:
                    for v in b0_vec:
                        f.write(f"{v}\n")
                        
             except Exception as e:
                 print(f"{YELLOW}Warning: Error processing bvecs for 3D image: {e}{RESET}")

        # Handle 'non-b0' outputs
        # Since input is 3D b0, there are NO non-b0 volumes.
        # We create empty/placeholder files to satisfy pipeline requirements (Snakemake).
        if output_dwi:
            print(f"{YELLOW}Warning: No non-b0 volumes in 3D input. Creating empty placeholder for: {output_dwi}{RESET}")
            # Create empty file
            with open(output_dwi, 'w') as f:
                pass
            
        if output_bvals:
            with open(output_bvals, 'w') as f:
                pass

        if output_bvecs:
            with open(output_bvecs, 'w') as f:
                pass
            
        return

    if len(img_shape) < 3: # Should capture 1D/2D which are invalid
        print(f"{RED}Input image dimensions {img_shape} are invalid.{RESET}")
        return None
        
    num_volumes = img_shape[shell_dimension]
    print(f"  Number of volumes: {num_volumes}")
    bvals = None
    bvecs = None
    
    # Load bvals if provided
    if bvals_path:
        print(f"\n{CYAN}Loading b-values file...{RESET}")
        print(f"  File: {bvals_path}")
        try:
            with open(bvals_path, 'r') as f:
                bvals = [float(val) for val in f.read().strip().split()]
            print(f"  Number of b-values: {len(bvals)}")
            
            # Check if number of bvals matches the last dimension
            if len(bvals) != num_volumes:
                print(f"{YELLOW}Warning: Number of bvals ({len(bvals)}) doesn't match "
                      f"the number of volumes ({num_volumes}).{RESET}")
                if shell_index is None:
                    print(f"{RED}Cannot determine b0 index without consistent bvals. "
                          f"Please specify --index.{RESET}")
                    return None
        except Exception as e:
            print(f"{RED}Error loading bvals file: {e}{RESET}")
            if shell_index is None:
                print(f"{RED}Cannot determine b0 index without bvals. Please specify --index.{RESET}")
                return None
    
    # Load bvecs if provided
    if bvecs_path:
        print(f"\n{CYAN}Loading b-vectors file...{RESET}")
        print(f"  File: {bvecs_path}")
        try:
            with open(bvecs_path, 'r') as f:
                bvec_lines = f.readlines()
            
            # Parse bvecs (3 rows x N columns format)
            if len(bvec_lines) == 3:
                bvecs = []
                for line in bvec_lines:
                    row_values = [float(val) for val in line.strip().split()]
                    bvecs.append(row_values)
                
                print(f"  Number of b-vectors: {len(bvecs[0])}")
                
                # Check if bvecs dimensions match volumes
                if len(bvecs[0]) != num_volumes:
                    print(f"{YELLOW}Warning: Number of bvecs ({len(bvecs[0])}) doesn't match "
                          f"the number of volumes ({num_volumes}).{RESET}")
            else:
                print(f"{YELLOW}Warning: Expected 3 rows in bvecs file, got {len(bvec_lines)}.{RESET}")
        except Exception as e:
            print(f"{RED}Error loading bvecs file: {e}{RESET}")
    
    # Determine b0 index
    print(f"\n{CYAN}Determining b0 volume index...{RESET}")
    if shell_index is not None:
        # Validate shell_index
        if shell_index < 0:
            # Handle negative indexing
            shell_index = num_volumes + shell_index
        
        if shell_index < 0 or shell_index >= num_volumes:
            print(f"{YELLOW}Warning: Index {shell_index} out of range (0-{num_volumes-1}). "
                  f"Using volume 0.{RESET}")
            shell_index = 0
            
        print(f"  Using specified index: {shell_index}")
        if bvals:
            print(f"  B-value at index {shell_index}: {bvals[shell_index]:.1f} s/mm²")
    elif bvals:
        # Find the first b0 volume based on threshold
        b0_indices = [i for i, val in enumerate(bvals) if val <= threshold]
        if not b0_indices:
            print(f"{RED}No b-value below threshold {threshold} found in bvals file.{RESET}")
            print(f"{YELLOW}Minimum b-value: {min(bvals):.1f} s/mm²{RESET}")
            print(f"{YELLOW}Try increasing --threshold or use --index to specify manually.{RESET}")
            return None
        
        shell_index = b0_indices[0]
        print(f"  Found first b0 volume (b={bvals[shell_index]:.1f} s/mm²) at index {shell_index}")
        if len(b0_indices) > 1:
            print(f"  {MAGENTA}Note: {len(b0_indices)} b0 volumes found. "
                  f"Extracting first one (index {shell_index}).{RESET}")
            print(f"  {MAGENTA}Use --index to extract a different b0 volume.{RESET}")
    else:
        print(f"{RED}No index or bvals provided. Cannot determine which volume is b0.{RESET}")
        return None
    
    # Extract the b0 volume
    print(f"\n{CYAN}Extracting b0 volume at index {shell_index}...{RESET}")
    
    # Create a dynamic indexing tuple to select along the specified dimension
    idx = tuple(slice(None) if i != shell_dimension else shell_index 
               for i in range(len(img_data.shape)))
    
    # Extract the b0 volume along the specified dimension
    b0_data = img_data[idx]
    print(f"  B0 volume shape: {b0_data.shape}")
    
    # Save b0 volume if output path specified
    if output_path:
        b0_nib = nib.Nifti1Image(b0_data, img_affine, img_nib.header)
        nib.save(b0_nib, output_path)
        print(f"{GREEN}Saved b0 volume to: {output_path}{RESET}")
    
    # Extract and save non-b0 volumes if requested
    if output_dwi:
        print(f"\n{CYAN}Extracting non-b0 volumes...{RESET}")
        # Create list of indices without the b0 volume
        non_b0_indices = list(range(num_volumes))
        non_b0_indices.pop(shell_index)
        
        if not non_b0_indices:
            print(f"{YELLOW}Warning: No non-b0 volumes to extract (single volume input?){RESET}")
        else:
            # Extract non-b0 volumes
            non_b0_data = img_data[..., non_b0_indices]
            print(f"  Non-b0 shape: {non_b0_data.shape}")
            
            # Save non-b0 volumes
            non_b0_nib = nib.Nifti1Image(non_b0_data, img_affine, img_nib.header)
            nib.save(non_b0_nib, output_dwi)
            print(f"{GREEN}Saved non-b0 volumes to: {output_dwi}{RESET}")
    
    # Save b0-only bval if requested
    if bvals and b0_bval:
        print(f"\n{CYAN}Saving b0-only b-value file...{RESET}")
        # Extract only the b0 bval
        b0_bval_value = bvals[shell_index]
        
        # Write b0-only bval to file
        with open(b0_bval, 'w') as f:
            f.write(str(int(b0_bval_value)))
        
        print(f"{GREEN}Saved b0-only bval to: {b0_bval}{RESET}")
        print(f"  Value: {int(b0_bval_value)} s/mm²")
    
    # Save b0-only bvec if requested
    if bvecs and b0_bvec:
        print(f"\n{CYAN}Saving b0-only b-vector file...{RESET}")
        # Extract only the b0 bvec (one column from each of the 3 rows)
        b0_bvec_values = [direction[shell_index] for direction in bvecs]

        # Write b0-only bvec to file (3 rows, 1 column)
        with open(b0_bvec, 'w') as f:
            for val in b0_bvec_values:
                f.write(f"{val}\n")
        
        print(f"{GREEN}Saved b0-only bvec to: {b0_bvec}{RESET}")
        print(f"  Direction: [{b0_bvec_values[0]:.3f}, {b0_bvec_values[1]:.3f}, {b0_bvec_values[2]:.3f}]")
    
    # Update bvals and save if requested (non-b0 only)
    if bvals and output_bvals:
        print(f"\n{CYAN}Saving updated b-values file (non-b0 only)...{RESET}")
        # Remove b0 entry
        updated_bvals = [val for i, val in enumerate(bvals) if i != shell_index]
        
        # Write updated bvals to file
        with open(output_bvals, 'w') as f:
            f.write(' '.join([str(int(val)) for val in updated_bvals]))
        
        print(f"{GREEN}Saved updated bvals to: {output_bvals}{RESET}")
        print(f"  Number of values: {len(updated_bvals)}")
    
    # Update bvecs and save if requested (non-b0 only)
    if bvecs and output_bvecs:
        print(f"\n{CYAN}Saving updated b-vectors file (non-b0 only)...{RESET}")
        # Remove b0 entry from each direction
        updated_bvecs = []
        for direction in bvecs:
            updated_bvecs.append([val for i, val in enumerate(direction) if i != shell_index])
        
        # Write updated bvecs to file
        with open(output_bvecs, 'w') as f:
            for direction in updated_bvecs:
                f.write(' '.join([str(val) for val in direction]) + '\n')
        
        print(f"{GREEN}Saved updated bvecs to: {output_bvecs}{RESET}")
        print(f"  Number of vectors: {len(updated_bvecs[0])}")
    
    print(f"\n{GREEN}{BOLD}B0 extraction completed successfully!{RESET}")


if __name__ == "__main__":
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="Extract b=0 volume from DWI and update gradient files",
        add_help=False  # Use custom help
    )
    parser.add_argument("--input", required=True, help="Path to DWI image")
    parser.add_argument("--bvals", help="Path to bvals file")
    parser.add_argument("--bvecs", help="Path to bvecs file")
    parser.add_argument("--output", required=True, help="Path for extracted b0 volume")
    parser.add_argument("--output-dwi", help="Path for output non-b0 volumes")
    parser.add_argument("--output-bvals", help="Path for output bvals file (non-b0 only)")
    parser.add_argument("--output-bvecs", help="Path for output bvecs file (non-b0 only)")
    parser.add_argument("--threshold", type=float, default=50, 
                        help="Maximum b-value to consider as b0 (default: 50 s/mm²)")
    parser.add_argument("--index", type=int, 
                        help="Directly specify volume index to extract (overrides bvals)")
    parser.add_argument("--shell-dimension", type=int, default=3, 
                        help="Dimension of volume axis (default: 3, last dimension)")
    parser.add_argument("--b0-bval", help="Path for b0-only bval file")
    parser.add_argument("--b0-bvec", help="Path for b0-only bvec file")

    args = parser.parse_args()
    
    try:
        # Ensure either bvals or index is provided when needed
        if args.index is None and args.bvals is None:
            print(f"{RED}Error: Either --bvals or --index must be provided to identify the b0 volume{RESET}")
            sys.exit(1)
        
        # Ensure bvecs is provided if output-bvecs is requested
        if args.output_bvecs and not args.bvecs:
            print(f"{RED}Error: --bvecs must be provided when --output-bvecs is specified{RESET}")
            sys.exit(1)
        
        # Ensure bvals is provided if output-bvals is requested
        if args.output_bvals and not args.bvals:
            print(f"{RED}Error: --bvals must be provided when --output-bvals is specified{RESET}")
            sys.exit(1)
        
        # Ensure bvecs is provided if b0-bvec is requested
        if args.b0_bvec and not args.bvecs:
            print(f"{RED}Error: --bvecs must be provided when --b0-bvec is specified{RESET}")
            sys.exit(1)
        
        # Ensure bvals is provided if b0-bval is requested
        if args.b0_bval and not args.bvals:
            print(f"{RED}Error: --bvals must be provided when --b0-bval is specified{RESET}")
            sys.exit(1)
        
        # Extract the b0 volume and process other files
        extract_b0(
            args.input,
            args.bvals,
            args.bvecs,
            args.output,
            args.output_dwi,
            args.output_bvals,
            args.output_bvecs,
            args.threshold,
            args.index,
            args.shell_dimension,
            args.b0_bval,
            args.b0_bvec
        )
        
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\n{RED}{BOLD}File not found:{RESET}")
        print(f"  {str(e)}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n{RED}{BOLD}Error during b0 extraction:{RESET}")
        print(f"  {str(e)}")
        print(f"\n{YELLOW}Run 'micaflow extract_b0 --help' for usage information.{RESET}")
        sys.exit(1)
