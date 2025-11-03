"""
synth_b0 - Synthetic Undistorted B0 Generation using Deep Learning

Part of the micaflow processing pipeline for neuroimaging data.

This module generates synthetic undistorted B0 images from T1-weighted anatomical 
images and distorted B0 images using the SynB0-DISCO deep learning framework. The 
approach uses an ensemble of 3D U-Net models trained to predict undistorted B0 
contrast from T1w images, enabling distortion correction even when opposite 
phase-encoding acquisitions are unavailable.

What is SynB0-DISCO?
-------------------
SynB0-DISCO (Synthetic B0 for DIStortion COrrection) is a deep learning method that:
- Learns the relationship between T1w anatomy and undistorted B0 contrast
- Generates synthetic undistorted B0 images without requiring field maps
- Uses multiple 3D U-Net models in an ensemble for robust predictions
- Enables distortion correction when only single phase-encoding data is available

Why Synthetic B0?
----------------
Traditional susceptibility distortion correction requires two acquisitions with 
opposite phase-encoding directions (e.g., AP and PA). However:
- Legacy datasets often lack reverse phase-encoding acquisitions
- Some protocols only acquire data in one direction
- Acquiring reverse PE images adds scan time
- SynB0-DISCO enables correction with just T1w + single PE B0

The Problem:
- EPI images (DWI, fMRI) suffer from susceptibility distortions
- Distortions cause misalignment with anatomical images
- Affected regions: orbitofrontal cortex, temporal poles, brainstem
- Traditional correction needs blip-up/blip-down pairs

The Solution:
- Train deep learning models on datasets WITH reverse PE
- Models learn to predict undistorted B0 from T1w images
- Apply to new data that only has single PE acquisition
- Generate synthetic undistorted B0 for distortion correction

Deep Learning Architecture:
--------------------------
- Model: 3D U-Net with skip connections
- Input channels: 2 (T1w + distorted B0)
- Output: 1 (synthetic undistorted B0)
- Training: Paired T1w and undistorted B0 images
- Ensemble: 5 models for robust prediction (reduces variance)
- Framework: PyTorch with GPU acceleration

How It Works:
1. Load T1w and distorted B0 images
2. Register B0 to T1w space (affine)
3. Run inference with ensemble of 5 models
4. Average predictions for robust result
5. Register synthetic B0 back to original B0 space
6. Compute distortion field (synthetic B0 → distorted B0)
7. Apply correction to all DWI volumes

Features:
--------
- Ensemble of 5 models for robust predictions
- GPU acceleration with PyTorch (CUDA support)
- ANTs-based registration for spatial alignment
- Restricted transformations along phase-encoding direction
- 4D DWI support with volume-by-volume correction
- Automatic temporary file cleanup
- Multi-threading support for CPU operations

When to Use:
-----------
- When reverse phase-encoding data is unavailable
- For legacy datasets with single PE direction
- When scan time doesn't allow reverse PE acquisition
- After motion correction (for DWI)
- Before registration to anatomical space

When NOT to Use:
---------------
- If you have reverse PE data (use SDC instead)
- For non-EPI sequences without distortion
- If T1w image has severe artifacts or pathology
- When anatomical-functional alignment is already good

Command-Line Usage:
------------------
# Basic usage with AP phase-encoding (default)
micaflow synth_b0 \\
    --t1 <path/to/T1w.nii.gz> \\
    --b0 <path/to/distorted_b0.nii.gz> \\
    --dwi <path/to/dwi_no_b0.nii.gz> \\
    --output <path/to/corrected_dwi.nii.gz> \\
    --intermediate <path/to/synthetic_b0_t1space.nii.gz> \\
    --warp <path/to/warp_field.nii.gz> \\
    --b0-to-T1-affine <path/to/b0_to_t1_affine.mat> \\
    --b0-to-T1-warp <path/to/b0_to_t1_warp.nii.gz>

# With LR phase-encoding
micaflow synth_b0 \\
    --t1 <path/to/T1w.nii.gz> \\
    --b0 <path/to/distorted_b0.nii.gz> \\
    --dwi <path/to/dwi_no_b0.nii.gz> \\
    --output <path/to/corrected_dwi.nii.gz> \\
    --intermediate <path/to/synthetic_b0_t1space.nii.gz> \\
    --warp <path/to/warp_field.nii.gz> \\
    --phase-encoding lr \\
    --b0-to-T1-affine <path/to/b0_to_t1_affine.mat> \\
    --b0-to-T1-warp <path/to/b0_to_t1_warp.nii.gz>

# Force CPU usage (no GPU)
micaflow synth_b0 \\
    --t1 <path/to/T1w.nii.gz> \\
    --b0 <path/to/distorted_b0.nii.gz> \\
    --dwi <path/to/dwi_no_b0.nii.gz> \\
    --output <path/to/corrected_dwi.nii.gz> \\
    --intermediate <path/to/synthetic_b0_t1space.nii.gz> \\
    --warp <path/to/warp_field.nii.gz> \\
    --cpu \\
    --b0-to-T1-affine <path/to/b0_to_t1_affine.mat> \\
    --b0-to-T1-warp <path/to/b0_to_t1_warp.nii.gz>

# With custom thread count
micaflow synth_b0 \\
    --t1 <path/to/T1w.nii.gz> \\
    --b0 <path/to/distorted_b0.nii.gz> \\
    --dwi <path/to/dwi_no_b0.nii.gz> \\
    --output <path/to/corrected_dwi.nii.gz> \\
    --intermediate <path/to/synthetic_b0_t1space.nii.gz> \\
    --warp <path/to/warp_field.nii.gz> \\
    --threads 8 \\
    --b0-to-T1-affine <path/to/b0_to_t1_affine.mat> \\
    --b0-to-T1-warp <path/to/b0_to_t1_warp.nii.gz>

# Save corrected B0 for QC
micaflow synth_b0 \\
    --t1 <path/to/T1w.nii.gz> \\
    --b0 <path/to/distorted_b0.nii.gz> \\
    --dwi <path/to/dwi_no_b0.nii.gz> \\
    --output <path/to/corrected_dwi.nii.gz> \\
    --intermediate <path/to/synthetic_b0_t1space.nii.gz> \\
    --warp <path/to/warp_field.nii.gz> \\
    --corrected-b0 <path/to/corrected_b0.nii.gz> \\
    --b0-to-T1-affine <path/to/b0_to_t1_affine.mat> \\
    --b0-to-T1-warp <path/to/b0_to_t1_warp.nii.gz>

Python API Usage:
----------------
>>> from micaflow.scripts.synth_b0 import main
>>> import sys
>>> 
>>> # Set up arguments
>>> sys.argv = [
...     'synth_b0',
...     '--t1', 'T1w.nii.gz',
...     '--b0', 'distorted_b0.nii.gz',
...     '--dwi', 'dwi_no_b0.nii.gz',
...     '--output', 'corrected_dwi.nii.gz',
...     '--intermediate', 'synthetic_b0_t1space.nii.gz',
...     '--warp', 'warp_field.nii.gz',
...     '--b0-to-T1-affine', 'b0_to_t1_affine.mat',
...     '--b0-to-T1-warp', 'b0_to_t1_warp.nii.gz'
... ]
>>> 
>>> # Run
>>> main()

Pipeline Integration:
--------------------
SynB0-DISCO is an alternative to traditional SDC when reverse PE data unavailable:

Diffusion MRI Pipeline (with reverse PE):
1. Denoising (denoise)
2. Motion correction (motion_correction)
3. Susceptibility correction (SDC) 
4. Registration (coregister)
5. DTI metrics (compute_fa_md)

Diffusion MRI Pipeline (without reverse PE):
1. Denoising (denoise)
2. Motion correction (motion_correction)
3. B0 extraction (extract_b0)
4. Synthetic B0 generation (synth_b0) 
5. Registration (coregister)
6. DTI metrics (compute_fa_md)

Exit Codes:
----------
0 : Success - synthetic B0 generation completed
1 : Error - invalid inputs, file not found, or processing failure

Technical Notes:
---------------
- Model architecture: 3D U-Net with 2 input channels, 1 output channel
- Ensemble size: 5 models (or fewer if models unavailable)
- Inference: Average of all model predictions
- Registration: ANTs affine for initial alignment
- Distortion field: ANTs SyN with restricted transformation
- Phase-encoding restriction: Only allows deformation along PE axis
- Multi-threading: Supports both PyTorch and ANTs threading
- Temporary files: Automatically cleaned up after completion

Registration Parameters:
-----------------------
Initial alignment (B0 → T1):
- Type: Affine (12 DOF)
- Metric: Mattes mutual information

Distortion field computation:
- Type: SyN (symmetric normalization)
- Iterations: (20, 10, 5) - reduced for gentler warps
- Gradient step: 0.1 (smaller than default 0.2)
- Flow sigma: 3 (smoothing of velocity field)
- Total sigma: 1 (smoothing of total deformation)
- Restriction: Along phase-encoding direction only

Phase-Encoding Directions:
-------------------------
- AP (Anterior-Posterior): y-axis restriction (0,1,0)
- PA (Posterior-Anterior): y-axis restriction (0,1,0)
- LR (Left-Right): x-axis restriction (1,0,0)
- RL (Right-Left): x-axis restriction (1,0,0)
- SI (Superior-Inferior): z-axis restriction (0,0,1)
- IS (Inferior-Superior): z-axis restriction (0,0,1)


Quality Control:
---------------
Visual inspection recommended:
1. Check alignment of synthetic B0 with T1w
2. Verify distortion field is smooth and localized
3. Compare corrected B0 with synthetic B0
4. Assess DWI alignment with anatomical after correction
5. Look for residual distortions in affected regions

Limitations:
-----------
- Requires high-quality T1w image
- May not generalize to severe pathology
- Less accurate than field map-based correction
- Assumes typical brain anatomy

See Also:
--------
- SDC : Traditional distortion correction with reverse PE
- extract_b0 : Extract B0 volumes from DWI
- motion_correction : Apply before synthetic B0
- coregister : Register corrected images to anatomical

References:
----------
1. Schilling KG, Blaber J, Huo Y, et al. Synthesized b0 for diffusion distortion 
   correction (Synb0-DisCo). Magn Reson Imaging. 2019;64:62-70. 
   doi:10.1016/j.mri.2019.05.008

"""

import argparse
import sys
import os
import torch
import nibabel as nib
import numpy as np
import ants
import shutil
import subprocess
from colorama import Fore, Style, init
import glob
import tempfile
from .synb0_DISCO.inference import inference
from .synb0_DISCO.model import UNet3D
from .synb0_DISCO.util import torch2nii
from lamareg import lamareg
from lamareg.scripts import apply_warp
from micaflow.scripts.apply_warp import apply_warp
from micaflow.scripts.coregister import coregister

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
    ║                  SYNTHETIC B0 GENERATION                       ║
    ║                      (SynB0-DISCO)                            ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script generates a synthetic undistorted B0 image from T1w and 
    distorted B0 images using deep learning (SynB0-DISCO). The process uses 
    an ensemble of 3D U-Net models to predict undistorted B0 contrast, 
    enabling distortion correction without reverse phase-encoding acquisitions.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow synth_b0 {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--t1{RESET}                : Path to T1w image (.nii.gz)
      {YELLOW}--b0{RESET}                : Path to distorted B0 image (.nii.gz)
      {YELLOW}--dwi{RESET}               : Path to DWI image without B0 volumes (.nii.gz)
      {YELLOW}--output{RESET}            : Path for corrected DWI output (.nii.gz)
      {YELLOW}--intermediate{RESET}      : Path to save synthetic B0 in T1 space (.nii.gz)
      {YELLOW}--warp{RESET}              : Path to save the warp field (.nii.gz)
      {YELLOW}--b0-to-T1-affine{RESET}   : Path to B0→T1 affine transform (.mat)
      {YELLOW}--b0-to-T1-warp{RESET}     : Path to B0→T1 warp field (.nii.gz)
    
    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--phase-encoding{RESET}           : Phase-encoding direction
                                   {MAGENTA}Choices: 'ap', 'pa', 'lr', 'rl', 'si', 'is'{RESET}
                                   {MAGENTA}Default: 'ap'{RESET}
      {YELLOW}--cpu{RESET}                      : Force CPU usage (default: use GPU)
      {YELLOW}--threads{RESET}                  : Number of CPU threads (default: all)
      {YELLOW}--temp-dir{RESET}                 : Directory for temporary files
      {YELLOW}--corrected-b0{RESET}             : Path to save corrected B0 (for QC)
      {YELLOW}--shell-dimension{RESET}          : Volume dimension in 4D (default: 3)
      {YELLOW}--b0-to-T1-warp-secondary{RESET}  : Secondary warp field (optional)
    
    {CYAN}{BOLD}──────────────────── EXAMPLE USAGE ───────────────────────{RESET}
    
    {BLUE}# Example 1: Basic usage with AP phase-encoding{RESET}
    micaflow synth_b0 \\
      {YELLOW}--t1{RESET} T1w.nii.gz \\
      {YELLOW}--b0{RESET} distorted_b0.nii.gz \\
      {YELLOW}--dwi{RESET} dwi_no_b0.nii.gz \\
      {YELLOW}--output{RESET} corrected_dwi.nii.gz \\
      {YELLOW}--intermediate{RESET} synthetic_b0_t1space.nii.gz \\
      {YELLOW}--warp{RESET} warp_field.nii.gz \\
      {YELLOW}--b0-to-T1-affine{RESET} b0_to_t1.mat \\
      {YELLOW}--b0-to-T1-warp{RESET} b0_to_t1_warp.nii.gz
    
    {BLUE}# Example 2: With LR phase-encoding{RESET}
    micaflow synth_b0 \\
      {YELLOW}--t1{RESET} T1w.nii.gz \\
      {YELLOW}--b0{RESET} distorted_b0.nii.gz \\
      {YELLOW}--dwi{RESET} dwi_no_b0.nii.gz \\
      {YELLOW}--output{RESET} corrected_dwi.nii.gz \\
      {YELLOW}--intermediate{RESET} synthetic_b0_t1space.nii.gz \\
      {YELLOW}--warp{RESET} warp_field.nii.gz \\
      {YELLOW}--phase-encoding{RESET} lr \\
      {YELLOW}--b0-to-T1-affine{RESET} b0_to_t1.mat \\
      {YELLOW}--b0-to-T1-warp{RESET} b0_to_t1_warp.nii.gz
    
    {BLUE}# Example 3: CPU only, save corrected B0 for QC{RESET}
    micaflow synth_b0 \\
      {YELLOW}--t1{RESET} T1w.nii.gz \\
      {YELLOW}--b0{RESET} distorted_b0.nii.gz \\
      {YELLOW}--dwi{RESET} dwi_no_b0.nii.gz \\
      {YELLOW}--output{RESET} corrected_dwi.nii.gz \\
      {YELLOW}--intermediate{RESET} synthetic_b0_t1space.nii.gz \\
      {YELLOW}--warp{RESET} warp_field.nii.gz \\
      {YELLOW}--cpu{RESET} \\
      {YELLOW}--corrected-b0{RESET} corrected_b0.nii.gz \\
      {YELLOW}--threads{RESET} 8 \\
      {YELLOW}--b0-to-T1-affine{RESET} b0_to_t1.mat \\
      {YELLOW}--b0-to-T1-warp{RESET} b0_to_t1_warp.nii.gz
    
    {CYAN}{BOLD}────────────── WHY SYNTHETIC B0? ───────────────────────{RESET}
    
    {GREEN}The Problem:{RESET}
    {MAGENTA}•{RESET} EPI images suffer from susceptibility distortions
    {MAGENTA}•{RESET} Traditional correction needs reverse phase-encoding scans
    {MAGENTA}•{RESET} Legacy datasets often lack reverse PE acquisitions
    {MAGENTA}•{RESET} Some protocols only acquire single PE direction
    
    {GREEN}The Solution:{RESET}
    {MAGENTA}•{RESET} Deep learning predicts undistorted B0 from T1w + distorted B0
    {MAGENTA}•{RESET} No reverse PE acquisition needed
    {MAGENTA}•{RESET} Works with existing datasets
    {MAGENTA}•{RESET} Enables distortion correction for legacy data
    
    {CYAN}{BOLD}──────────────── SynB0-DISCO METHOD ────────────────────{RESET}
    
    {GREEN}Deep Learning Architecture:{RESET}
    {MAGENTA}•{RESET} 3D U-Net with skip connections
    {MAGENTA}•{RESET} Input: T1w + distorted B0 (2 channels)
    {MAGENTA}•{RESET} Output: Synthetic undistorted B0 (1 channel)
    {MAGENTA}•{RESET} Ensemble: 5 models averaged for robust prediction
    
    {GREEN}How It Works:{RESET}
    {MAGENTA}1.{RESET} Register B0 to T1 space (affine)
    {MAGENTA}2.{RESET} Run inference with 5 models
    {MAGENTA}3.{RESET} Average predictions
    {MAGENTA}4.{RESET} Register synthetic B0 to original space
    {MAGENTA}5.{RESET} Compute distortion field (restricted to PE direction)
    {MAGENTA}6.{RESET} Apply correction to all DWI volumes
    
    {CYAN}{BOLD}────────────────────── WHEN TO USE ──────────────────────{RESET}
    
    {GREEN}Recommended:{RESET}
    {MAGENTA}•{RESET} Reverse phase-encoding data unavailable
    {MAGENTA}•{RESET} Legacy datasets with single PE direction
    {MAGENTA}•{RESET} Scan time doesn't allow reverse PE
    {MAGENTA}•{RESET} After motion correction (for DWI)
    {MAGENTA}•{RESET} Before registration to anatomical space
    
    {RED}Not recommended:{RESET}
    {MAGENTA}•{RESET} If reverse PE data available (use SDC instead)
    {MAGENTA}•{RESET} T1w has severe artifacts or pathology
    {MAGENTA}•{RESET} Non-EPI sequences without distortion
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    {MAGENTA}•{RESET} Model ensemble: Up to 5 models (uses all available)
    {MAGENTA}•{RESET} Requires high-quality T1w image
    {MAGENTA}•{RESET} Distortion field restricted to PE direction only
    {MAGENTA}•{RESET} Supports 4D DWI with multiple volumes
    {MAGENTA}•{RESET} Automatic temporary file cleanup
    {MAGENTA}•{RESET} Multi-threading supported for CPU operations
    
    {CYAN}{BOLD}─────────────── PIPELINE POSITION ───────────────────────{RESET}
    {BLUE}With reverse PE (use traditional SDC):{RESET}
    1. Denoising
    2. Motion correction
    3. Susceptibility correction (SDC)
    
    {BLUE}Without reverse PE (use SynB0-DISCO):{RESET}
    1. Denoising
    2. Motion correction
    3. B0 extraction
    {GREEN}4. Synthetic B0 generation (synth_b0){RESET} {MAGENTA}← You are here{RESET}
    5. Registration
    
    {CYAN}{BOLD}────────────────────── EXIT CODES ───────────────────────{RESET}
    {GREEN}0{RESET} : Success - synthetic B0 generation completed
    {RED}1{RESET} : Error - invalid inputs, file not found, or processing failure
    
    {CYAN}{BOLD}───────────────── COMMON ISSUES ─────────────────────────{RESET}
    {YELLOW}Issue:{RESET} Poor correction quality
    {GREEN}Solution:{RESET} Check T1w quality, verify phase-encoding direction
    
    {YELLOW}Issue:{RESET} Out of memory error
    {GREEN}Solution:{RESET} Use CPU mode, reduce number of threads
    
    {YELLOW}Issue:{RESET} Very slow processing
    {GREEN}Solution:{RESET} Use GPU if available (10-30x speedup)
    
    {YELLOW}Issue:{RESET} Models not found
    {GREEN}Solution:{RESET} Check micaflow installation, verify models directory
    """
    print(help_text)


def get_restrict_transformation(phase_encoding):
    """
    Get the restriction vector for ANTs registration based on phase-encoding direction.
    
    This function maps phase-encoding direction codes to restriction vectors that
    constrain deformable registration to only allow transformations along the 
    phase-encoding axis. This is critical for susceptibility distortion correction,
    as distortions primarily occur along the PE direction.
    
    Parameters
    ----------
    phase_encoding : str
        Phase-encoding direction code:
        - 'lr' or 'rl': Left-Right (x-axis)
        - 'ap' or 'pa': Anterior-Posterior (y-axis)
        - 'si' or 'is': Superior-Inferior (z-axis)
        
    Returns
    -------
    tuple of int
        A 3-element tuple (x, y, z) where:
        - 1 indicates transformation is allowed along that axis
        - 0 indicates transformation is restricted (not allowed)
        
    Notes
    -----
    - Most DWI acquisitions use AP/PA encoding (y-axis)
    - Restricting transformation prevents unrealistic warps
    - Only the PE direction should be allowed to deform
    - Case-insensitive: 'AP', 'ap', 'Ap' all work
    - Default: (0,1,0) for unknown directions
    
    Examples
    --------
    >>> # AP/PA encoding (most common)
    >>> restrict = get_restrict_transformation('ap')
    >>> print(restrict)
    (0, 1, 0)
    >>> 
    >>> # LR/RL encoding
    >>> restrict = get_restrict_transformation('lr')
    >>> print(restrict)
    (1, 0, 0)
    >>> 
    >>> # SI/IS encoding
    >>> restrict = get_restrict_transformation('si')
    >>> print(restrict)
    (0, 0, 1)
    """
    # Convert to lowercase and normalize direction
    pe = phase_encoding.lower()
    
    # Set restrictions based on phase-encoding direction
    if pe in ['lr', 'rl']:  # Left-Right or Right-Left (x-axis)
        return (1, 0, 0)
    elif pe in ['ap', 'pa']:  # Anterior-Posterior or Posterior-Anterior (y-axis)
        return (0, 1, 0)
    elif pe in ['si', 'is']:  # Superior-Inferior or Inferior-Superior (z-axis)
        return (0, 0, 1)
    else:
        print(f"{YELLOW}Warning: Unknown phase-encoding '{phase_encoding}', using default (0,1,0){RESET}")
        return (0, 1, 0)  # Default to y-axis (AP/PA)



def main():
    """
    Generate synthetic undistorted B0 and correct DWI distortion using SynB0-DISCO.
    
    This function implements the complete SynB0-DISCO pipeline:
    1. Loads T1w and distorted B0 images
    2. Runs inference with an ensemble of 3D U-Net models
    3. Generates synthetic undistorted B0 image
    4. Computes distortion field between synthetic and distorted B0
    5. Applies correction to all DWI volumes
    
    The approach uses deep learning to predict undistorted B0 contrast from T1w
    images, enabling distortion correction without reverse phase-encoding acquisitions.
    
    Command-Line Arguments
    ----------------------
    Required:
        --t1 : Path to T1w image
        --b0 : Path to distorted B0 image
        --dwi : Path to DWI without B0 volumes
        --output : Path for corrected DWI output
        --intermediate : Path to save synthetic B0 in T1 space
        --warp : Path to save warp field
        --b0-to-T1-affine : Path to B0→T1 affine transform
        --b0-to-T1-warp : Path to B0→T1 warp field
    
    Optional:
        --phase-encoding : PE direction (default: 'ap')
        --cpu : Force CPU usage
        --threads : Number of CPU threads
        --temp-dir : Temporary file directory
        --corrected-b0 : Path to save corrected B0
        --shell-dimension : Volume dimension (default: 3)
        --b0-to-T1-warp-secondary : Secondary warp field
    
    Returns
    -------
    None
        Writes outputs to specified file paths.
        
    Raises
    ------
    FileNotFoundError
        If required input files cannot be found.
    RuntimeError
        If model loading or inference fails.
    ValueError
        If invalid parameters provided.
        
    Notes
    -----
    - GPU highly recommended (10-30x faster than CPU)
    - Processing time: 5-15 min (GPU), 30-90 min (CPU)
    - Uses ensemble of up to 5 models (averages predictions)
    - Distortion field restricted to PE direction only
    - Automatically cleans up temporary files
    - Multi-threading supported for CPU operations
    - Progress messages color-coded for clarity
    
    Examples
    --------
    # See print_help_message() for detailed usage examples
    """
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)
        
    parser = argparse.ArgumentParser(
        description="Generate synthetic B0 and correct DWI distortion using SynB0-DISCO.",
        add_help=False  # Use custom help
    )
    parser.add_argument('--t1', required=True, help='Path to T1w input image (.nii.gz)')
    parser.add_argument('--b0', required=True, help='Path to distorted B0 image (.nii.gz)')
    parser.add_argument('--dwi', required=True, help='Path to DWI image without B0 volumes (.nii.gz)')
    parser.add_argument('--output', required=True, help='Path for corrected DWI output image (.nii.gz)')
    parser.add_argument('--intermediate', required=True, help='Path to save the synthetic B0 before inverse transform')
    parser.add_argument('--phase-encoding', default='ap', 
                        choices=['ap', 'pa', 'lr', 'rl', 'si', 'is'],
                        help='Phase-encoding direction (default: ap)')
    parser.add_argument('--warp', required=True, help='Path to save the warp field')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage (default: use GPU if available)')
    parser.add_argument('--temp-dir', help='Directory for temporary files (default: current directory)')
    parser.add_argument('--corrected-b0', help='Path to save the corrected B0 image (optional)')
    parser.add_argument('--threads', type=int, help='Number of threads for ANTs (default: all)')
    parser.add_argument('--synthseg-threads', type=int, help='Number of threads for SynthSeg (default: all)')
    parser.add_argument('--shell-dimension', type=int, default=3,
                        help='Shell dimension for DWI, default=3')
    parser.add_argument('--b0-to-T1-affine', required=True, help='Path to the B0 to T1 affine matrix')
    parser.add_argument('--b0-to-T1-warp', required=True, help='Path to the B0 to T1 warp field')
    parser.add_argument('--b0-to-T1-warp-secondary', help='Path to secondary warp field for B0 to T1')

    args = parser.parse_args()
    
    try:
        
        # Validate required files exist
        for filepath, name in [(args.t1, "T1w"), (args.b0, "B0"), (args.dwi, "DWI"),
                              (args.b0_to_T1_affine, "B0→T1 affine"), 
                              (args.b0_to_T1_warp, "B0→T1 warp")]:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"{name} file not found: {filepath}")
        
        shell_dim = args.shell_dimension
        
        # Define all file paths
        t1_path = args.t1
        b0_path = args.b0
        dwi_path = args.dwi
        
        print(f"{CYAN}Input files:{RESET}")
        print(f"  T1w: {t1_path}")
        print(f"  B0: {b0_path}")
        print(f"  DWI: {dwi_path}")
        
        # Load DWI and extract first volume
        print(f"\n{CYAN}Loading DWI image...{RESET}")
        dwi_image = ants.image_read(dwi_path)
        dwi_data = dwi_image.numpy()
        print(f"  Shape: {dwi_data.shape}")
        
        first_vol_idx = tuple(slice(None) if i != shell_dim else 0 for i in range(len(dwi_data.shape)))
        first_vol_data = dwi_data[first_vol_idx]
        first_dwi_image = ants.from_numpy(
            first_vol_data,
            origin=dwi_image.origin[:3],
            spacing=dwi_image.spacing[:3],
            direction=dwi_image.direction[:3, :3]
        )
        print(f"  Extracted first volume as reference")
        
        output = args.output
        intermediate_output = args.intermediate 
        
        # Get restriction vector
        restrict_transform = get_restrict_transformation(args.phase_encoding)
        print(f"\n{CYAN}Phase-encoding configuration:{RESET}")
        print(f"  Direction: {args.phase_encoding.upper()}")
        print(f"  Restriction vector: {restrict_transform}")
        
        # Create temp directory
        temp_dir = args.temp_dir if args.temp_dir else tempfile.mkdtemp()
        os.makedirs(temp_dir, exist_ok=True)
        print(f"\n{CYAN}Temporary directory:{RESET} {temp_dir}")
        
        # Find models
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        print(f"{CYAN}Models directory:{RESET} {models_dir}")
        
        # Set thread count
        num_threads = args.threads if args.threads is not None else os.cpu_count()
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(num_threads)
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        torch.backends.mkldnn.enabled = True
        print(f"\n{CYAN}Threading configuration:{RESET}")
        print(f"  CPU threads: {num_threads}")
        
        # Get device
        if args.cpu:
            device = torch.device("cpu")
            print(f"  Device: CPU (forced)")
        else:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"  Device: GPU - {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                print(f"  Device: CPU (no CUDA available)")
        
        # Find and load models
        print(f"\n{CYAN}Loading models...{RESET}")
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt') or f.endswith('.pth')]
        model_count = min(5, len(model_files))
        
        if model_count == 0:
            raise FileNotFoundError(f"No model files found in {models_dir}")
        
        if model_count < 5:
            print(f"  {YELLOW}Warning: Only {model_count} models found{RESET}")
        else:
            print(f"  Found {model_count} models")
        
        model_files = model_files[:model_count]
        model_paths = [os.path.join(models_dir, f) for f in model_files]

        # Run inference with ensemble
        print(f"\n{CYAN}Running inference with model ensemble...{RESET}")
        all_predictions = []

        for i, model_path in enumerate(model_paths):
            print(f"\n  {CYAN}Model {i+1}/{model_count}: {os.path.basename(model_path)}{RESET}")
            
            try:
                model = UNet3D(2, 1).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                print(f"    {GREEN}Model loaded{RESET}")
            except Exception as e:
                raise RuntimeError(f"Error loading model: {e}")

            try:
                with torch.no_grad():
                    img_model = inference(t1_path, b0_path, model, device)
                prediction = torch2nii(img_model.detach().cpu())
                all_predictions.append(prediction)
                print(f"    {GREEN}Inference completed{RESET}")
            except Exception as e:
                raise RuntimeError(f"Error during inference: {e}")
        
        # Average predictions
        print(f"\n{CYAN}Combining predictions from all models...{RESET}")
        combined_prediction = np.mean(all_predictions, axis=0)
        print(f"  {GREEN}Ensemble average computed{RESET}")
        
        # Save intermediate result
        nii_template = nib.load(t1_path)
        intermediate_nii = nib.Nifti1Image(combined_prediction, 
                                            nii_template.affine, nii_template.header)
        nib.save(intermediate_nii, intermediate_output)
        print(f"  {GREEN}Saved to: {intermediate_output}{RESET}")
        
        # Register synthetic B0
        print(f"\n{CYAN}Registering synthetic B0...{RESET}")
        synthetic_b0 = ants.image_read(intermediate_output)
        b0_img = ants.image_read(b0_path)
        synthetic_b0_registration = ants.registration(
            fixed=b0_img,
            moving=synthetic_b0,
            type_of_transform='Affine'
        )
        synthetic_b0_in_T1space = synthetic_b0_registration['warpedmovout']
        ants.image_write(synthetic_b0_in_T1space, intermediate_output)
        print(f"  {GREEN}Registration completed{RESET}")

        # Compute distortion field
        print(f"\n{CYAN}Computing distortion field...{RESET}")
        print(f"  Registration type: SyN (restricted to PE direction)")
        print(f"  Iterations: (20, 10, 5)")
        
        transforms = ants.registration(
            fixed=synthetic_b0_in_T1space,
            moving=b0_img,
            type_of_transform='SyNOnly',
            reg_iterations=(20, 10, 5),
            restrict_transformation=restrict_transform,
            # grad_step=0.1,
            # flow_sigma=3,
            # total_sigma=1,
            initial_transform='Identity',
        )
        print(f"  {GREEN}Distortion field computed{RESET}")

        # Save corrected B0 if requested
        if args.corrected_b0:
            print(f"\n{CYAN}Saving corrected B0 for quality control...{RESET}")
            b0_in_DWI = ants.apply_transforms(
                fixed=first_dwi_image, 
                moving=transforms['warpedmovout'], 
                transformlist=[args.b0_to_T1_affine], 
                whichtoinvert=[True],
                interpolator='bSpline'
            )
            ants.image_write(b0_in_DWI, args.corrected_b0)
            print(f"  {GREEN}Saved to: {args.corrected_b0}{RESET}")

        # Apply correction to DWI
        print(f"\n{CYAN}Applying correction to DWI volumes...{RESET}")
        dwi_reference = nib.load(dwi_path)
        num_volumes = dwi_reference.shape[shell_dim]
        print(f"  Processing {num_volumes} volumes")
        
        # Create volume processing directory
        volume_temp_dir = os.path.join(temp_dir, "volume_processing")
        os.makedirs(volume_temp_dir, exist_ok=True)
        
        dwi_data = dwi_reference.get_fdata()
        corrected_volumes = []
        
        # Process each volume
        for vol_idx in range(num_volumes):
            print(f"  Volume {vol_idx+1}/{num_volumes}...", end=' ')
            
            vol_idx_tuple = tuple(slice(None) if i != shell_dim else vol_idx 
                                for i in range(len(dwi_data.shape)))
            vol_data = dwi_data[vol_idx_tuple]
            vol_path = os.path.join(volume_temp_dir, f"vol_{vol_idx}.nii.gz")
            
            vol_img = nib.Nifti1Image(vol_data, dwi_reference.affine)
            nib.save(vol_img, vol_path)
            
            vol_ants = ants.image_read(vol_path)
            transformsformlist = [transforms['fwdtransforms'][0]]
            if args.b0_to_T1_warp_secondary:
                transformsformlist.append(args.b0_to_T1_warp_secondary)
            transformsformlist.append(args.b0_to_T1_warp)
            transformsformlist.append(args.b0_to_T1_affine)
            
            corrected_vol_T1space = ants.apply_transforms(
                fixed=synthetic_b0_in_T1space,
                moving=vol_ants,
                transformlist=transformsformlist,
                interpolator='bSpline'
            )

            corrected_vol = ants.apply_transforms(
                fixed=first_dwi_image,
                moving=corrected_vol_T1space,
                transformlist=[args.b0_to_T1_affine],
                whichtoinvert=[True],
                interpolator='bSpline'
            )
            corrected_volumes.append(corrected_vol.numpy())
        
        # Save corrected DWI
        print(f"\n{CYAN}Saving corrected DWI...{RESET}")
        corrected_4d = np.stack(corrected_volumes, axis=shell_dim)
        output_nii = nib.Nifti1Image(corrected_4d, dwi_reference.affine, dwi_reference.header)
        nib.save(output_nii, output)
        print(f"  {GREEN}Saved to: {output}{RESET}")

        # Save warp field
        shutil.copy(transforms['fwdtransforms'][0], args.warp)
        print(f"\n{CYAN}Warp field saved to:{RESET} {args.warp}")
        
        
        if not args.temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"  {GREEN}Removed temporary directory{RESET}")
            except Exception as e:
                print(f"  {YELLOW}Warning: Could not remove temporary directory: {e}{RESET}")
        
        print(f"\n{GREEN}{BOLD}Synthetic B0 generation and DWI distortion correction complete!{RESET}")
        print(f"  Corrected DWI: {output}")
        print(f"  Synthetic B0: {intermediate_output}")
        print(f"  Warp field: {args.warp}")
        if args.corrected_b0:
            print(f"  Corrected B0: {args.corrected_b0}")
        print()
        
        # Clean up transform files from registrations
        if 'fwdtransforms' in synthetic_b0_registration:
            for transform_file in synthetic_b0_registration['fwdtransforms']:
                if os.path.exists(transform_file):
                    try:
                        os.remove(transform_file)
                    except Exception as e:
                        print(f"{YELLOW}Warning: Could not remove {transform_file}: {e}{RESET}")
        
        if 'invtransforms' in synthetic_b0_registration:
            for transform_file in synthetic_b0_registration['invtransforms']:
                if os.path.exists(transform_file):
                    try:
                        os.remove(transform_file)
                    except Exception as e:
                        print(f"{YELLOW}Warning: Could not remove {transform_file}: {e}{RESET}")

        if 'fwdtransforms' in transforms:
            for transform_file in transforms['fwdtransforms']:
                if os.path.exists(transform_file):
                    try:
                        os.remove(transform_file)
                    except Exception as e:
                        print(f"{YELLOW}Warning: Could not remove {transform_file}: {e}{RESET}")
        
        if 'invtransforms' in transforms:
            for transform_file in transforms['invtransforms']:
                if os.path.exists(transform_file):
                    try:
                        os.remove(transform_file)
                    except Exception as e:
                        print(f"{YELLOW}Warning: Could not remove {transform_file}: {e}{RESET}")
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\n{RED}{BOLD}File not found:{RESET}")
        print(f"  {str(e)}")
        sys.exit(1)
        
    except RuntimeError as e:
        print(f"\n{RED}{BOLD}Runtime error:{RESET}")
        print(f"  {str(e)}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n{RED}{BOLD}Error during synthetic B0 generation:{RESET}")
        print(f"  {str(e)}")
        print(f"\n{YELLOW}Run 'micaflow synth_b0 --help' for usage information.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()