Motion Correction
=================

motion_correction - Diffusion MRI Motion Artifact Removal

Part of the micaflow processing pipeline for neuroimaging data.

This module corrects for subject motion in diffusion-weighted images (DWI) by registering
each volume to the first volume (typically a B0 image). Subject movement during 
acquisition is one of the primary sources of artifacts in diffusion MRI, causing 
misalignment between volumes that can severely impact analysis. This implementation uses
ANTs' SyNRA algorithm, which combines rigid, affine, and deformable transformations for 
robust inter-volume alignment.

Features:
--------
- Volume-by-volume registration to a reference B0 image
- Combines rigid, affine, and deformable transformations using ANTs SyNRA
- Preserves original image header information and coordinates
- Progress visualization with volume-wise completion tracking
- Compatible with standard diffusion acquisition protocols
- No gradient reorientation needed (performed at tensor fitting stage)

API Usage:
---------
micaflow motion_correction 
    --denoised <path/to/dwi.nii.gz>
    --bval <path/to/dwi.bval>
    --bvec <path/to/dwi.bvec>
    --output <path/to/motion_corrected_dwi.nii.gz>

Python Usage:
-----------
>>> from micaflow.scripts.motion_correction import run_motion_correction
>>> run_motion_correction(
...     dwi_path="denoised_dwi.nii.gz",
...     bval_path="dwi.bval",
...     bvec_path="dwi.bvec", 
...     output="motion_corrected_dwi.nii.gz"
... )

Command Line Usage
-----------------

.. code-block:: bash

    micaflow motion_correction [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow/blob/main/micaflow/scripts/motion_correction.py>`_

Description
-----------

This script corrects for subject motion in diffusion-weighted images (DWI)
    by registering each volume to the first volume (typically a B0 image).
    It uses ANTs SyNRA registration which combines rigid, affine, and
    deformable transformations.

Full Help
---------

.. code-block:: text

    
        ╔════════════════════════════════════════════════════════════════╗
        ║                     MOTION CORRECTION                          ║
        ╚════════════════════════════════════════════════════════════════╝
        
        This script corrects for subject motion in diffusion-weighted images (DWI)
        by registering each volume to the first volume (typically a B0 image).
        It uses ANTs SyNRA registration which combines rigid, affine, and
        deformable transformations.
        
        ────────────────────────── USAGE ──────────────────────────
          micaflow motion_correction [options]
        
        ─────────────────── REQUIRED ARGUMENTS ───────────────────
          --denoised   : Path to the input denoised DWI image (.nii.gz)
          --bval       : Path to the b-values file (.bval)
          --bvec       : Path to the b-vectors file (.bvec)
          --output     : Output path for the motion-corrected image (.nii.gz)
        
        ─────────────────── EXAMPLE USAGE ───────────────────
          micaflow motion_correction \
            --denoised denoised_dwi.nii.gz \
            --bval dwi.bval \
            --bvec dwi.bvec \
            --output motion_corrected_dwi.nii.gz
        
        ────────────────────────── NOTES ─────────────────────────
        - The first volume is assumed to be a B0 image and used as the reference
        - Each subsequent volume is registered to this reference
        - The process can take significant time depending on volume count
        - Progress is displayed using a progress bar
        
    


.. automodule:: micaflow.scripts.motion_correction
   :noindex:
