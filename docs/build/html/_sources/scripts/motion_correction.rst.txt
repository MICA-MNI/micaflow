Motion Correction
=================

Command Line Usage
-----------------

.. code-block:: bash

    micaflow motion_correction [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow2.0/blob/main/micaflow/scripts/motion_correction.py>`_

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
