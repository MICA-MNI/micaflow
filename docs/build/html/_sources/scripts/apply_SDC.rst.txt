Apply Sdc
=========

apply_SDC - Susceptibility Distortion Correction for diffusion MRI

Part of the micaflow processing pipeline for neuroimaging data.

This module applies susceptibility distortion correction (SDC) to diffusion MRI images
by using a pre-calculated displacement field to unwarp geometric distortions caused by
magnetic field inhomogeneities. These distortions typically occur along the phase-encoding
direction (usually the y-axis).

The module works by:
1. Loading a distorted diffusion image (typically after motion correction)
2. Applying a voxel-wise displacement field to each volume in the 4D image
3. Using linear interpolation to resample the image at the corrected coordinates
4. Saving the unwarped image with the original affine transformation

API Usage:
---------
micaflow apply_SDC 
    --input <path/to/distorted_image.nii.gz>
    --warp <path/to/field_map.nii.gz>
    --affine <path/to/reference_image.nii.gz>
    --output <path/to/corrected_output.nii.gz>

Python Usage:
-----------
>>> from micaflow.scripts.apply_SDC import apply_SD_correction
>>> apply_SD_correction(
...     motion_corr_path="distorted_image.nii.gz",
...     warp_field=warp_field_array,
...     moving_affine=affine_matrix,
...     output="corrected_output.nii.gz"
... )

Command Line Usage
-----------------

.. code-block:: bash

    micaflow apply_SDC [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow/blob/main/micaflow/scripts/apply_SDC.py>`_

Description
-----------

This script applies susceptibility distortion correction to diffusion images
    using a pre-calculated warp field. It takes a motion-corrected diffusion image
    and applies the warp field to each 3D volume along the y-axis.

Full Help
---------

.. code-block:: text

    
        ╔════════════════════════════════════════════════════════════════╗
        ║          APPLY SUSCEPTIBILITY DISTORTION CORRECTION            ║
        ╚════════════════════════════════════════════════════════════════╝
        
        This script applies susceptibility distortion correction to diffusion images
        using a pre-calculated warp field. It takes a motion-corrected diffusion image
        and applies the warp field to each 3D volume along the y-axis.
        
        ────────────────────────── USAGE ──────────────────────────
          micaflow apply_SDC [options]
        
        ────────────────── REQUIRED ARGUMENTS ───────────────────
          --input       : Path to the motion-corrected DWI image (.nii.gz)
          --warp        : Path to the warp field estimated from SDC (.nii.gz)
          --affine      : Path to an image from which to extract the affine matrix
          --output      : Output path for the corrected image
        
        ─────────────────── EXAMPLE USAGE ───────────────────────
        
        # Apply SDC to a motion-corrected DWI image
        micaflow apply_SDC \
          --input subj_motion_corrected.nii.gz \
          --warp SDC.nii.gz \
          --affine original_dwi.nii.gz \
          --output corrected_dwi.nii.gz
        
        ────────────────────────── NOTES ───────────────────────
        • The warp field should contain displacement values along the y-axis
        • This implementation assumes that susceptibility distortions are primarily 
          in the phase-encoding direction (typically y-axis)
        
    


.. automodule:: micaflow.scripts.apply_SDC
   :noindex:
