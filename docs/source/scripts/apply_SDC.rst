Apply Sdc
=========

Command Line Usage
-----------------

.. code-block:: bash

    micaflow apply_SDC [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow2.0/blob/main/micaflow/scripts/apply_SDC.py>`_

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
