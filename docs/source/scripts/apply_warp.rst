Apply Warp
==========

Command Line Usage
-----------------

.. code-block:: bash

    micaflow apply_warp [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow2.0/blob/main/micaflow/scripts/apply_warp.py>`_

Description
-----------

This script applies both an affine transformation and a warp field to
    register a moving image to a reference space.

Full Help
---------

.. code-block:: text

    
        ╔════════════════════════════════════════════════════════════════╗
        ║                        APPLY WARP                              ║
        ╚════════════════════════════════════════════════════════════════╝
        
        This script applies both an affine transformation and a warp field to
        register a moving image to a reference space.
        
        ────────────────────────── REQUIRED ARGUMENTS ──────────────────────────
          --moving     : Path to the input image to be warped (.nii.gz)
          --reference  : Path to the target/reference image (.nii.gz)
          --affine     : Path to the affine transformation file (.mat)
          --warp       : Path to the warp field (.nii.gz)
        
        ────────────────────────── OPTIONAL ARGUMENTS ──────────────────────────
          --output     : Output path for the warped image (default: warped_image.nii.gz)
        
        ────────────────────────── EXAMPLE USAGE ──────────────────────────
        
        # Apply warp transformation
        micaflow apply_warp --moving subject_t1w.nii.gz --reference mni152.nii.gz \
          --affine transform.mat --warp warpfield.nii.gz --output registered_t1w.nii.gz
        
        ────────────────────────── NOTES ──────────────────────────
        • The order of transforms matters: the warp field is applied first, 
          followed by the affine transformation.
        • This is the standard order in ANTs for composite transformations.
        
    


.. automodule:: micaflow.scripts.apply_warp
   :noindex:
