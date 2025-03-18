Coregister
==========

Command Line Usage
-----------------

.. code-block:: bash

    micaflow coregister [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow2.0/blob/main/micaflow/scripts/coregister.py>`_

Description
-----------

This script performs linear (rigid + affine) and nonlinear (SyN) registration 
    between two images using ANTs. The registration aligns the moving image to 
    match the fixed reference image space.

Full Help
---------

.. code-block:: text

    
        ╔════════════════════════════════════════════════════════════════╗
        ║                      IMAGE COREGISTRATION                      ║
        ╚════════════════════════════════════════════════════════════════╝
        
        This script performs linear (rigid + affine) and nonlinear (SyN) registration 
        between two images using ANTs. The registration aligns the moving image to 
        match the fixed reference image space.
        
        ────────────────────────── USAGE ──────────────────────────
          micaflow coregister [options]
        
        ─────────────────── REQUIRED ARGUMENTS ───────────────────
          --fixed-file   : Path to the fixed/reference image (.nii.gz)
          --moving-file  : Path to the moving image to be registered (.nii.gz)
          --output       : Output path for the registered image (.nii.gz)
        
        ─────────────────── OPTIONAL ARGUMENTS ───────────────────
          --warp-file      : Path to save the forward warp field (.nii.gz)
          --affine-file    : Path to save the forward affine transform (.mat)
          --rev-warp-file  : Path to save the reverse warp field (.nii.gz)
          --rev-affine-file: Path to save the reverse affine transform (.mat)
        
        ────────────────── EXAMPLE USAGE ────────────────────────
        
        # Register a moving image to a fixed image
        micaflow coregister --fixed-file mni152.nii.gz --moving-file subject_t1w.nii.gz \
          --output registered_t1w.nii.gz --warp-file warp.nii.gz --affine-file affine.mat
        
        ────────────────────────── NOTES ───────────────────────
        • The registration performs SyNRA transformation (rigid+affine+SyN)
        • Forward transforms convert from moving space to fixed space
        • Reverse transforms convert from fixed space to moving space
        • The transforms can be applied to other images using apply_warp
        
    


.. automodule:: micaflow.scripts.coregister
   :noindex:
