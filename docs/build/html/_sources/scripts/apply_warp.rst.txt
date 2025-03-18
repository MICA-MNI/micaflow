Apply Warp
==========

apply_warp - Image registration transformation application

Part of the micaflow processing pipeline for neuroimaging data.

This module applies spatial transformations to register images from one space to another
using both affine and non-linear (warp field) transformations. It's commonly used to:
- Transform subject images to a standard space (e.g., MNI152)
- Register images across modalities (e.g., T1w to FLAIR)
- Apply previously calculated transformations to derived images (e.g., segmentations)

The module leverages ANTsPy to apply the transformations in the correct order (warp 
field first, then affine) to achieve accurate spatial registration.

API Usage:
---------
micaflow apply_warp 
    --moving <path/to/source_image.nii.gz>
    --reference <path/to/target_space.nii.gz>
    --affine <path/to/transform.mat>
    --warp <path/to/warpfield.nii.gz>
    [--output <path/to/registered_image.nii.gz>]

Python Usage:
-----------
>>> import ants
>>> from micaflow.scripts.apply_warp import apply_warp
>>> moving_img = ants.image_read("subject_t1w.nii.gz")
>>> reference_img = ants.image_read("mni152.nii.gz")
>>> apply_warp(
...     moving_img=moving_img,
...     reference_img=reference_img,
...     affine_file="transform.mat",
...     warp_file="warpfield.nii.gz", 
...     out_file="registered_t1w.nii.gz"
... )

References:
----------
1. Avants BB, Tustison NJ, Song G, et al. A reproducible evaluation of ANTs 
   similarity metric performance in brain image registration. NeuroImage. 
   2011;54(3):2033-2044. doi:10.1016/j.neuroimage.2010.09.025

Command Line Usage
-----------------

.. code-block:: bash

    micaflow apply_warp [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow/blob/main/micaflow/scripts/apply_warp.py>`_

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
