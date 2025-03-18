Denoise
=======

Command Line Usage
-----------------

.. code-block:: bash

    micaflow denoise [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow2.0/blob/main/micaflow/scripts/denoise.py>`_

Description
-----------

This script denoises diffusion-weighted images (DWI) using the Patch2Self 
    algorithm, which leverages redundant information across diffusion gradients
    to remove noise without requiring additional reference scans.

Full Help
---------

.. code-block:: text

    
        ╔════════════════════════════════════════════════════════════════╗
        ║                     DWI IMAGE DENOISING                        ║
        ╚════════════════════════════════════════════════════════════════╝
        
        This script denoises diffusion-weighted images (DWI) using the Patch2Self 
        algorithm, which leverages redundant information across diffusion gradients
        to remove noise without requiring additional reference scans.
        
        ────────────────────────── USAGE ──────────────────────────
          micaflow denoise [options]
        
        ─────────────────── REQUIRED ARGUMENTS ───────────────────
          --input     : Path to the input DWI image (.nii.gz)
          --bval      : Path to the b-values file (.bval)
          --bvec      : Path to the b-vectors file (.bvec)
          --output    : Output path for the denoised image (.nii.gz)
        
        ─────────────────── EXAMPLE USAGE ───────────────────
          micaflow denoise \
            --input raw_dwi.nii.gz \
            --bval dwi.bval \
            --bvec dwi.bvec \
            --output denoised_dwi.nii.gz
        
        ────────────────────────── NOTES ─────────────────────────
        - Patch2Self is a self-supervised learning method for denoising
        - Processing preserves anatomical structure while removing noise
        - The implementation uses OLS regression with b0 threshold of 50 s/mm²
        - B0 volumes are not denoised separately in this implementation
        
    


.. automodule:: micaflow.scripts.denoise
   :noindex:
