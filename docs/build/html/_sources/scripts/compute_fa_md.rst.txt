Compute Fa Md
=============

Command Line Usage
-----------------

.. code-block:: bash

    micaflow compute_fa_md [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow2.0/blob/main/micaflow/scripts/compute_fa_md.py>`_

Description
-----------

This script computes Fractional Anisotropy (FA) and Mean Diffusivity (MD)
    maps from diffusion-weighted images using the diffusion tensor model.

Full Help
---------

.. code-block:: text

    
        ╔════════════════════════════════════════════════════════════════╗
        ║                DIFFUSION TENSOR METRICS (FA/MD)                ║
        ╚════════════════════════════════════════════════════════════════╝
        
        This script computes Fractional Anisotropy (FA) and Mean Diffusivity (MD)
        maps from diffusion-weighted images using the diffusion tensor model.
        
        ────────────────────────── USAGE ──────────────────────────
          micaflow compute_fa_md [options]
        
        ─────────────────── REQUIRED ARGUMENTS ───────────────────
          --input      : Path to the input DWI image (.nii.gz)
          --mask       : Path to the brain mask image (.nii.gz)
          --bval       : Path to the b-values file (.bval)
          --bvec       : Path to the b-vectors file (.bvec)
          --output-fa  : Output path for the FA map (.nii.gz)
          --output-md  : Output path for the MD map (.nii.gz)
        
        ──────────────────── EXAMPLE USAGE ──────────────────────
          micaflow compute_fa_md \
            --input corrected_dwi.nii.gz \
            --mask brain_mask.nii.gz \
            --bval dwi.bval \
            --bvec dwi.bvec \
            --output-fa fa.nii.gz \
            --output-md md.nii.gz
        
        ────────────────────────── NOTES ─────────────────────────
        - FA (Fractional Anisotropy) values range from 0 (isotropic) to 1 (anisotropic)
        - MD (Mean Diffusivity) measures the overall magnitude of diffusion
        - Processing requires a brain mask to exclude non-brain regions
        
        
    


.. automodule:: micaflow.scripts.compute_fa_md
   :noindex:
