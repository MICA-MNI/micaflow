Compute Fa Md
=============

compute_fa_md - Diffusion Tensor Imaging Metrics Calculator

Part of the micaflow processing pipeline for neuroimaging data.

This module computes diffusion tensor imaging (DTI) scalar metrics, specifically 
Fractional Anisotropy (FA) and Mean Diffusivity (MD), from preprocessed diffusion-weighted 
images (DWI). FA quantifies the directional preference of water diffusion, serving as a 
marker of white matter integrity, while MD represents the overall magnitude of diffusion. 
These metrics are essential for investigating white matter microstructure and are widely 
used in clinical and research neuroimaging.

Features:
--------
- Computes DTI model using robust tensor fitting from DIPY
- Generates both FA and MD maps in a single operation
- Supports masking to restrict calculations to brain tissue
- Compatible with standard neuroimaging file formats (NIfTI)
- Preserves image header and spatial information in output files

API Usage:
---------
micaflow compute_fa_md 
    --input <path/to/dwi.nii.gz>
    --mask <path/to/brain_mask.nii.gz>
    --bval <path/to/dwi.bval>
    --bvec <path/to/dwi.bvec>
    --output-fa <path/to/fa_map.nii.gz>
    --output-md <path/to/md_map.nii.gz>

Python Usage:
-----------
>>> from micaflow.scripts.compute_fa_md import compute_fa_md
>>> fa_path, md_path = compute_fa_md(
...     bias_corr_path="corrected_dwi.nii.gz",
...     mask_path="brain_mask.nii.gz",
...     moving_bval="dwi.bval",
...     moving_bvec="dwi.bvec",
...     fa_path="fa.nii.gz",
...     md_path="md.nii.gz"
... )

Command Line Usage
-----------------

.. code-block:: bash

    micaflow compute_fa_md [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow/blob/main/micaflow/scripts/compute_fa_md.py>`_

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
