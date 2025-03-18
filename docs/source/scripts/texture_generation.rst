Texture Generation
==================

texture_generation - MRI Texture Feature Extraction

Part of the micaflow processing pipeline for neuroimaging data.

This module computes advanced texture features from MRI data that can be used for
tissue characterization, lesion analysis, or radiomics applications. It performs
automatic tissue segmentation and extracts quantitative imaging features including
gradient magnitude and relative intensity maps, which capture local intensity variations
and tissue contrast properties respectively.

Features:
--------
- Automatic tissue segmentation into gray matter, white matter, and CSF
- Gradient magnitude computation for edge and boundary detection
- Relative intensity calculation for normalized tissue contrast
- Masked processing to focus analysis on brain regions only
- Output in standard NIfTI format compatible with other neuroimaging tools
- Efficient implementation using ANTs image processing functions

API Usage:
---------
micaflow texture_generation 
    --input <path/to/image.nii.gz>
    --mask <path/to/brain_mask.nii.gz>
    --output <path/to/output_prefix>

Python Usage:
-----------
>>> from micaflow.scripts.texture_generation import run_texture_pipeline
>>> run_texture_pipeline(
...     input="preprocessed_t1w.nii.gz",
...     mask="brain_mask.nii.gz",
...     output_dir="output_texture_maps"
... )

Command Line Usage
-----------------

.. code-block:: bash

    micaflow texture_generation [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow/blob/main/micaflow/scripts/texture_generation.py>`_

Description
-----------

This script generates texture feature maps from neuroimaging data using
    various computational approaches. The features include gradient magnitude,
    relative intensity, and tissue segmentation.

Full Help
---------

.. code-block:: text

    
        ╔════════════════════════════════════════════════════════════════╗
        ║                    TEXTURE FEATURE EXTRACTION                  ║
        ╚════════════════════════════════════════════════════════════════╝
        
        This script generates texture feature maps from neuroimaging data using
        various computational approaches. The features include gradient magnitude,
        relative intensity, and tissue segmentation.
        
        ────────────────────────── USAGE ──────────────────────────
          micaflow texture_generation [options]
        
        ─────────────────── REQUIRED ARGUMENTS ───────────────────
          --input, -i   : Path to the input image file (.nii.gz)
          --mask, -m    : Path to the binary mask file (.nii.gz)
          --output, -o  : Output directory for texture feature maps
        
        ──────────────────────── EXAMPLE USAGE ───────────────────────
          micaflow texture_generation \
            --input t1w_preprocessed.nii.gz \
            --mask brain_mask.nii.gz \
            --output /path/to/output_dir
        
        ────────────────────────── NOTES ─────────────────────────
        - The script automatically segments the input into tissue types
        - Computed features include gradient magnitude and relative intensity
        - All features are saved as separate NIfTI files in the output directory
        - Processing may take several minutes depending on image size
        
        
    


.. automodule:: micaflow.scripts.texture_generation
   :noindex:
