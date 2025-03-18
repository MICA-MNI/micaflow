Texture Generation
==================

Command Line Usage
-----------------

.. code-block:: bash

    micaflow texture_generation [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow2.0/blob/main/micaflow/scripts/texture_generation.py>`_

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
