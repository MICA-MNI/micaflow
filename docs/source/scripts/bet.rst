Bet
===

bet - Brain Extraction Tool using HD-BET

Part of the micaflow processing pipeline for neuroimaging data.

This module provides brain extraction (skull stripping) functionality using the High-Definition
Brain Extraction Tool (HD-BET), a deep learning-based approach that accurately segments the 
brain from surrounding tissues in MR images. HD-BET offers superior performance over traditional
methods, particularly for clinical and non-standard MR images.

Features:
--------
- Deep learning-based brain extraction with state-of-the-art accuracy
- Support for both CPU and GPU execution modes
- Compatible with various MRI modalities (T1w, T2w, FLAIR)
- Produces both skull-stripped images and binary brain masks
- Robust to imaging artifacts and pathologies

API Usage:
---------
micaflow bet 
    --input <path/to/image.nii.gz>
    --output <path/to/brain.nii.gz>
    --output-mask <path/to/brain_mask.nii.gz>
    [--cpu]

Python Usage:
-----------
>>> import subprocess
>>> from micaflow.scripts.bet import run_hdbet
>>> run_hdbet(
...     input_file="t1w.nii.gz",
...     output_file="brain.nii.gz",
...     mask_file="brain_mask.nii.gz",
...     use_cpu=False
... )

Command Line Usage
-----------------

.. code-block:: bash

    micaflow bet [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow/blob/main/micaflow/scripts/bet.py>`_

Description
-----------

This script performs brain extraction (skull stripping) on MRI images 
    using the HD-BET deep learning tool. It accurately segments the brain 
    from surrounding tissues.

Full Help
---------

.. code-block:: text

    
        ╔════════════════════════════════════════════════════════════════╗
        ║                           HD-BET                               ║
        ╚════════════════════════════════════════════════════════════════╝
        
        This script performs brain extraction (skull stripping) on MRI images 
        using the HD-BET deep learning tool. It accurately segments the brain 
        from surrounding tissues.
        
        ────────────────────────── USAGE ──────────────────────────
          micaflow bet [options]
        
        ─────────────────── REQUIRED ARGUMENTS ───────────────────
          --input, -i      : Path to the input MR image (.nii.gz)
          --output, -o     : Path for the output brain-extracted image (.nii.gz)
          --output-mask, -m: Path for the output brain mask (.nii.gz)
        
        ─────────────────── OPTIONAL ARGUMENTS ───────────────────
          --cpu            : Use CPU instead of GPU for computation (slower but works without CUDA)
        
        ────────────────── EXAMPLE USAGE ────────────────────────
        
        # Run HD-BET with GPU
        micaflow bet --input t1w.nii.gz --output t1w_brain.nii.gz --output-mask t1w_brain_mask.nii.gz
        
        # Run HD-BET with CPU
        micaflow bet --input t1w.nii.gz --output t1w_brain.nii.gz --output-mask t1w_brain_mask.nii.gz --cpu
        
        ────────────────────────── NOTES ─────────────────────────
        - GPU acceleration is used by default for faster processing
        - The output is a brain-extracted image and a binary brain mask
        
        
    


.. automodule:: micaflow.scripts.bet
   :noindex:
