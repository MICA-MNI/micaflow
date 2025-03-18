Bet
===

HD-BET (High-Definition Brain Extraction Tool) wrapper script.

This script provides a simplified command-line interface to the HD-BET brain extraction
tool, which performs accurate skull stripping on brain MR images using a deep learning approach.
It supports both CPU and GPU execution modes.

The script is a wrapper around the HD-BET entry_point.py script that simplifies the interface
and handles path resolution.

Example:
    python hdbet.py --input t1w.nii.gz --output t1w_brain.nii.gz
    python hdbet.py --input t1w.nii.gz --output t1w_brain.nii.gz --cpu

Command Line Usage
-----------------

.. code-block:: bash

    micaflow bet [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow2.0/blob/main/micaflow/scripts/bet.py>`_

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
