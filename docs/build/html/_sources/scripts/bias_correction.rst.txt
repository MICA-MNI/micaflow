Bias Correction
===============

N4 Bias Field Correction script for both anatomical and diffusion MR images.

This script provides functionality to correct intensity non-uniformity (bias field)
in MR images using the N4 algorithm from the Advanced Normalization Tools (ANTs) library.
It supports both 3D anatomical images and 4D diffusion-weighted images.

Examples:
    # For anatomical (3D) images:
    python bias_correction.py --input t1w.nii.gz --output corrected.nii.gz

    # For anatomical images with mask:
    python bias_correction.py --input t1w.nii.gz --output corrected.nii.gz --mask brain_mask.nii.gz

    # For diffusion (4D) images:
    python bias_correction.py --input dwi.nii.gz --output corrected.nii.gz --mask brain_mask.nii.gz --mode 4d

Command Line Usage
-----------------

.. code-block:: bash

    micaflow bias_correction [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow2.0/blob/main/micaflow/scripts/bias_correction.py>`_

Description
-----------

This script corrects intensity non-uniformity (bias field) in MR images 
    using the N4 algorithm from ANTs. It supports both 3D anatomical images 
    and 4D diffusion-weighted images.

Full Help
---------

.. code-block:: text

    
        ╔════════════════════════════════════════════════════════════════╗
        ║                    N4 BIAS FIELD CORRECTION                    ║
        ╚════════════════════════════════════════════════════════════════╝
        
        This script corrects intensity non-uniformity (bias field) in MR images 
        using the N4 algorithm from ANTs. It supports both 3D anatomical images 
        and 4D diffusion-weighted images.
        
        ──────────────────── REQUIRED ARGUMENTS ────────────────────
          --input, -i    : Path to the input image (.nii.gz)
          --output, -o   : Path for the output bias-corrected image (.nii.gz)
        
        ──────────────────── OPTIONAL ARGUMENTS ────────────────────
          --mask, -m     : Path to a brain mask image (required for 4D images)
          --mode         : Processing mode: 3d, 4d, or auto (default: auto)
        
        ──────────────────── EXAMPLE USAGE ────────────────────
        
        # For anatomical (3D) images:
        micaflow bias_correction \
          --input t1w.nii.gz \
          --output corrected_t1w.nii.gz
        
        # For diffusion (4D) images with mask:
        micaflow bias_correction \
          --input dwi.nii.gz \
          --output corrected_dwi.nii.gz \
          --mask brain_mask.nii.gz \
          --mode 4d
        
        ────────────────────────── NOTES ───────────────────────
        • In 'auto' mode, the script detects whether the input is 3D or 4D
        • For 3D images, a mask is optional (one will be generated if not provided)
        • For 4D images, a mask is required
        • 4D processing applies the correction to each volume separately
        
    


.. automodule:: micaflow.scripts.bias_correction
   :noindex:
