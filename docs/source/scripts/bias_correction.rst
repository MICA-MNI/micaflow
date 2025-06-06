Bias Correction
===============

bias_correction - N4 Bias Field Correction for MRI data

Part of the micaflow processing pipeline for neuroimaging data.

This module corrects intensity non-uniformity (bias field) in MR images using the 
N4 algorithm from Advanced Normalization Tools (ANTs). Intensity bias appears as a 
smooth variation of signal intensity across the image and can affect subsequent analysis 
steps like segmentation or registration. The N4 algorithm estimates this bias field 
and removes it, producing more uniform intensities across tissues.

Features:
--------
- Supports both 3D anatomical images and 4D diffusion-weighted images
- Automatic detection of image dimensionality (3D vs 4D)
- Optional brain mask input for improved correction accuracy
- Volume-by-volume processing for 4D images preserves temporal dynamics
- Maintains image header information in the corrected output

API Usage:
---------
micaflow bias_correction 
    --input <path/to/image.nii.gz>
    --output <path/to/corrected.nii.gz>
    [--mask <path/to/brain_mask.nii.gz>]
    [--mode <3d|4d|auto>]

Python Usage:
-----------
>>> from micaflow.scripts.bias_correction import run_bias_field_correction
>>> run_bias_field_correction(
...     image_path="t1w.nii.gz",
...     output_path="corrected_t1w.nii.gz",
...     mask_path="brain_mask.nii.gz",  # optional for 3D images
...     mode="auto"  # auto, 3d, or 4d
... )

Command Line Usage
-----------------

.. code-block:: bash

    micaflow bias_correction [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow/blob/main/micaflow/scripts/bias_correction.py>`_

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
