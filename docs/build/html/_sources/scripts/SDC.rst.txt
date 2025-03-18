Sdc
===

SDC - Susceptibility Distortion Correction for EPI/DWI

Part of the micaflow processing pipeline for neuroimaging data.

This module corrects geometric distortions in echo-planar imaging (EPI) MR images 
caused by magnetic field inhomogeneities. It implements the HYSCO (HYperellastic 
Susceptibility artifact COrrection) algorithm that uses a pair of images acquired 
with opposite phase-encoding directions to estimate and correct these distortions. 
The algorithm estimates a displacement field along the phase-encoding direction 
(typically y-axis) that can be used to unwarp the distorted images.

Features:
--------
- B0 field estimation using phase-encoding reversed image pairs
- GPU acceleration with PyTorch for faster processing when available
- Automatic initial alignment using ANTs affine registration
- Advanced optimization using Alternating Direction Method of Multipliers (ADMM)
- Outputs both corrected images and estimated displacement fields for further usage
- Temporary file management for clean processing pipeline

API Usage:
---------
micaflow SDC 
    --input <path/to/forward_phase_encoded.nii.gz>
    --reverse-image <path/to/reverse_phase_encoded.nii.gz>
    --output <path/to/corrected_image.nii.gz>
    --output-warp <path/to/displacement_field.nii.gz>

Python Usage:
-----------
>>> from micaflow.scripts.SDC import run
>>> run(
...     data_image="forward_phase_encoded.nii.gz",
...     reverse_image="reverse_phase_encoded.nii.gz",
...     output_name="corrected_image.nii.gz",
...     output_warp="displacement_field.nii.gz"
... )

Command Line Usage
-----------------

.. code-block:: bash

    micaflow SDC [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow/blob/main/micaflow/scripts/SDC.py>`_

Description
-----------

This script corrects geometric distortions in echo-planar imaging (EPI) 
    MR images caused by magnetic field inhomogeneities. It uses the HYSCO 
    algorithm with a pair of images acquired with opposite phase-encoding 
    directions.

Full Help
---------

.. code-block:: text

    
        ╔════════════════════════════════════════════════════════════════╗
        ║               SUSCEPTIBILITY DISTORTION CORRECTION             ║
        ╚════════════════════════════════════════════════════════════════╝
        
        This script corrects geometric distortions in echo-planar imaging (EPI) 
        MR images caused by magnetic field inhomogeneities. It uses the HYSCO 
        algorithm with a pair of images acquired with opposite phase-encoding 
        directions.
        
        ────────────────────────── USAGE ──────────────────────────
          micaflow SDC [options]
        
        ─────────────────── REQUIRED ARGUMENTS ───────────────────
          --input         : Path to the main EPI image (.nii.gz)
          --reverse-image : Path to the reverse phase-encoded image (.nii.gz)
          --output        : Output path for the corrected image (.nii.gz)
          --output-warp   : Output path for the estimated warp field (.nii.gz)
        
        ──────────────────────── EXAMPLE USAGE ───────────────────────
          micaflow SDC \
            --input main_epi.nii.gz \
            --reverse-image reverse_epi.nii.gz \
            --output corrected_epi.nii.gz \
            --output-warp warp_field.nii.gz
        
        ────────────────────────── NOTES ─────────────────────────
        - The algorithm extracts the first volume from 4D input images
        - GPU acceleration is used if available (recommended)
        - The correction estimates a displacement field along the y-axis
        - This implementation uses the HYSCO (HYperellastic Susceptibility 
          artifact COrrection) algorithm
        
        
    


.. automodule:: micaflow.scripts.SDC
   :noindex:
