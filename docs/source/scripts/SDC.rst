Sdc
===

Command Line Usage
-----------------

.. code-block:: bash

    micaflow SDC [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow2.0/blob/main/micaflow/scripts/SDC.py>`_

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
