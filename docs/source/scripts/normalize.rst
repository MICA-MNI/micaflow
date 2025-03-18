Normalize
=========

normalize_intensity - Percentile-based Intensity Normalization for MRI Data

Part of the micaflow processing pipeline for neuroimaging data.

This script performs intensity normalization on MRI data by:
1. Clamping values at the 1st and 99th percentiles to reduce outlier effects
2. Rescaling the clamped values to a standardized 0-100 range

This normalization helps improve consistency across different scans and scanners,
making downstream analysis and visualization more robust.

API Usage:
---------
micaflow normalize_intensity 
    --input <path/to/image.nii.gz>
    --output <path/to/normalized.nii.gz>
    [--lower-percentile <value>]
    [--upper-percentile <value>]
    [--min-value <value>]
    [--max-value <value>]

Python Usage:
-----------
>>> from micaflow.scripts.normalize_intensity import normalize_intensity
>>> normalize_intensity(
...     input_file="t1w.nii.gz",
...     output_file="t1w_normalized.nii.gz",
...     lower_percentile=1,
...     upper_percentile=99,
...     min_val=0,
...     max_val=100
... )

Command Line Usage
-----------------

.. code-block:: bash

    micaflow normalize [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow/blob/main/micaflow/scripts/normalize.py>`_

Description
-----------

This script normalizes MRI intensity values by clamping at specified 
    percentiles and rescaling to a standard range.

Full Help
---------

.. code-block:: text

    
        ╔════════════════════════════════════════════════════════════════╗
        ║                 INTENSITY NORMALIZATION                        ║
        ╚════════════════════════════════════════════════════════════════╝
        
        This script normalizes MRI intensity values by clamping at specified 
        percentiles and rescaling to a standard range.
        
        ────────────────────────── USAGE ──────────────────────────
          micaflow normalize_intensity [options]
        
        ─────────────────── REQUIRED ARGUMENTS ───────────────────
          --input, -i  : Path to the input image file (.nii.gz)
          --output, -o : Path for the normalized output image (.nii.gz)
        
        ─────────────────── OPTIONAL ARGUMENTS ───────────────────
          --lower-percentile : Lower percentile for clamping (default: 1.0)
          --upper-percentile : Upper percentile for clamping (default: 99.0)
          --min-value        : Minimum value in output range (default: 0)
          --max-value        : Maximum value in output range (default: 100)
        
        ──────────────────── EXAMPLE USAGE ──────────────────────
        
        # Basic usage with default parameters
        micaflow normalize_intensity --input t1w.nii.gz --output t1w_norm.nii.gz
        
        # Custom percentiles and range
        micaflow normalize_intensity --input t1w.nii.gz --output t1w_norm.nii.gz       --lower-percentile 2.0 --upper-percentile 98.0 --min-value 0 --max-value 1
        
        ────────────────────────── NOTES ─────────────────────────
        - Clamping at percentiles helps reduce the effect of outliers
        - Data type is preserved in the output image
        - Non-brain voxels (zeros) remain zero after normalization
        
        
    


.. automodule:: micaflow.scripts.normalize
   :noindex:
