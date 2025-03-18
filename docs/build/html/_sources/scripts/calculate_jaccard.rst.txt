Calculate Jaccard
=================

Command Line Usage
-----------------

.. code-block:: bash

    micaflow calculate_jaccard [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow2.0/blob/main/micaflow/scripts/calculate_jaccard.py>`_

Description
-----------

This script calculates the Jaccard similarity index (intersection over union)
    between two segmentation volumes, either globally or for each ROI.

Full Help
---------

.. code-block:: text

    
        ╔════════════════════════════════════════════════════════════════╗
        ║                    JACCARD INDEX CALCULATOR                    ║
        ╚════════════════════════════════════════════════════════════════╝
        
        This script calculates the Jaccard similarity index (intersection over union)
        between two segmentation volumes, either globally or for each ROI.
        
        ────────────────────────── USAGE ──────────────────────────
          micaflow calculate_jaccard [options]
        
        ─────────────────── REQUIRED ARGUMENTS ───────────────────
          --input, -i     : Path to the first input segmentation volume (.nii.gz)
          --reference, -r : Path to the reference segmentation volume (.nii.gz)
          --output, -o    : Output path for the CSV file with Jaccard indices
        
        ─────────────────── OPTIONAL ARGUMENTS ───────────────────
          --mask, -m      : Optional mask to restrict comparison to a specific region
          --threshold, -t : Threshold value for probabilistic segmentations (default: 0.5)
        
        ──────────────────── EXAMPLE USAGE ──────────────────────
          micaflow calculate_jaccard \
            --input segmentation1.nii.gz \
            --reference ground_truth.nii.gz \
            --output jaccard_metrics.csv
          
          # With mask and custom threshold:
          micaflow calculate_jaccard \
            --input segmentation1.nii.gz \
            --reference ground_truth.nii.gz \
            --output jaccard_metrics.csv \
            --mask brain_mask.nii.gz \
            --threshold 0.75
        
        ────────────────────────── NOTES ─────────────────────────
        - For multi-label segmentations, the Jaccard index is computed for each label
        - Values range from 0 (no overlap) to 1 (perfect overlap)
        - A global Jaccard index is calculated across all labels
        
    


.. automodule:: micaflow.scripts.calculate_jaccard
   :noindex:
