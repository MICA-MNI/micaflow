Synthseg
========

This script enables to launch predictions with SynthSeg from the terminal.

If you use this code, please cite one of the SynthSeg papers:
https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib

Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.

Command Line Usage
-----------------

.. code-block:: bash

    micaflow synthseg [options]

Source Code
-----------

View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow2.0/blob/main/micaflow/scripts/synthseg.py>`_

Description
-----------

This script runs the SynthSeg neural network-based tool for brain MRI
    segmentation. It provides automated segmentation of anatomical structures
    even across different contrasts and acquisition types.

Full Help
---------

.. code-block:: text

    
        ╔════════════════════════════════════════════════════════════════╗
        ║                         SYNTHSEG                               ║
        ╚════════════════════════════════════════════════════════════════╝
        
        This script runs the SynthSeg neural network-based tool for brain MRI
        segmentation. It provides automated segmentation of anatomical structures
        even across different contrasts and acquisition types.
        
        ────────────────────────── USAGE ──────────────────────────
          micaflow synthseg [options]
        
        ─────────────────── REQUIRED ARGUMENTS ───────────────────
          --i PATH       : Input image(s) to segment (file or folder)
          --o PATH       : Output segmentation file(s) or folder
        
        ─────────────────── OPTIONAL ARGUMENTS ───────────────────
          --parc         : Enable cortical parcellation
          --robust       : Use robust mode (slower but better quality)
          --fast         : Faster processing (less postprocessing)
          --threads N    : Set number of CPU threads (default: 1)
          --cpu          : Force CPU processing (instead of GPU)
          --vol PATH     : Output volumetric CSV file
          --qc PATH      : Output quality control scores CSV file
          --post PATH    : Output posterior probability maps
          --resample PATH: Output resampled images
          --crop N [N ...]: Size of 3D patches to analyze (default: 192)
          --ct           : Clip intensities for CT scans [0,80]
          --v1           : Use SynthSeg 1.0 instead of 2.0
        
        ────────────────── EXAMPLE USAGE ────────────────────────
        
        # Basic segmentation
        micaflow synthseg \
          --i t1w_scan.nii.gz \
          --o segmentation.nii.gz
        
        # With cortical parcellation
        micaflow synthseg \
          --i t1w_scan.nii.gz \
          --o segmentation.nii.gz \
          --parc
        
        # Batch processing with volume calculation
        micaflow synthseg \
          --i input_folder/ \
          --o output_folder/ \
          --vol volumes.csv
        
        ────────────────────────── NOTES ───────────────────────
        • SynthSeg works with any MRI contrast without retraining
        • GPU acceleration is used by default for faster processing
        • The robust mode provides better quality but is slower
        • For batch processing, input and output paths must be folders
        
    


.. automodule:: micaflow.scripts.synthseg
   :noindex:
