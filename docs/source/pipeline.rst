.. _pipeline:

MICAflow Pipeline
================

Overview
--------

The MICAflow pipeline provides a comprehensive framework for processing structural and diffusion MRI data through a series of coordinated steps implemented as a Snakemake workflow.

.. tabs::

   .. tab:: Pipeline Structure

      The pipeline is organized into several processing stages that are executed in a specific order, with dependencies managed automatically by Snakemake:

      1. **Structural Processing:**
         - Skull stripping of T1w and FLAIR images
         - Bias field correction
         - SynthSeg segmentation

      2. **Registration:**
         - T1w to MNI152 space
         - FLAIR to T1w space
         - DWI to T1w space (if DWI data is available)

      3. **Texture Feature Generation:**
         - Gradient magnitude maps
         - Relative intensity maps

      4. **Diffusion Processing (Optional):**
         - Denoising
         - Motion correction
         - Susceptibility distortion correction
         - Computation of FA and MD maps

      5. **Quality Metrics:**
         - Jaccard similarity for registration accuracy

   .. tab:: Inputs & Outputs

      **Required Inputs:**
      
      - ``--subject``: Subject ID (e.g., sub-01)
      - ``--output``: Output directory
      - ``--t1w-file``: Path to T1-weighted image file
      
      **Optional Inputs:**
      
      - ``--session``: Session ID (e.g., ses-01)
      - ``--flair-file``: Path to FLAIR image file
      - ``--dwi-file``: Diffusion weighted image
      - ``--bval-file``: B-value file for DWI
      - ``--bvec-file``: B-vector file for DWI
      - ``--inverse-dwi-file``: Inverse (PA) DWI for distortion correction
      
      **Primary Outputs:**
      
      The pipeline generates the following directory structure:
      
      .. code-block:: text
      
         <OUTPUT_DIR>/
         └── <SUBJECT>/
             └── <SESSION>/
                 ├── anat/                    # Preprocessed anatomical images
                 ├── dwi/                     # Preprocessed diffusion data (if available)
                 ├── metrics/                 # Quality assessment metrics
                 ├── textures/                # Texture feature maps
                 └── xfm/                     # Transformation files

   .. tab:: Running the Pipeline

      **Command Line Usage:**

      .. code-block:: bash

         micaflow pipeline \
           --subject SUB001 \
           --session SES01 \
           --output /path/to/output \
           --data-directory /path/to/data \
           --t1w-file /path/to/t1w.nii.gz \
           [options]

      **Additional Options:**

      - ``--threads N``: Number of threads to use (default: 1)
      - ``--cpu``: Force CPU computation instead of GPU
      - ``--dry-run``: Show what would be executed without running commands

      **Pipeline Configuration:**

      The pipeline can also be configured using a YAML file:

      .. code-block:: bash

         micaflow pipeline --config-file config.yaml

   .. tab:: Texture Generation

      The texture generation component extracts advanced features from neuroimaging data:

      .. code-block:: text

         ╔════════════════════════════════════════════════════════════════╗
         ║                    TEXTURE FEATURE EXTRACTION                  ║
         ╚════════════════════════════════════════════════════════════════╝

         This script generates texture feature maps from neuroimaging data using
         various computational approaches. The features include:
         
         - Gradient magnitude computation for edge and boundary detection
         - Relative intensity calculation for normalized tissue contrast
         - Automatic tissue segmentation into gray matter, white matter, and CSF
         - Masked processing to focus analysis on brain regions only

      **Output Features:**

      - **Gradient Magnitude Maps**: Highlight tissue boundaries and structural transitions
      - **Relative Intensity Maps**: Normalize intensity patterns across the brain
      - **Segmentation Maps**: Tissue class probabilities

Implementation Details
---------------------

The pipeline implementation follows a modular design where each processing step is encapsulated as a separate rule in the Snakefile:

.. code-block:: python

   # Key pipeline rules from Snakefile
   rule skull_strip:
       # Extract brain tissue from T1w/FLAIR images
   
   rule bias_field_correction:
       # Correct intensity non-uniformities
   
   rule synthseg_t1w:
       # AI-based segmentation of T1w images
   
   rule registration_t1w:
       # Register FLAIR to T1w space
   
   rule registration_mni152:
       # Register T1w to standard space
   
   rule run_texture:
       # Generate texture feature maps
   
   # Additional DWI processing rules when enabled
   if RUN_DWI:
       rule dwi_denoise:
           # Remove noise from diffusion images
       
       rule dwi_motion_correction:
           # Correct for head motion in diffusion data
       
       # ... additional DWI rules ...

Quality Control
--------------

The pipeline includes quality assessment metrics to evaluate the performance of critical processing steps:

1. **Registration Accuracy**: Jaccard similarity metrics between registered images
2. **Transformation Files**: All transformation matrices and warp fields are saved for inspection
3. **Intermediate Results**: Preprocessed images at each stage for quality checks

For complete implementation details, refer to the `Snakefile <https://github.com/yourusername/micaflow/blob/main/micaflow/resources/Snakefile>`_ in the repository.