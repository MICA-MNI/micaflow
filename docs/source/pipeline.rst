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

      **Example Configuration File (config.yaml):**

      .. code-block:: yaml

         # Example configuration for micaflow pipeline
         subject: "sub-01"                             # Subject ID
         session: "ses-01"                             # Session ID
         
         # Input data paths
         t1w_file: "/data/bids/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"
         flair_file: "/data/bids/sub-01/ses-01/anat/sub-01_ses-01_FLAIR.nii.gz"
         dwi_file: "/data/bids/sub-01/ses-01/dwi/sub-01_ses-01_dir-AP_dwi.nii.gz"
         bval_file: "/data/bids/sub-01/ses-01/dwi/sub-01_ses-01_dir-AP_dwi.bval"
         bvec_file: "/data/bids/sub-01/ses-01/dwi/sub-01_ses-01_dir-AP_dwi.bvec"
         inverse_dwi_file: "/data/bids/sub-01/ses-01/dwi/sub-01_ses-01_dir-PA_dwi.nii.gz"
         
         # Output settings
         output: "/results/micaflow"
         data_directory: "/data/bids"
         
         # Processing options
         threads: 8               # Number of CPU threads to use
         cpu: true                # Force CPU processing (set to false for GPU)
         run_dwi: true            # Whether to process DWI data
         
         # Advanced options (optional)
         dry_run: false           # Just show commands without executing
         cleanup: true            # Remove temporary files after completion

      This configuration can be adapted for different subjects and datasets by updating
      the paths and processing parameters accordingly.
      
   .. tab:: Data Input Methods

      MICAflow supports two approaches for providing input data to the pipeline:

      **1. BIDS Directory Approach:**

      When using a BIDS-compliant dataset, you can specify the root directory and the pipeline will
      automatically locate files based on subject and session IDs:

      .. code-block:: bash

         micaflow pipeline \
           --subject sub-01 \
           --session ses-01 \
           --data-directory /path/to/bids/dataset \
           --output /path/to/output

      In this case, the pipeline will look for files at standard BIDS locations:
      
      .. code-block:: text
      
         /path/to/bids/dataset/
         └── sub-01/
             └── ses-01/
                 ├── anat/
                 │   ├── sub-01_ses-01_T1w.nii.gz
                 │   └── sub-01_ses-01_FLAIR.nii.gz
                 └── dwi/
                     ├── sub-01_ses-01_dwi.nii.gz
                     ├── sub-01_ses-01_dwi.bval
                     └── sub-01_ses-01_dwi.bvec

      **2. Direct File Path Approach:**

      For datasets that are not BIDS-compliant or when working with specific files,
      you can directly specify the path to each input file:

      .. code-block:: bash

         micaflow pipeline \
           --subject SUB001 \
           --session SES01 \
           --t1w-file /path/to/t1w.nii.gz \
           --flair-file /path/to/flair.nii.gz \
           --dwi-file /path/to/dwi.nii.gz \
           --output /path/to/output

      **Advantages of Each Approach:**

      - **BIDS Directory:** 
        - Simpler command line with fewer parameters
        - Automatic file discovery based on BIDS naming conventions
        - Better for batch processing multiple subjects with consistent naming
        
      - **Direct File Paths:**
        - Works with any directory structure
        - Flexible for non-standard file naming
        - Allows processing specific files from different locations

      Both approaches can be used in the YAML configuration file as well:

      .. code-block:: yaml
         
         # BIDS Directory approach
         subject: "sub-01"
         session: "ses-01"
         data_directory: "/path/to/bids/dataset"
         
         # OR Direct File Paths approach
         subject: "SUB001"
         session: "SES01"
         t1w_file: "/path/to/t1w.nii.gz"
         flair_file: "/path/to/flair.nii.gz"
         dwi_file: "/path/to/dwi.nii.gz"

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