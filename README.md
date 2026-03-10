![MicaFlow logo](https://raw.githubusercontent.com/MICA-MNI/micaflow/main/docs/source/_static/images/logo.png)
# MicaFlow: Accessible, fast, and pythonic MRI processing pipeline

<div align="left">

[![Version](https://img.shields.io/github/v/tag/MICA-MNI/micaflow)](https://github.com/MICA-MNI/micaflow)
[![PyPI version](https://img.shields.io/pypi/v/micaflow.svg)](https://pypi.org/project/micaflow/)
[![PyPI downloads](https://img.shields.io/pypi/dm/micaflow.svg)](https://pypi.org/project/micaflow/)
[![GitHub issues](https://img.shields.io/github/issues/MICA-MNI/micaflow?color=brightgreen)](https://github.com/MICA-MNI/micaflow/issues)
[![GitHub stars](https://img.shields.io/github/stars/MICA-MNI/micaflow.svg?style=flat&label=%E2%AD%90%EF%B8%8F%20stars&color=brightgreen)](https://github.com/MICA-MNI/micaflow/stargazers)

</div>

MicaFlow is a comprehensive neuroimaging pipeline designed for processing structural and diffusion MRI data. It offers modular components that can be used as part of a cohesive workflow or as standalone tools.or processing structural and diffusion MRI data. It offers modular components that can be used as part of a cohesive workflow or as standalone tools.

## Features

- **Texture Features**: Advanced texture feature generation
- **Quality Control**: Built-in QC metrics and visualization
- **Batch Processing**: Automated BIDS directory scanning and processing- **Brain Extraction**: SynthSeg-based brain extraction with optional cerebellum removal
- **Brain-Extracted Outputs**: Optional dedicated directory for all skull-stripped imagesng Segmentation**: SynthSeg for brain segmentation and parcellation
- **Temporary File Management**: Option to preserve intermediate files for debugging- **Image Registration**: Multi-modal coregistration and spatial normalization to standard spaces
- **Modular Design**: Components can be used independently or as a complete pipelineres**: Advanced texture feature generation
- **Flexible Configuration**: Via command line arguments or YAML configuration files- **Quality Control**: Built-in QC metrics and visualization
mated BIDS directory scanning and processing
## Installation- **Brain-Extracted Outputs**: Optional dedicated directory for all skull-stripped images
agement**: Option to preserve intermediate files for debugging
### Prerequisites- **Modular Design**: Components can be used independently or as a complete pipeline
xible Configuration**: Via command line arguments or YAML configuration files
- Python 3.10, 3.11, or 3.12

### Installation Steps
### Prerequisites
```bash
# Clone the repository.11, or 3.12
git clone https://github.com/MICA-MNI/micaflow.git
cd micaflows

# Install the packagebash
pip install -e .# Clone the repository
e https://github.com/MICA-MNI/micaflow.git
# Verify installationcd micaflow
micaflow --help
```# Install the package

## Usage
y installation
MicaFlow can be used as a complete pipeline or as individual modules:

### Running the Full Pipeline

```bash
# Basic usage with T1w onlyed as a complete pipeline or as individual modules:
micaflow pipeline --subject sub-001 --session ses-01 \
  --data-directory /path/to/data --t1w-file sub-001_ses-01_T1w.nii.gz \
  --output /output/path --cores 4

# With FLAIR image# Basic usage with T1w only
micaflow pipeline --subject sub-001 --session ses-01 \ubject sub-001 --session ses-01 \
  --data-directory /path/to/data --t1w-file sub-001_ses-01_T1w.nii.gz \s-01_T1w.nii.gz \
  --flair-file sub-001_ses-01_FLAIR.nii.gz --output /output/path \
  --cores 4

# With diffusion data
micaflow pipeline --subject sub-001 --session ses-01 \--t1w-file sub-001_ses-01_T1w.nii.gz \
  --data-directory /path/to/data --t1w-file sub-001_ses-01_T1w.nii.gz \-flair-file sub-001_ses-01_FLAIR.nii.gz --output /output/path \
  --dwi-file sub-001_ses-01_dwi.nii.gz \  --cores 4
  --bval-file sub-001_ses-01_dwi.bval --bvec-file sub-001_ses-01_dwi.bvec \
  --inverse-dwi-file sub-001_ses-01_acq-PA_dwi.nii.gz \# With diffusion data
  --output /output/path --cores 4
```  --data-directory /path/to/data --t1w-file sub-001_ses-01_T1w.nii.gz \
-file sub-001_ses-01_dwi.nii.gz \
### Batch Processing (BIDS)
ile sub-001_ses-01_acq-PA_dwi.nii.gz \
To process an entire BIDS dataset automatically using the batch command:-output /output/path --cores 4
```
```bash
micaflow bids --bids-dir /path/to/bids_root --output-dir /path/to/derivatives \
  --cores 4 --gpu
```e batch command:

This command will:```bash
1. Scan the BIDS directory for valid subjects and sessions.ds-dir /path/to/bids_root --output-dir /path/to/derivatives \
2. Automatically identify T1w, FLAIR (optional), and DWI (optional) files based on suffixes.
3. Run the pipeline sequentially for each session found.
4. Generate a `micaflow_runs_summary.json` in the output directory tracking execution status.

**key arguments:**
- `--bids-dir`: Root path to the BIDS dataset.2. Automatically identify T1w, FLAIR (optional), and DWI (optional) files based on suffixes.
- `--output-dir`: Path where derivatives will be saved.ally for each session found.
- `--participant-label`: (Optional) Space-separated list of subject IDs to process (e.g., `001 002`).4. Generate a `micaflow_runs_summary.json` in the output directory tracking execution status.
- `--session-label`: (Optional) Space-separated list of session IDs to process.
- `--t1w-suffix`, `--dwi-suffix`, etc.: Customize matching patterns for input files.**key arguments:**
path to the BIDS dataset.
### Using Individual Modules- `--output-dir`: Path where derivatives will be saved.
rticipant-label`: (Optional) Space-separated list of subject IDs to process (e.g., `001 002`).
Each module can be used independently:
--t1w-suffix`, `--dwi-suffix`, etc.: Customize matching patterns for input files.
#### Brain Extraction

```bash
micaflow bet --input t1w.nii.gz --output brain.nii.gz --parcellation segmentation.nii.gz --output-mask mask.nii.gzdule can be used independently:
```
# Brain Extraction
#### Brain Segmentation (SynthSeg)

```bashmicaflow bet --input t1w.nii.gz --output brain.nii.gz --parcellation segmentation.nii.gz --output-mask mask.nii.gz
micaflow synthseg --i t1w.nii.gz --o segmentation.nii.gz --parc --fast --threads 4
```

#### Image Registration
```bash
```bashnii.gz --o segmentation.nii.gz --parc --fast --threads 4
micaflow coregister --fixed-file target.nii.gz --moving-file source.nii.gz \```
  --output registered.nii.gz --warp-file warp.nii.gz --affine-file affine.mat
```

#### Apply Transformationsbash
micaflow coregister --fixed-file target.nii.gz --moving-file source.nii.gz \
```bash.gz --warp-file warp.nii.gz --affine-file affine.mat
micaflow apply_warp --moving image.nii.gz --reference target.nii.gz \```
  --warp warp.nii.gz --affine affine.mat --output warped.nii.gz
```rmations

#### Diffusion Processing```bash
 --moving image.nii.gz --reference target.nii.gz \
```bash
# Denoise DWI data```
micaflow denoise --input dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --output denoised_dwi.nii.gz

# Motion correction
micaflow motion_correction --denoised denoised_dwi.nii.gz --input-bvecs dwi.bvec --output-bvecs corrected.bvec --output motion_corrected_dwi.nii.gz```bash

# Susceptibility distortion correctioni.nii.gz
micaflow SDC --input motion_corrected_dwi.nii.gz --reverse-image reverse_phase_dwi.nii.gz \
  --output corrected_dwi.nii.gz --output-warp sdc_warpfield.nii.gzotion correction
micaflow motion_correction --denoised denoised_dwi.nii.gz --input-bvecs dwi.bvec --output-bvecs corrected.bvec --output motion_corrected_dwi.nii.gz
# Compute DTI metrics
micaflow compute_fa_md --input preprocessed_dwi.nii.gz --bval dwi.bval --bvec dwi.bvec \# Susceptibility distortion correction
  --output-fa fa_map.nii.gz --output-md md_map.nii.gzw SDC --input motion_corrected_dwi.nii.gz --reverse-image reverse_phase_dwi.nii.gz \
```

#### Texture Feature Extraction# Compute DTI metrics
md --input preprocessed_dwi.nii.gz --bval dwi.bval --bvec dwi.bvec \
```bash  --output-fa fa_map.nii.gz --output-md md_map.nii.gz
micaflow texture_generation --input image.nii.gz --mask mask.nii.gz --output texture_features
```

## Pipeline Workflow

The pipeline performs the following processing steps:k.nii.gz --output texture_features

1. **Brain Extraction**: Using SynthSeg-based segmentation for T1w, FLAIR, and DWI
   - Optionally remove cerebellum with `--rm-cerebellum`
2. **Bias Field Correction**: Using N4 bias field correction
3. **Brain Segmentation**: Using SynthSeg for T1w and FLAIRowing processing steps:
4. **Registration**: 
   - FLAIR to T1w spaceased segmentation for T1w, FLAIR, and DWI
   - T1w to MNI152 standard spacerebellum`
5. **DWI Processing** (if enabled):Using N4 bias field correction
   - Denoising with Patch2SelfLAIR
   - Motion correction
   - Susceptibility distortion correction
   - Tensor fitting and DTI metrics calculation
   - Registration to T1w space
6. **Texture Feature Generation**: On normalized images
7. **Brain-Extracted Outputs** (if `--extract-brain` enabled):   - Motion correction
   - Creates skull-stripped versions of all outputs in dedicated directory distortion correction
   - Normalized versions also created   - Tensor fitting and DTI metrics calculation
8. **Quality Control**: Calculating quality metrics
9. **Cleanup**: Removes temporary files (unless `--keep-temp` is specified)6. **Texture Feature Generation**: On normalized images
**Brain-Extracted Outputs** (if `--extract-brain` enabled):
## Output Structureeates skull-stripped versions of all outputs in dedicated directory
d versions also created
The pipeline creates a structured output directory:ol**: Calculating quality metrics

```
output/
├── <subject>/
│   └── <session>/
│       ├── anat/             # Anatomical images (brain-extracted, bias-corrected)
│       ├── dwi/              # Processed diffusion data and DTI metrics```
│       ├── metrics/          # Quality metrics and DICE scores
│       ├── textures/         # Texture features├── <subject>/
│       └── xfm/              # Transformation matrices and warps
```│       ├── anat/             # Anatomical images (brain-extracted, bias-corrected)

## Configuration

MicaFlow can be configured via:│       └── xfm/              # Transformation matrices and warps

1. **Command Line Arguments**: For quick setup and individual module usage
2. **Configuration File**: YAML file for complex setups (specify with `--config-file`)
3. **Default Configuration**: Located in config.yaml

## System Requirements
1. **Command Line Arguments**: For quick setup and individual module usage
- **CPU**: Multi-core recommended for parallel processing YAML file for complex setups (specify with `--config-file`)
- **RAM**: 8GB minimum, 16GB+ recommended3. **Default Configuration**: Located in config.yaml
- **GPU**: Optional but recommended for faster processing (CUDA compatible)
- **Disk Space**: Depends on dataset size, minimum 10GB recommended

## Supported Image Formatsrecommended for parallel processing
- **RAM**: 8GB minimum, 16GB+ recommended
- NIfTI (.nii, .nii.gz)





For issues, questions or feature requests, please open an issue on the GitHub repository.## Support and Contact- BIDS-compatible directory structures- **Disk Space**: Depends on dataset size, minimum 10GB recommended

## Supported Image Formats

- NIfTI (.nii, .nii.gz)
- BIDS-compatible directory structures

## Support and Contact

For issues, questions or feature requests, please open an issue on the GitHub repository.