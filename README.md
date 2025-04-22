![MicaFlow logo](docs/source/_static/images/logo.png)
# MicaFlow: Accessible, fast, and pythonic MRI processing pipeline

MicaFlow is a comprehensive neuroimaging pipeline designed for processing structural and diffusion MRI data. It offers modular components that can be used as part of a cohesive workflow or as standalone tools.

## Features

- **Structural MRI Processing**: T1w and FLAIR image processing
- **Diffusion MRI Processing**: Complete pipeline for DWI processing
- **Brain Extraction**: HD-BET for robust skull stripping
- **Deep Learning Segmentation**: SynthSeg for brain segmentation and parcellation
- **Image Registration**: Multi-modal coregistration and spatial normalization to standard spaces
- **Texture Features**: Advanced texture feature generation
- **Quality Control**: Built-in QC metrics and visualization
- **Modular Design**: Components can be used independently or as a complete pipeline
- **Flexible Configuration**: Via command line arguments or YAML configuration files

## Installation

### Prerequisites

- Python 3.9, 3.10, or 3.11

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/MICA-Lab/micaflow.git
cd micaflow

# Install the package
pip install -e .

# Verify installation
micaflow --help
```

## Usage

MicaFlow can be used as a complete pipeline or as individual modules:

### Running the Full Pipeline

```bash
# Basic usage with T1w only
micaflow pipeline --subject sub-001 --session ses-01 \
  --data-directory /path/to/data --t1w-file sub-001_ses-01_T1w.nii.gz \
  --out-dir /output/path --cores 4 --cpu

# With FLAIR image
micaflow pipeline --subject sub-001 --session ses-01 \
  --data-directory /path/to/data --t1w-file sub-001_ses-01_T1w.nii.gz \
  --flair-file sub-001_ses-01_FLAIR.nii.gz --out-dir /output/path \
  --cores 4 --cpu

# With diffusion data
micaflow pipeline --subject sub-001 --session ses-01 \
  --data-directory /path/to/data --t1w-file sub-001_ses-01_T1w.nii.gz \
  --run-dwi --dwi-file sub-001_ses-01_dwi.nii.gz \
  --bval-file sub-001_ses-01_dwi.bval --bvec-file sub-001_ses-01_dwi.bvec \
  --inverse-dwi-file sub-001_ses-01_acq-PA_dwi.nii.gz \
  --out-dir /output/path --cores 4 --cpu
```

### Using Individual Modules

Each module can be used independently:

#### Brain Extraction (HD-BET)

```bash
micaflow bet --input t1w.nii.gz --output brain.nii.gz --output-mask mask.nii.gz
```

#### Brain Segmentation (SynthSeg)

```bash
micaflow synthseg --i t1w.nii.gz --o segmentation.nii.gz --parc --fast --threads 4
```

#### Image Registration

```bash
micaflow coregister --fixed-file target.nii.gz --moving-file source.nii.gz \
  --output registered.nii.gz --warp-file warp.nii.gz --affine-file affine.mat
```

#### Apply Transformations

```bash
micaflow apply_warp --moving image.nii.gz --reference target.nii.gz \
  --warp warp.nii.gz --affine affine.mat --output warped.nii.gz
```

#### Diffusion Processing

```bash
# Denoise DWI data
micaflow denoise --input dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --output denoised_dwi.nii.gz

# Motion correction
micaflow motion_correction --denoised denoised_dwi.nii.gz --bval dwi.bval --bvec dwi.bvec --output motion_corrected_dwi.nii.gz

# Susceptibility distortion correction
micaflow SDC --input motion_corrected_dwi.nii.gz --reverse-image reverse_phase_dwi.nii.gz \
  --output corrected_dwi.nii.gz --output-warp sdc_warpfield.nii.gz

# Compute DTI metrics
micaflow compute_fa_md --input preprocessed_dwi.nii.gz --bval dwi.bval --bvec dwi.bvec \
  --output-fa fa_map.nii.gz --output-md md_map.nii.gz
```

#### Texture Feature Extraction

```bash
micaflow texture_generation --input image.nii.gz --mask mask.nii.gz --output texture_features
```

## Pipeline Workflow

The pipeline performs the following processing steps:

1. **Brain Extraction**: Using HD-BET for T1w, FLAIR, and DWI
2. **Bias Field Correction**: Using N4 bias field correction
3. **Brain Segmentation**: Using SynthSeg for T1w and FLAIR
4. **Registration**: 
   - FLAIR to T1w space
   - T1w to MNI152 standard space
5. **DWI Processing** (if enabled):
   - Denoising with Patch2Self
   - Motion correction
   - Susceptibility distortion correction
   - Tensor fitting and DTI metrics calculation
   - Registration to T1w space
6. **Texture Feature Generation**: On normalized images
7. **Quality Control**: Calculating quality metrics

## Output Structure

The pipeline creates a structured output directory:

```
output/
├── <subject>/
│   └── <session>/
│       ├── anat/             # Anatomical images (brain-extracted, bias-corrected)
│       ├── dwi/              # Processed diffusion data and DTI metrics
│       ├── metrics/          # Quality metrics and DICE scores
│       ├── textures/         # Texture features
│       └── xfm/              # Transformation matrices and warps
```

## Configuration

MicaFlow can be configured via:

1. **Command Line Arguments**: For quick setup and individual module usage
2. **Configuration File**: YAML file for complex setups (specify with `--config-file`)
3. **Default Configuration**: Located in config.yaml

## System Requirements

- **CPU**: Multi-core recommended for parallel processing
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: Optional but recommended for faster processing (CUDA compatible)
- **Disk Space**: Depends on dataset size, minimum 10GB recommended

## Supported Image Formats

- NIfTI (.nii, .nii.gz)
- BIDS-compatible directory structures

## Support and Contact

For issues, questions or feature requests, please open an issue on the GitHub repository.