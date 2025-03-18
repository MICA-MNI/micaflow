# SynthSeg Registration

## Overview

A contrast-agnostic MRI registration tool that leverages SynthSeg brain parcellation to enable accurate registration between images with different contrasts (T1w to T2w, FLAIR to T1w, etc.).

## Description

Traditional intensity-based registration methods often struggle with multi-contrast registration because they assume similar intensity profiles between images. This script uses SynthSeg's deep learning-based parcellation to create contrast-independent label maps, which are then used for registration, achieving more robust results.

## Workflow

1. **Parcellation**: Generate brain segmentations of both input and reference images using SynthSeg
2. **Registration**: Register the parcellation volumes to each other (contrast-agnostic)
3. **Transformation**: Apply the resulting transformation to the original input image

## Requirements

- MicaFlow installed (`pip install -e .` from the MicaFlow repository)
- Python 3.9+
- Input and reference brain MRI images in NIfTI format

## Usage

```bash
python synthseg_registration.py --input <input_image.nii.gz> --reference <reference_image.nii.gz> --output <output_image.nii.gz> [--workdir <directory>]
```

### Arguments

- `--input`: Input image to be registered (any contrast)
- `--reference`: Reference/target image (any contrast)
- `--output`: Path where the registered image will be saved
- `--workdir`: (Optional) Directory for intermediate files (default: current directory)

### Example

```bash
python synthseg_registration.py \
  --input subject01_FLAIR.nii.gz \
  --reference template_T1w.nii.gz \
  --output subject01_FLAIR_registered.nii.gz \
  --workdir ./registration_workspace
```

## How It Works

The script leverages the fact that SynthSeg produces consistent segmentation labels regardless of input image contrast. The registration process:

1. Uses SynthSeg to generate parcellations of both the input and reference images
2. Performs registration between these parcellations using MicaFlow's `coregister` module
3. Applies the resulting transformation (affine + warp field) to the original input image using the `apply_warp` module

This approach is particularly effective when:
- Input and reference images have different contrasts
- Traditional intensity-based registration fails due to contrast differences
- Registering pathological images to healthy templates

## Intermediate Files

The script generates and saves several intermediate files in the specified working directory:
- `input_parcellation.nii.gz`: SynthSeg parcellation of the input image
- `reference_parcellation.nii.gz`: SynthSeg parcellation of the reference image
- `affine_transform.mat`: Affine transformation matrix
- `warp_field.nii.gz`: Non-linear deformation field
- `inverse_warp_field.nii.gz`: Inverse deformation field
- `registered_parcellation.nii.gz`: Registered parcellation result

## Notes

- For best results, input images should be brain-extracted
- Processing time depends on image resolution and computational resources
- Uses CPU by default; for GPU acceleration, ensure SynthSeg and ANTs are configured to use GPU

## References

- MicaFlow: https://github.com/MICA-Lab/micaflow
- SynthSeg: Billot, B., et al. (2023). SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining. Medical Image Analysis, 101672.