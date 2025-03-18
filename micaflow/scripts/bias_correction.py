"""
bias_correction - N4 Bias Field Correction for MRI data

Part of the micaflow processing pipeline for neuroimaging data.

This module corrects intensity non-uniformity (bias field) in MR images using the 
N4 algorithm from Advanced Normalization Tools (ANTs). Intensity bias appears as a 
smooth variation of signal intensity across the image and can affect subsequent analysis 
steps like segmentation or registration. The N4 algorithm estimates this bias field 
and removes it, producing more uniform intensities across tissues.

Features:
--------
- Supports both 3D anatomical images and 4D diffusion-weighted images
- Automatic detection of image dimensionality (3D vs 4D)
- Optional brain mask input for improved correction accuracy
- Volume-by-volume processing for 4D images preserves temporal dynamics
- Maintains image header information in the corrected output

API Usage:
---------
micaflow bias_correction 
    --input <path/to/image.nii.gz>
    --output <path/to/corrected.nii.gz>
    [--mask <path/to/brain_mask.nii.gz>]
    [--mode <3d|4d|auto>]

Python Usage:
-----------
>>> from micaflow.scripts.bias_correction import run_bias_field_correction
>>> run_bias_field_correction(
...     image_path="t1w.nii.gz",
...     output_path="corrected_t1w.nii.gz",
...     mask_path="brain_mask.nii.gz",  # optional for 3D images
...     mode="auto"  # auto, 3d, or 4d
... )

"""

import ants
import numpy as np
import argparse
import sys
from colorama import init, Fore, Style

init()

def print_help_message():
    """Print an extended help message with examples."""
    # ANSI color codes
    CYAN = Fore.CYAN
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    BLUE = Fore.BLUE
    MAGENTA = Fore.MAGENTA
    BOLD = Style.BRIGHT
    RESET = Style.RESET_ALL
    
    help_text = f"""
    {CYAN}{BOLD}╔════════════════════════════════════════════════════════════════╗
    ║                    N4 BIAS FIELD CORRECTION                    ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script corrects intensity non-uniformity (bias field) in MR images 
    using the N4 algorithm from ANTs. It supports both 3D anatomical images 
    and 4D diffusion-weighted images.
    
    {CYAN}{BOLD}──────────────────── REQUIRED ARGUMENTS ────────────────────{RESET}
      {YELLOW}--input{RESET}, {YELLOW}-i{RESET}    : Path to the input image (.nii.gz)
      {YELLOW}--output{RESET}, {YELLOW}-o{RESET}   : Path for the output bias-corrected image (.nii.gz)
    
    {CYAN}{BOLD}──────────────────── OPTIONAL ARGUMENTS ────────────────────{RESET}
      {YELLOW}--mask{RESET}, {YELLOW}-m{RESET}     : Path to a brain mask image (required for 4D images)
      {YELLOW}--mode{RESET}         : Processing mode: 3d, 4d, or auto (default: auto)
    
    {CYAN}{BOLD}──────────────────── EXAMPLE USAGE ────────────────────{RESET}
    
    {BLUE}# For anatomical (3D) images:{RESET}
    micaflow bias_correction \\
      {YELLOW}--input{RESET} t1w.nii.gz \\
      {YELLOW}--output{RESET} corrected_t1w.nii.gz
    
    {BLUE}# For diffusion (4D) images with mask:{RESET}
    micaflow bias_correction \\
      {YELLOW}--input{RESET} dwi.nii.gz \\
      {YELLOW}--output{RESET} corrected_dwi.nii.gz \\
      {YELLOW}--mask{RESET} brain_mask.nii.gz \\
      {YELLOW}--mode{RESET} 4d
    
    {CYAN}{BOLD}────────────────────────── NOTES ───────────────────────{RESET}
    {MAGENTA}•{RESET} In 'auto' mode, the script detects whether the input is 3D or 4D
    {MAGENTA}•{RESET} For 3D images, a mask is optional (one will be generated if not provided)
    {MAGENTA}•{RESET} For 4D images, a mask is required
    {MAGENTA}•{RESET} 4D processing applies the correction to each volume separately
    """
    print(help_text)
    
def bias_field_correction_3d(image_path, output_path, mask_path=None):
    """Perform N4 bias field correction on a 3D medical image.
    
    Args:
        image_path (str): Path to the input image file.
        output_path (str): Path where the bias-corrected image will be saved.
        mask_path (str, optional): Path to a brain mask image file. If not provided,
            a mask will be automatically generated.
    
    Returns:
        str: Path to the output corrected image.
    """
    img = ants.image_read(image_path)
    mask_img = ants.image_read(mask_path) if mask_path else ants.get_mask(img)
    corrected_img = ants.n4_bias_field_correction(img, mask=mask_img)
    ants.image_write(corrected_img, output_path)
    return output_path


def bias_field_correction_4d(image_path, mask_path, output_path):
    """Apply N4 bias field correction to each 3D volume of a 4D image.

    Parameters:
        image_path (str): Path to the input 4D image.
        mask_path (str): Path to the mask image (must be 3D).
        output_path (str): Path for the output bias-corrected image.

    Returns:
        str: Path to the output corrected image.
    """
    img = ants.image_read(image_path)
    mask_ants = ants.image_read(mask_path)
    img_data = img.numpy()

    corrected_vols = []
    for i in range(img_data.shape[-1]):
        vol = img_data[..., i]
        vol_ants = ants.from_numpy(
            vol,
            spacing=img.spacing[:3],
            origin=img.origin[:3],
            direction=img.direction[:3, :3],
        )
        corrected_vol_ants = ants.n4_bias_field_correction(vol_ants, mask=mask_ants)
        corrected_vols.append(corrected_vol_ants.numpy())
    
    corrected_array = np.stack(corrected_vols, axis=-1)
    corrected_img = ants.from_numpy(
        corrected_array, spacing=img.spacing, origin=img.origin, direction=img.direction
    )

    ants.image_write(corrected_img, output_path)
    return output_path


def run_bias_field_correction(image_path, output_path, mask_path=None, mode="auto"):
    """
    Run bias field correction on an image, automatically detecting dimensionality.
    
    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path for the output bias-corrected image.
        mask_path (str, optional): Path to mask (required for 4D images).
        mode (str): Processing mode: "3d", "4d", or "auto" (detect automatically).
    
    Returns:
        str: Path to the output corrected image.
    """
    # If auto mode, determine if image is 3D or 4D
    if mode == "auto":
        img = ants.image_read(image_path)
        dims = img.shape
        mode = "4d" if (len(dims) > 3 and dims[3] > 1) else "3d"
    
    # Process according to mode
    if mode == "4d":
        if not mask_path:
            raise ValueError("4D images require a mask. Please provide a mask with --mask.")
        return bias_field_correction_4d(image_path, mask_path, output_path)
    else:  # 3d
        return bias_field_correction_3d(image_path, output_path, mask_path)


if __name__ == "__main__":
    # Check if no arguments provided or help requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="N4 Bias Field Correction for 3D anatomical and 4D diffusion MR images",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input", "-i", required=True, 
        help="Path to the input image (NIfTI file)."
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output path for the bias-corrected image."
    )
    parser.add_argument(
        "--mask", "-m",
        help="Path to a mask image (required for 4D images, optional for 3D)."
    )
    parser.add_argument(
        "--mode", choices=["3d", "4d", "auto"], default="auto",
        help="Processing mode: 3d=anatomical, 4d=diffusion, auto=detect (default)"
    )

    args = parser.parse_args()
    
    out_path = run_bias_field_correction(
        args.input, 
        args.output, 
        args.mask, 
        args.mode
    )
    
    print(f"Bias-corrected image saved as: {out_path}")