"""
apply_SDC - Susceptibility Distortion Correction for diffusion MRI

Part of the micaflow processing pipeline for neuroimaging data.

This module applies susceptibility distortion correction (SDC) to diffusion MRI images
by using a pre-calculated displacement field to unwarp geometric distortions caused by
magnetic field inhomogeneities. These distortions typically occur along the phase-encoding
direction (usually the y-axis).

The module works by:
1. Loading a distorted diffusion image (typically after motion correction)
2. Applying a voxel-wise displacement field to each volume in the 4D image
3. Using linear interpolation to resample the image at the corrected coordinates
4. Saving the unwarped image with the original affine transformation

API Usage:
---------
micaflow apply_SDC 
    --input <path/to/distorted_image.nii.gz>
    --warp <path/to/field_map.nii.gz>
    --affine <path/to/reference_image.nii.gz>
    --output <path/to/corrected_output.nii.gz>

Python Usage:
-----------
>>> from micaflow.scripts.apply_SDC import apply_SD_correction
>>> apply_SD_correction(
...     motion_corr_path="distorted_image.nii.gz",
...     warp_field=warp_field_array,
...     moving_affine=affine_matrix,
...     output="corrected_output.nii.gz"
... )

"""
import argparse
import nibabel as nib
import numpy as np
import sys
from scipy.ndimage import map_coordinates
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
    ║          APPLY SUSCEPTIBILITY DISTORTION CORRECTION            ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script applies susceptibility distortion correction to diffusion images
    using a pre-calculated warp field. It takes a motion-corrected diffusion image
    and applies the warp field to each 3D volume along the y-axis.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow apply_SDC {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--input{RESET}       : Path to the motion-corrected DWI image (.nii.gz)
      {YELLOW}--warp{RESET}        : Path to the warp field estimated from SDC (.nii.gz)
      {YELLOW}--affine{RESET}      : Path to an image from which to extract the affine matrix
      {YELLOW}--output{RESET}      : Output path for the corrected image
    
    {CYAN}{BOLD}─────────────────── EXAMPLE USAGE ───────────────────────{RESET}
    
    {BLUE}# Apply SDC to a motion-corrected DWI image{RESET}
    micaflow apply_SDC \\
      {YELLOW}--input{RESET} subj_motion_corrected.nii.gz \\
      {YELLOW}--warp{RESET} SDC.nii.gz \\
      {YELLOW}--affine{RESET} original_dwi.nii.gz \\
      {YELLOW}--output{RESET} corrected_dwi.nii.gz
    
    {CYAN}{BOLD}────────────────────────── NOTES ───────────────────────{RESET}
    {MAGENTA}•{RESET} The warp field should contain displacement values along the y-axis
    {MAGENTA}•{RESET} This implementation assumes that susceptibility distortions are primarily 
      in the phase-encoding direction (typically y-axis)
    """
    print(help_text)


def apply_warpfield_y(data_array, warp_field):
    """
    Apply a warpfield to a 3D data array along the second dimension (y-axis) using linear interpolation.

    Parameters:
    - data_array: 3D numpy array (e.g. one volume)
    - warp_field: 3D numpy array of shape (nx, ny, nz) with displacement values along the y-axis.

    Returns:
    - warped: 3D numpy array after applying warp_field.
    """
    nx, ny, nz = data_array.shape
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    coords = np.stack((grid_x, grid_y, grid_z), axis=-1).astype(
        np.float64
    )  # Ensure float64 type
    new_coords = coords.copy()
    new_coords[..., 1] += warp_field  # Apply displacement along the y-axis
    # Rearrange shape to (3, nx, ny, nz) and flatten each
    new_coords = new_coords.transpose(3, 0, 1, 2)
    flat_coords = [c.flatten() for c in new_coords]
    warped_flat = map_coordinates(data_array, flat_coords, order=1)
    warped = warped_flat.reshape(data_array.shape)
    return warped


def apply_SD_correction(motion_corr_path, warp_field, moving_affine, output):
    """
    Apply susceptibility distortion correction by warping each 3D volume of the motion-corrected image along the y-axis.

    Parameters:
    - motion_corr_path: Path to the motion-corrected image (NIfTI file).
    - warp_field: Numpy array of shape (nx, ny, nz) representing the displacement field along the y-axis.
    - moving_affine: The affine matrix to use for the output NIfTI image.
    - output: Path where the corrected image will be saved.

    Returns:
    - out_path: Path to the SD-corrected output image.
    """
    data_img = nib.load(motion_corr_path)
    data_arr = data_img.get_fdata()
    # Ensure the warpfield has the same dimensions as the image
    if warp_field.shape[1] > data_arr.shape[1]:
        warp_field = warp_field[:, : data_arr.shape[1], :]
    transformed_vols = [
        apply_warpfield_y(data_arr[..., i], warp_field)
        for i in range(data_arr.shape[-1])
    ]
    SD_corrected = np.stack(transformed_vols, axis=-1)
    nib.save(nib.Nifti1Image(SD_corrected, moving_affine), output)
    return output


if __name__ == "__main__":
    # Print help message if no arguments provided
    if len(sys.argv) == 1 or '-h' in sys.argv or '--help' in sys.argv:
        print_help_message()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Apply susceptibility distortion correction using a warp field along the y-axis."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the motion-corrected image (NIfTI file).",
    )
    parser.add_argument(
        "--warp",
        type=str,
        required=True,
        help="Path to the warp field (NIfTI file containing the displacement field).",
    )
    parser.add_argument(
        "--affine",
        type=str,
        required=True,
        help="Path to an image (NIfTI file) from which to extract the moving affine.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the SD-corrected image.",
    )

    args = parser.parse_args()

    # Load warp field as a numpy displacement field
    warp_img = nib.load(args.warp)
    warp_field = warp_img.get_fdata()  # Expected shape: (nx, ny, nz)

    # Load the moving affine from given image
    moving_affine = nib.load(args.affine).affine

    out_path = apply_SD_correction(
        args.input, warp_field, moving_affine, args.output
    )
    print("SD-corrected image saved as:", out_path)