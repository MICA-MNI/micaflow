"""
apply_SDC - Susceptibility Distortion Correction for diffusion MRI

Part of the micaflow processing pipeline for neuroimaging data.

This module applies susceptibility distortion correction (SDC) to diffusion MRI images
by using a pre-calculated displacement field to unwarp geometric distortions caused by
magnetic field inhomogeneities. These distortions occur along the phase-encoding
direction, which can be anterior-posterior (AP/PA), left-right (LR/RL), or 
superior-inferior (SI/IS).

The module works by:
1. Loading a distorted diffusion image (typically after motion correction)
2. Loading a 3D warp field containing displacement values (expected shape: nx, ny, nz)
3. Determining the phase-encoding direction from the provided argument
4. Applying the warp field to each 3D volume independently along the specified direction
5. Using linear interpolation (scipy.ndimage.map_coordinates with order=1) to resample
7. Saving the unwarped image with the specified affine transformation

API Usage:
---------
micaflow apply_SDC 
    --input <path/to/distorted_image.nii.gz>
    --warp <path/to/warp_field.nii.gz>
    --affine <path/to/reference_image.nii.gz>
    --phase-encoding <ap|pa|lr|rl|si|is>
    --output <path/to/corrected_output.nii.gz>

Python Usage:
-----------
>>> from micaflow.scripts.apply_SDC import apply_SD_correction
>>> apply_SD_correction(
...     motion_corr_path="distorted_image.nii.gz",
...     warp_field=warp_field_array,  # Must be 3D numpy array (nx, ny, nz)
...     moving_affine=affine_matrix,
...     output="corrected_output.nii.gz",
...     ped="ap"  # Phase encoding direction
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
    using a pre-calculated warp field. It takes a 4D diffusion image (typically
    motion-corrected) and applies the displacement field to each 3D volume 
    independently along the specified phase-encoding direction.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow apply_SDC {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--input{RESET}            : Path to the input 4D DWI image (.nii.gz)
                        Typically motion-corrected, but any 4D image is accepted
      {YELLOW}--warp{RESET}             : Path to the 3D warp field (.nii.gz)
                        Contains displacement values along the phase-encoding direction
                        Expected shape after squeezing: (nx, ny, nz)
      {YELLOW}--affine{RESET}           : Path to a reference image (.nii.gz)
                        Used to extract the affine matrix for the output image
      {YELLOW}--output{RESET}           : Output path for the corrected 4D image (.nii.gz)
    
    {CYAN}{BOLD}──────────────── OPTIONAL ARGUMENTS ─────────────────────{RESET}
      {YELLOW}--phase-encoding{RESET}   : Phase-encoding direction (default: ap)
                        Options: ap, pa, lr, rl, si, is
                        • ap/pa: Anterior-Posterior/Posterior-Anterior (y-axis)
                        • lr/rl: Left-Right/Right-Left (x-axis)
                        • si/is: Superior-Inferior/Inferior-Superior (z-axis)
    
    {CYAN}{BOLD}─────────────────── EXAMPLE USAGE ───────────────────────{RESET}
    
    {BLUE}# Apply SDC with anterior-posterior phase encoding (default){RESET}
    micaflow apply_SDC \\
      {YELLOW}--input{RESET} motion_corrected_dwi.nii.gz \\
      {YELLOW}--warp{RESET} sdc_warpfield.nii.gz \\
      {YELLOW}--affine{RESET} original_dwi.nii.gz \\
      {YELLOW}--output{RESET} corrected_dwi.nii.gz
    
    {BLUE}# Apply SDC with left-right phase encoding{RESET}
    micaflow apply_SDC \\
      {YELLOW}--input{RESET} motion_corrected_dwi.nii.gz \\
      {YELLOW}--warp{RESET} sdc_warpfield.nii.gz \\
      {YELLOW}--affine{RESET} original_dwi.nii.gz \\
      {YELLOW}--phase-encoding{RESET} lr \\
      {YELLOW}--output{RESET} corrected_dwi.nii.gz
    
    {CYAN}{BOLD}────────────────────────── NOTES ───────────────────────{RESET}
    {MAGENTA}•{RESET} The warp field contains displacement values along the phase-encoding direction
    {MAGENTA}•{RESET} Phase-encoding direction must match the direction used in SDC warp calculation
    {MAGENTA}•{RESET} Uses linear interpolation (scipy.ndimage.map_coordinates, order=1)
    {MAGENTA}•{RESET} Each 3D volume in the 4D input is warped independently
    {MAGENTA}•{RESET} Output affine is taken from the --affine reference image, not from --input
    {MAGENTA}•{RESET} Common phase-encoding directions:
              - AP (anterior to posterior): y-axis, posterior distortion
              - PA (posterior to anterior): y-axis, anterior distortion
              - LR (left to right): x-axis, right distortion
              - RL (right to left): x-axis, left distortion
    """
    print(help_text)


def apply_warpfield(data_array, warp_field, pe_dim=1):
    """
    Apply a warpfield to a 3D data array along the specified phase-encoding dimension.

    This function applies geometric distortion correction by warping a 3D volume
    using a displacement field. The warp is applied along the phase-encoding direction
    to correct susceptibility-induced distortions.

    Parameters
    ----------
    data_array : numpy.ndarray
        3D numpy array representing one volume (e.g., shape (nx, ny, nz))
    warp_field : numpy.ndarray
        3D numpy array of shape (nx, ny, nz) containing displacement values
        in voxels along the phase-encoding direction
    pe_dim : int, optional
        Phase-encoding dimension index (0=x-axis, 1=y-axis, 2=z-axis)
        Default is 1 (y-axis, typical for AP/PA encoding)

    Returns
    -------
    numpy.ndarray
        3D warped array with the same shape as data_array

    Notes
    -----
    - Uses linear interpolation (scipy.ndimage.map_coordinates with order=1)
    - Out-of-bounds values are handled with 'nearest' mode
    """
    nx, ny, nz = data_array.shape
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    coords = np.stack((grid_x, grid_y, grid_z), axis=-1).astype(
        np.float64
    )  # Ensure float64 type
    new_coords = coords.copy()
    
    # Apply displacement along the specified phase-encoding dimension
    new_coords[..., pe_dim] += warp_field
    
    # Rearrange shape to (3, nx, ny, nz) and flatten each
    new_coords = new_coords.transpose(3, 0, 1, 2)
    flat_coords = [c.flatten() for c in new_coords]
    warped_flat = map_coordinates(data_array, flat_coords, order=1)
    warped = warped_flat.reshape(data_array.shape)
    return warped

def get_pe_dimension(phase_encoding):
    """
    Get the dimension index for the phase-encoding direction.
    
    Parameters
    ----------
    phase_encoding : str
        Phase-encoding direction ('ap', 'pa', 'lr', 'rl', 'si', 'is')
        
    Returns
    -------
    int
        The dimension index (0=x, 1=y, 2=z) for the phase-encoding direction.
    """
    # Convert to lowercase and normalize direction
    pe = phase_encoding.lower()
    
    # Set dimensions based on phase-encoding direction
    if pe in ['lr', 'rl']:  # Left-Right or Right-Left (x-axis)
        return 0
    elif pe in ['ap', 'pa']:  # Anterior-Posterior or Posterior-Anterior (y-axis)
        return 1
    elif pe in ['si', 'is']:  # Superior-Inferior or Inferior-Superior (z-axis)
        return 2
    else:
        print(f"Warning: Unknown phase-encoding '{phase_encoding}', using default (y-axis)")
        return 1  # Default to y-axis (AP/PA)
    
def apply_SD_correction(motion_corr_path, warp_field, moving_affine, output, ped="ap"):
    """
    Apply susceptibility distortion correction by warping each 3D volume of the motion-corrected image.

    Parameters:
    - motion_corr_path: Path to the motion-corrected image (NIfTI file).
    - warp_field: Numpy array of shape (nx, ny, nz) representing the displacement field.
    - moving_affine: The affine matrix to use for the output NIfTI image.
    - output: Path where the corrected image will be saved.
    - ped: Phase encoding direction ('ap', 'pa', 'lr', 'rl', 'si', 'is'), default is "ap"

    Returns:
    - out_path: Path to the SD-corrected output image.
    """
    data_img = nib.load(motion_corr_path)
    data_arr = data_img.get_fdata()
    
    # Get the phase encoding dimension
    pe_dim = get_pe_dimension(ped)
    print(f"Using phase-encoding direction: {ped} (dimension: {pe_dim})")
    
    # Ensure the warpfield has the same dimensions as the image for all dimensions
    for dim in range(3):
        if warp_field.shape[dim] != data_arr.shape[dim]:
            print(f"Warning: Warp field dimension {dim} ({warp_field.shape[dim]}) doesn't match image ({data_arr.shape[dim]})")
            if warp_field.shape[dim] > data_arr.shape[dim]:
                # Crop to match
                if dim == 0:
                    warp_field = warp_field[:data_arr.shape[dim], :, :]
                elif dim == 1:
                    warp_field = warp_field[:, :data_arr.shape[dim], :]
                elif dim == 2:
                    warp_field = warp_field[:, :, :data_arr.shape[dim]]
                print(f"  Cropped warp field dimension {dim} to {data_arr.shape[dim]}")
            else:
                print(f"  ERROR: Warp field dimension {dim} is smaller than image dimension!")
    
    print(f"Final image shape: {data_arr.shape[:3]}, warp field shape: {warp_field.shape}")
    
    # Apply the correction to each volume using the correct phase encoding dimension
    transformed_vols = [
        apply_warpfield(data_arr[..., i], warp_field, pe_dim)
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
        "--phase-encoding",
        type=str,
        default="ap",
        choices=["ap", "pa", "lr", "rl", "si", "is"],
        help="Phase-encoding direction (default: ap)"
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
    warp_field = warp_img.get_fdata().squeeze()  # Expected shape: (nx, ny, nz)
    print("Warp field shape:", warp_field.shape)
    # Load the moving affine from given image
    moving_affine = nib.load(args.affine).affine

    out_path = apply_SD_correction(
        args.input, warp_field, moving_affine, args.output, ped=args.phase_encoding
    )
    print(f"SD-corrected image saved as: {out_path} (phase encoding: {args.phase_encoding})")