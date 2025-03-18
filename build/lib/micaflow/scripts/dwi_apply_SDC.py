import argparse
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates


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


def apply_topup_correction(motion_corr_path, warp_field, moving_affine, output):
    """
    Apply topup correction by warping each 3D volume of the motion-corrected image along the y-axis.

    Parameters:
    - motion_corr_path: Path to the motion-corrected image (NIfTI file).
    - warp_field: Numpy array of shape (nx, ny, nz) representing the displacement field along the y-axis.
    - moving_affine: The affine matrix to use for the output NIfTI image.

    Returns:
    - out_path: Path to the topup-corrected output image.
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
    topup_corrected = np.stack(transformed_vols, axis=-1)
    nib.save(nib.Nifti1Image(topup_corrected, moving_affine), output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply topup correction using a warp field along the y-axis."
    )
    parser.add_argument(
        "--motion_corr",
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
        help="Output path for the topup-corrected image.",
    )

    args = parser.parse_args()

    # Load warp field as a numpy displacement field
    warp_img = nib.load(args.warp)
    warp_field = warp_img.get_fdata()  # Expected shape: (nx, ny, nz)

    # Load the moving affine from given image
    moving_affine = nib.load(args.affine).affine

    out_path = apply_topup_correction(
        args.motion_corr, warp_field, moving_affine, args.output
    )
    print("Topup-corrected image saved as:", out_path)
