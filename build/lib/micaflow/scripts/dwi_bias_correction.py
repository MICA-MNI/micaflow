import ants
import numpy as np
import argparse


def run_bias_field_correction(image_path, mask_path, output):
    """
    Apply N4 bias field correction to each 3D volume (along the last axis).

    Parameters:
    - image_path: path to the input image.
    - mask_path: path to the mask image.

    Returns:
    - out_path: path to the bias-corrected image.
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

    ants.image_write(corrected_img, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply N4 bias field correction to each 3D volume of an image."
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the input image (NIfTI file)."
    )
    parser.add_argument(
        "--mask", type=str, required=True, help="Path to the mask image (NIfTI file)."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the bias-corrected image.",
    )

    args = parser.parse_args()
    out_path = run_bias_field_correction(args.image, args.mask, args.output)
    print("Bias-corrected image saved as:", out_path)
