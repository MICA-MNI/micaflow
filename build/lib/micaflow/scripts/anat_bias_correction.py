"""N4 Bias Field Correction script for MR images.

This script provides functionality to correct intensity non-uniformity (bias field)
in MR images using the N4 algorithm from the Advanced Normalization Tools (ANTs) library.
The N4 algorithm is an improved version of the popular N3 bias field correction method
that is more robust to different scan types and field strengths.

Example:
    python N4BiasFieldCorrection.py -i input_image.nii.gz -o corrected_image.nii.gz
    python N4BiasFieldCorrection.py -i input_image.nii.gz -o corrected_image.nii.gz -m brain_mask.nii.gz

"""

import ants
import argparse


def bias_field_correction(image, output, mask=None):
    """Perform N4 bias field correction on a medical image.
    
    This function applies the N4 bias field correction algorithm to correct intensity 
    non-uniformity in MR images. If no mask is provided, an automated brain mask is 
    generated from the input image.
    
    Args:
        image (str): Path to the input image file.
        output (str): Path where the bias-corrected image will be saved.
        mask (str, optional): Path to a brain mask image file. If not provided,
            a mask will be automatically generated. Defaults to None.
    
    Returns:
        None: The function saves the corrected image to the specified output path
        but does not return any values.
    
    Notes:
        This function uses the ANTsPy library's implementation of the N4 algorithm,
        which is an improved version of the N3 bias field correction method.
    """
    img = ants.image_read(image)
    mask_img = ants.image_read(mask) if mask else ants.get_mask(img)
    corrected_img = ants.n4_bias_field_correction(img, mask_img)
    ants.image_write(corrected_img, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform N4 Bias Field Correction")
    parser.add_argument("--input", "-i", required=True, help="Input image file")
    parser.add_argument(
        "--output", "-o", required=True, help="Output corrected image file"
    )
    parser.add_argument("-m", "--mask", help="Brain mask file (optional)")
    args = parser.parse_args()
    bias_field_correction(args.input, args.output, args.mask)