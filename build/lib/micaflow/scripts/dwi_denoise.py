import argparse
import nibabel as nib
from dipy.denoise.patch2self import patch2self
from dipy.io.gradients import read_bvals_bvecs



def run_denoise(moving, moving_bval, moving_bvec, output):
    """Denoise diffusion-weighted images using the Patch2Self algorithm.
    
    This function applies Patch2Self denoising to diffusion-weighted images (DWI),
    which uses a self-supervised learning approach to remove noise while preserving 
    anatomical structure. It leverages redundant information across diffusion gradients.
    
    Args:
        moving (str): Path to the input DWI image (NIfTI file).
        moving_bval (str): Path to the b-values file (.bval).
        moving_bvec (str): Path to the b-vectors file (.bvec).
        output (str): Path where the denoised image will be saved.
        
    Returns:
        str: Path to the saved denoised image.
        
    Notes:
        The implementation uses an Ordinary Least Squares regression model,
        shifts intensity values to ensure positivity, and does not denoise
        b0 volumes separately. The b0 threshold is set to 50 s/mm².
    """
    moving_image = nib.load(moving)
    moving_bval_value, moving_bvec_value = read_bvals_bvecs(moving_bval, moving_bvec)
    denoised = patch2self(
        moving_image.get_fdata(),
        moving_bval_value,
        model="ols",
        shift_intensity=True,
        clip_negative_vals=False,
        b0_threshold=50,
        b0_denoising=False,
    )

    nib.save(nib.Nifti1Image(denoised, moving_image.affine), output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Denoise a DWI image using patch2self."
    )
    parser.add_argument(
        "--moving",
        type=str,
        required=True,
        help="Path to the input DWI image (NIfTI file).",
    )
    parser.add_argument(
        "--bval", type=str, required=True, help="Path to the bvals file."
    )
    parser.add_argument(
        "--bvec", type=str, required=True, help="Path to the bvecs file."
    )
    parser.add_argument("--output", type=str, required=True, help="output path")

    args = parser.parse_args()
    output_path = run_denoise(args.moving, args.bval, args.bvec, args.output)
    print("Denoised image saved as:", output_path)
