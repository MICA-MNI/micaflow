import ants
import argparse


def apply_warp(moving_file, reference_file, affine_file, warp_file, out_file):
    """Apply an affine transform and a warp field to a moving image.
    
    This function takes a moving image and applies both an affine transformation 
    and a nonlinear warp field to register it to a reference image space. The 
    transformation is applied using ANTsPy's apply_transforms function with the
    appropriate transform order.
    
    Parameters
    ----------
    moving_file : str
        Path to the moving image that will be transformed (.nii.gz).
    reference_file : str
        Path to the reference/fixed image that defines the target space (.nii.gz).
    affine_file : str
        Path to the affine transformation file (.mat).
    warp_file : str
        Path to the nonlinear warp field (.nii.gz).
    out_file : str
        Path where the transformed image will be saved.
        
    Returns
    -------
    None
        The function saves the transformed image to the specified output path
        but does not return any values.
        
    Notes
    -----
    The order of transforms matters: the warp field is applied first, followed 
    by the affine transformation. This is the standard order in ANTs for 
    composite transformations.
    """
    # Load images and transforms
    moving_img = ants.image_read(moving_file)
    reference_img = ants.image_read(reference_file)

    # The order of transforms in transformlist matters (last Transform will be applied first).
    # Usually you put the nonlinear warp first, then the affine:
    transformed = ants.apply_transforms(
        fixed=reference_img, moving=moving_img, transformlist=[warp_file, affine_file]
    )

    # Save the transformed image
    ants.image_write(transformed, out_file)
    print(f"Saved warped image as {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply an affine (.mat) and a warp field (.nii.gz) to an image using ANTsPy."
    )
    parser.add_argument(
        "--moving", required=True, help="Path to the moving image (.nii.gz)."
    )
    parser.add_argument(
        "--reference", required=True, help="Path to the reference image (.nii.gz)."
    )
    parser.add_argument(
        "--affine", required=True, help="Path to the affine transform (.mat)."
    )
    parser.add_argument(
        "--warp", required=True, help="Path to the warp field (.nii.gz)."
    )
    parser.add_argument(
        "--out", default="warped_image.nii.gz", help="Output warped image filename."
    )
    args = parser.parse_args()

    apply_warp(args.moving, args.reference, args.affine, args.warp, args.out)


if __name__ == "__main__":
    main()
