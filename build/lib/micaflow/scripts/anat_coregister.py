import ants
import argparse
import shutil


def ants_linear_nonlinear_registration(
    fixed_file,
    moving_file,
    out_file="registered_image.nii",
    warp_file=None,
    affine_file=None,
    rev_warp_file=None,
    rev_affine_file=None,
):
    """Perform linear (rigid + affine) and nonlinear registration using ANTsPy.
    
    This function performs registration between two images using ANTs' SyNRA transform, 
    which includes both linear (rigid + affine) and nonlinear (SyN) components. 
    The registered image is saved to the specified output path, and the transform 
    files can optionally be saved as well.
    
    Args:
        fixed_file (str): Path to the fixed/reference image.
        moving_file (str): Path to the moving image that will be registered.
        out_file (str, optional): Path where the registered image will be saved. 
            Defaults to "registered_image.nii".
        warp_file (str, optional): Path to save the forward warp field. 
            Defaults to None.
        affine_file (str, optional): Path to save the forward affine transform. 
            Defaults to None.
        rev_warp_file (str, optional): Path to save the reverse warp field. 
            Defaults to None.
        rev_affine_file (str, optional): Path to save the reverse affine transform. 
            Defaults to None.
            
    Returns:
        None: The function saves the registered image and transform files to disk
        but does not return any values.
    """
    # Load images
    fixed = ants.image_read(fixed_file)
    moving = ants.image_read(moving_file)

    # 'SyN' transform includes both linear and nonlinear registration.
    transforms = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyNRA")  

    # The result of the registration is a dictionary containing, among other keys:
    registered = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=transforms["fwdtransforms"], interpolator="nearestNeighbor")
    
    # Save the registered moving image
    ants.image_write(registered, out_file)
    print(f"Registration complete. Saved registered image as {out_file}")

    # If specified, save the transform files
    # Typically, transforms["fwdtransforms"][0] is the warp field, and [1] is the affine.
    if warp_file:
        shutil.copyfile(transforms["fwdtransforms"][0], warp_file)
        print(f"Saved warp field as {warp_file}")
    if affine_file:
        shutil.copyfile(transforms["fwdtransforms"][1], affine_file)
        print(f"Saved affine transform as {affine_file}")
    if rev_warp_file:
        shutil.copyfile(transforms["invtransforms"][0], rev_warp_file)
        print(f"Saved reverse warp field as {rev_warp_file}")
    if rev_affine_file:
        shutil.copyfile(transforms["invtransforms"][1], rev_affine_file)
        print(f"Saved reverse affine transform as {rev_affine_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run linear + nonlinear (SyN) registration using ANTsPy."
    )
    parser.add_argument("--fixed-file", required=True, help="Path to the fixed image.")
    parser.add_argument(
        "--moving-file", required=True, help="Path to the moving image."
    )
    parser.add_argument(
        "--out-file",
        default="registered_image.nii",
        help="Output path for the registered image.",
    )
    parser.add_argument(
        "--warp-file", default=None, help="Optional path to save the warp field."
    )
    parser.add_argument(
        "--affine-file",
        default=None,
        help="Optional path to save the affine transform.",
    )
    parser.add_argument(
        "--rev-warp-file",
        default=None,
        help="Optional path to save the reverse warp field.",
    )
    parser.add_argument(
        "--rev-affine-file",
        default=None,
        help="Optional path to save the reverse affine transform.",
    )
    args = parser.parse_args()

    ants_linear_nonlinear_registration(
        args.fixed_file,
        args.moving_file,
        out_file=args.out_file,
        warp_file=args.warp_file,
        affine_file=args.affine_file,
        rev_warp_file=args.rev_warp_file,
        rev_affine_file=args.rev_affine_file,
    )
    print(f"Registration complete. Saved as {args.out_file}")
