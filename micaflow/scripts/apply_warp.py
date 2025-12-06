"""
apply_warp - Image registration transformation application

Part of the micaflow processing pipeline for neuroimaging data.

This module applies spatial transformations to register images from one space to another
using affine and/or non-linear (warp field) transformations. It's commonly used to:
- Transform subject images to a standard space (e.g., MNI152)
- Apply previously calculated transformations to derived images (e.g., segmentations)
- Chain multiple transformations together (e.g., subject→intermediate→template space)

The module leverages ANTsPy to apply the transformations in the correct order, supporting
both full nonlinear registration (warp + affine) and linear-only registration (affine only).
Multiple warp fields can be chained using the --secondary-warp option.

API Usage:
---------
micaflow apply_warp
    --moving <path/to/source_image.nii.gz>
    --reference <path/to/target_space.nii.gz>
    [--affine <path/to/transform.mat>]
    [--warp <path/to/warpfield.nii.gz>]
    [--secondary-warp <path/to/second_warpfield.nii.gz>]
    [--transforms <path/to/t2> <path/to/t1> ...]
    [--output <path/to/registered_image.nii.gz>]
    [--interpolation <method>]

Note: At least one transform (--affine, --warp, --secondary-warp, or --transforms) must be provided.

Transform Application Order:
--------------------------
ANTs applies transforms in REVERSE order from the transform list:
1. Affine (applied first to moving image)
2. Primary warp field (applied second)
3. Secondary warp field (applied last, if provided)

This allows proper composition: subject space → intermediate space → template space

Python Usage:
-----------
>>> import ants
>>> from micaflow.scripts.apply_warp import apply_warp
>>> moving_img = ants.image_read("subject_t1w.nii.gz")
>>> reference_img = ants.image_read("mni152.nii.gz")
>>> 
>>> # With both affine and warp field (nonlinear)
>>> transformed = apply_warp(
...     moving=moving_img,
...     reference=reference_img,
...     affine="transform.mat",
...     warp="warpfield.nii.gz",
...     output="registered_t1w.nii.gz"
... )
>>> 
>>> # With affine only (linear)
>>> transformed = apply_warp(
...     moving="subject_t1w.nii.gz",  # Can also pass file paths
...     reference="mni152.nii.gz",
...     affine="transform.mat",
...     output="linear_registered_t1w.nii.gz"
... )
>>> 
>>> # With chained warp fields
>>> transformed = apply_warp(
...     moving=moving_img,
...     reference=reference_img,
...     affine="affine_to_intermediate.mat",
...     warp="warp_to_intermediate.nii.gz",
...     secondary_warp="warp_intermediate_to_template.nii.gz",
...     output="template_space_t1w.nii.gz"
... )


"""

import ants
import argparse
import sys
from colorama import init, Fore, Style

init()


def print_help_message():
    """Print a help message with examples."""
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
    ║                        APPLY WARP                              ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script applies spatial transformations (affine and/or warp field) to
    register a moving image to a reference space. Multiple transformations can
    be chained together for multi-step registration pipelines.
    
    {CYAN}{BOLD}────────────────────────── REQUIRED ARGUMENTS ──────────────────────────{RESET}
      {YELLOW}--moving{RESET}     : Path to the input image to be warped (.nii.gz)
      {YELLOW}--reference{RESET}  : Path to the target/reference image (.nii.gz)
      
    
    {CYAN}{BOLD}────────────────────────── OPTIONAL ARGUMENTS ──────────────────────────{RESET}
      {YELLOW}--affine{RESET}          : Path to the affine transformation file (.mat)
      {YELLOW}--warp{RESET}            : Path to the primary warp field (.nii.gz)
      {YELLOW}--secondary-warp{RESET}  : Path to a secondary warp field (.nii.gz)
                        Used for chaining transformations through intermediate spaces
                        {MAGENTA}Note: At least one transform must be provided{RESET}
      {YELLOW}--transforms{RESET}      : List of transforms to apply in order (First -> Last)
                        Can be used instead of named transform arguments.
      {YELLOW}--output{RESET}          : Output path for the warped image (default: warped_image.nii.gz)
      {YELLOW}--interpolation{RESET}   : Interpolation method (default: linear)
                        Options: linear, nearestNeighbor, multiLabel, gaussian, 
                                 bSpline, cosineWindowedSinc, welchWindowedSinc,
                                 hammingWindowedSinc, lanczosWindowedSinc, genericLabel
    
    {CYAN}{BOLD}────────────────────────── EXAMPLE USAGE ──────────────────────────{RESET}
    
    {BLUE}# Apply both affine and warp transformation (nonlinear registration){RESET}
    micaflow {GREEN}apply_warp{RESET} {YELLOW}--moving{RESET} subject_t1w.nii.gz {YELLOW}--reference{RESET} mni152.nii.gz \\
      {YELLOW}--affine{RESET} transform.mat {YELLOW}--warp{RESET} warpfield.nii.gz {YELLOW}--output{RESET} registered_t1w.nii.gz
    
    {BLUE}# Apply only affine transformation (linear registration){RESET}
    micaflow {GREEN}apply_warp{RESET} {YELLOW}--moving{RESET} subject_t1w.nii.gz {YELLOW}--reference{RESET} mni152.nii.gz \\
      {YELLOW}--affine{RESET} transform.mat {YELLOW}--output{RESET} linear_registered_t1w.nii.gz
    
    {BLUE}# Chain multiple transformations (subject → intermediate → template){RESET}
    micaflow {GREEN}apply_warp{RESET} {YELLOW}--moving{RESET} subject_dwi.nii.gz {YELLOW}--reference{RESET} mni152.nii.gz \\
      {YELLOW}--affine{RESET} dwi_to_t1w.mat {YELLOW}--warp{RESET} dwi_to_t1w_warp.nii.gz \\
      {YELLOW}--secondary-warp{RESET} t1w_to_mni_warp.nii.gz {YELLOW}--output{RESET} dwi_in_mni.nii.gz
    
    {BLUE}# Use nearest neighbor interpolation for segmentation masks{RESET}
    micaflow {GREEN}apply_warp{RESET} {YELLOW}--moving{RESET} segmentation.nii.gz {YELLOW}--reference{RESET} mni152.nii.gz \\
      {YELLOW}--affine{RESET} transform.mat {YELLOW}--warp{RESET} warp.nii.gz \\
      {YELLOW}--interpolation{RESET} nearestNeighbor {YELLOW}--output{RESET} registered_seg.nii.gz
    
    {CYAN}{BOLD}────────────────────────── TRANSFORM ORDER ──────────────────────────{RESET}
    {MAGENTA}ANTs applies transforms in REVERSE order:{RESET}
      1. Affine is applied FIRST (linear alignment)
      2. Primary warp is applied SECOND (nonlinear deformation)
      3. Secondary warp is applied LAST (additional nonlinear deformation)
    
    This ordering allows proper composition through intermediate spaces:
      Subject space → [affine] → Intermediate space → [warp] → Template space
    
    {CYAN}{BOLD}────────────────────────── NOTES ──────────────────────────{RESET}
    {MAGENTA}•{RESET} At least one transform (affine, warp, or secondary-warp) must be provided
    {MAGENTA}•{RESET} Both moving and reference can be file paths or ANTs image objects in Python API
    {MAGENTA}•{RESET} The function returns the transformed ANTs image object
    {MAGENTA}•{RESET} Use 'genericLabel' interpolation for discrete labels/masks
    {MAGENTA}•{RESET} Use 'linear' or 'bSpline' for continuous intensity images
    {MAGENTA}•{RESET} Transform files must be in ANTs format (.mat for affine, .nii.gz for warps)
    """

    print(help_text)


def apply_warp(moving, reference, affine=None, warp=None, output="warped_image.nii.gz", 
               interpolation="linear", secondary_warp=None, transforms=None):
    """
    Apply spatial transformations to register a moving image to a reference space.
    
    This function applies one or more spatial transformations to align a moving image
    with a reference image. Transformations are applied in the order: affine (first),
    then primary warp, then secondary warp (last). This follows ANTs convention where
    transforms are applied in reverse order of the transform list.
    
    Parameters
    ----------
    moving : str or ants.ANTsImage
        Path to the moving image (.nii.gz) or ANTs image object to be transformed.
    reference : str or ants.ANTsImage
        Path to the reference image (.nii.gz) or ANTs image object defining target space.
    affine : str, optional
        Path to the affine transformation matrix file (.mat format).
        Applied first in the transformation chain.
    warp : str, optional
        Path to the primary warp field (.nii.gz format).
        Applied after affine transformation.
    secondary_warp : str, optional
        Path to a secondary warp field (.nii.gz format).
        Applied after primary warp, useful for chaining transformations through
        intermediate spaces (e.g., subject → intermediate → template).
    transforms : list of str, optional
        List of paths to transforms to apply.
        The transforms should be listed in the order of application (First -> Last).
        This overrides affine, warp, and secondary_warp if provided.
    output : str, optional
        Output path for the warped image. Default: "warped_image.nii.gz"
    interpolation : str, optional
        Interpolation method. Default: "linear"
        Options: linear, nearestNeighbor, multiLabel, gaussian, bSpline,
                cosineWindowedSinc, welchWindowedSinc, hammingWindowedSinc,
                lanczosWindowedSinc, genericLabel
        
    Returns
    -------
    ants.ANTsImage
        The transformed image in the reference space.
        
    Raises
    ------
    ValueError
        If no transformations are provided (all of affine, warp, secondary_warp are None).
        
    Notes
    -----
    - ANTs applies transformations in REVERSE order from the transform list
    - Order: affine (1st) → warp (2nd) → secondary_warp (3rd)
    - Use nearestNeighbor or multiLabel for discrete label images
    - Use linear or bSpline for continuous intensity images
    
    Examples
    --------
    >>> # Simple affine + warp registration
    >>> result = apply_warp(
    ...     moving="subject.nii.gz",
    ...     reference="template.nii.gz",
    ...     affine="transform.mat",
    ...     warp="warpfield.nii.gz",
    ...     output="registered.nii.gz"
    ... )
    
    >>> # Chained transformations through intermediate space
    >>> result = apply_warp(
    ...     moving="dwi.nii.gz",
    ...     reference="mni152.nii.gz",
    ...     affine="dwi_to_t1w.mat",
    ...     warp="dwi_to_t1w_warp.nii.gz",
    ...     secondary_warp="t1w_to_mni_warp.nii.gz",
    ...     output="dwi_in_mni.nii.gz"
    ... )
    """
    # Load images if they are file paths
    if isinstance(moving, str):
        moving_img = ants.image_read(moving)
    else:
        moving_img = moving
        
    if isinstance(reference, str):
        reference_img = ants.image_read(reference)
    else:
        reference_img = reference

    # Build transform list - ANTs applies these in REVERSE order
    # So list order is: [last applied, ..., first applied]
    transform_list = []
    
    if transforms is not None and len(transforms) > 0:
        transform_list = list(transforms)
    else:
        if secondary_warp is not None:
            transform_list.append(secondary_warp)  # Applied last
        if warp is not None:
            transform_list.append(warp)  # Applied second
        if affine is not None:
            transform_list.append(affine)  # Applied first
    
    if not transform_list:
        raise ValueError("At least one transform (affine, warp, secondary_warp, or transforms list) must be provided.")
    
    # Apply transformations
    transformed = ants.apply_transforms(
        fixed=reference_img,
        moving=moving_img,
        transformlist=transform_list,
        interpolator=interpolation,
    )
    
    # Save the result
    ants.image_write(transformed, output)
    print(f"Warped image saved to: {output}")
    
    return transformed


def main():
    # Check if no arguments were provided
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)

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
        "--affine", help="Path to the affine transform (.mat)."
    )
    parser.add_argument(
        "--warp", help="Path to the warp field (.nii.gz)."
    )
    parser.add_argument('--secondary-warp', help='Path to a secondary warp field (.nii.gz) to be applied after the primary warp and affine.')
    parser.add_argument(
        "--transforms", nargs="+", help="List of transforms to apply in order (First -> Last)."
    )
    parser.add_argument(
        "--output", default="warped_image.nii.gz", help="Output warped image filename."
    )
    parser.add_argument(
        "--interpolation",
        default="linear",
        help="Interpolation method (default: linear).",
    )
    args = parser.parse_args()
    
    # Call the apply_warp function with parsed arguments
    apply_warp(
        moving=args.moving,
        reference=args.reference,
        affine=args.affine,
        warp=args.warp,
        output=args.output,
        interpolation=args.interpolation,
        secondary_warp=args.secondary_warp
    )


if __name__ == "__main__":
    main()
