"""
coregister - Label-Augmented Image Registration for Aligning Neuroimaging Data

Part of the micaflow processing pipeline for neuroimaging data.

This module performs comprehensive image registration between two images using LAMAReg
(Label-Augmented Modality-Agnostic Registration), which combines anatomical image information
with segmentation labels to achieve more accurate registration across different imaging modalities. 
The registration aligns a moving image with a fixed reference space, enabling spatial normalization
of neuroimaging data for group analysis, multimodal integration, or atlas-based analyses.

LAMAReg improves upon traditional intensity-based registration by incorporating structural 
segmentation information, making it particularly robust for registering images with different 
contrasts (e.g., T1w to T2w, structural to diffusion) or when anatomical correspondence is 
challenging to establish from intensity alone.

Features:
--------
- Label-augmented registration for improved accuracy across different modalities
- Automatic segmentation generation using SynthSeg when not provided
- Combined rigid, affine, and SyN nonlinear registration in one step
- Linear-only registration option (rigid + affine) for faster processing
- DWI processing support with shell/volume extraction capabilities
- Bidirectional transformation capability (forward and inverse)
- Option to save all transformation components for later application
- Modality-agnostic approach for consistent registration across different contrasts
- Support for multi-threaded processing for both ANTs and SynthSeg components
- Secondary warp file support for complex multi-step registration pipelines

Registration Modes:
------------------
Full Nonlinear (Default):
  - Rigid alignment (6 DOF: 3 translations + 3 rotations)
  - Affine transformation (12 DOF: adds scaling and shearing)
  - SyN nonlinear deformation (captures local anatomical differences)
  - Best for: Normalizing subjects to standard space, high accuracy needed
  - Processing time: ~1-15 minutes (depending on image resolution and threads)

Linear Only (--linear-only):
  - Rigid alignment + Affine transformation only
  - No nonlinear deformation
  - Best for: Intra-subject registration, quick alignment, motion correction
  - Processing time: ~2-4 minutes

Command-Line Usage:
------------------
# Full nonlinear registration with automatic segmentation
micaflow coregister \\
    --fixed-file <path/to/reference.nii.gz> \\
    --moving-file <path/to/source.nii.gz> \\
    --output <path/to/registered.nii.gz> \\
    [--warp-file <path/to/warp.nii.gz>] \\
    [--affine-file <path/to/affine.mat>] \\
    [--rev-warp-file <path/to/reverse_warp.nii.gz>] \\
    [--output-segmentation <path/to/output_seg.nii.gz>] \\
    [--ants-threads <int>] \\
    [--synthseg-threads <int>]

# With pre-computed segmentations (faster)
micaflow coregister \\
    --fixed-file <path/to/reference.nii.gz> \\
    --moving-file <path/to/source.nii.gz> \\
    --fixed-segmentation <path/to/fixed_seg.nii.gz> \\
    --moving-segmentation <path/to/moving_seg.nii.gz> \\
    --output <path/to/registered.nii.gz>

# Linear-only registration
micaflow coregister \\
    --fixed-file <path/to/reference.nii.gz> \\
    --moving-file <path/to/source.nii.gz> \\
    --output <path/to/registered.nii.gz> \\
    --linear-only

Python API Usage:
----------------
>>> from micaflow.scripts.coregister import coregister
>>> 
>>> # Full nonlinear registration
>>> output = coregister(
...     fixed_file="mni152.nii.gz",
...     moving_file="subject_t1w.nii.gz",
...     output="registered_t1w.nii.gz",
...     warp_file="warp.nii.gz",
...     affine_file="affine.mat",
...     rev_warp_file="reverse_warp.nii.gz",
...     ants_threads=4,
...     synthseg_threads=2
... )
>>> 
>>> # Linear-only registration
>>> output = coregister(
...     fixed_file="mni152.nii.gz",
...     moving_file="subject_t1w.nii.gz",
...     output="registered_t1w.nii.gz",
...     linear_only=True,
...     affine_file="affine.mat"
... )

Or use the LAMAReg function directly:
>>> from lamareg.scripts.lamar import lamareg
>>> lamareg(
...     input_image="subject_t1w.nii.gz",
...     reference_image="mni152.nii.gz",
...     output_image="registered_t1w.nii.gz",
...     input_parc="subject_seg.nii.gz",  # Optional
...     reference_parc="mni152_seg.nii.gz",  # Optional
...     output_parc="registered_seg.nii.gz",
...     warp_file="warp.nii.gz",
...     affine_file="affine.mat",
...     inverse_warp_file="reverse_warp.nii.gz",
...     skip_moving_parc=False,  # Generate if not provided
...     skip_fixed_parc=False,   # Generate if not provided
...     ants_threads=4,
...     synthseg_threads=2
... )

Exit Codes:
----------
0 : Success - registration completed
1 : Error - invalid inputs, file not found, or processing failure

Technical Notes:
---------------
- LAMAReg uses ANTs SyN (Symmetric Normalization) for nonlinear registration
- Segmentations are generated using SynthSeg (deep learning-based segmentation)
- Forward transforms: moving → fixed space
- Reverse transforms: fixed → moving space
- Secondary warp files allow enable greater accuracy in due to less interpolation
- When secondary warps are not provided, warp fields are composed into single transform
- Multi-threading significantly reduces processing time on multi-core systems
- SynthSeg can process nearly any MRI contrast without retraining

See Also:
--------
- synthseg : For generating segmentation/parcellation volumes
- apply_warp : For applying saved transformations to other images
- calculate_dice : For validating registration quality

References:
----------

2. Billot B, Greve DN, Puonti O, et al. SynthSeg: Segmentation of brain MRI scans of 
   any contrast and resolution without retraining. Medical Image Analysis. 2023;86:102789.
   doi:10.1016/j.media.2023.102789

3. Tustison NJ, Cook PA, Klein A, et al. Large-scale evaluation of ANTs and FreeSurfer 
   cortical thickness measurements. Neuroimage. 2014;99:166-179. 
   doi:10.1016/j.neuroimage.2014.05.044
"""

import argparse
import sys
import os
from colorama import init, Fore, Style
import ants
import shutil

init()

# ANSI color codes
CYAN = Fore.CYAN
GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
BLUE = Fore.BLUE
MAGENTA = Fore.MAGENTA
RED = Fore.RED
BOLD = Style.BRIGHT
RESET = Style.RESET_ALL


def print_help_message():
    """Print comprehensive help message with examples and technical details."""
    
    help_text = f"""
    {CYAN}{BOLD}╔════════════════════════════════════════════════════════════════╗
    ║                LABEL-AUGMENTED COREGISTRATION                  ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script performs label-augmented modality-agnostic registration (LAMAReg) 
    between two images. The registration aligns the moving image to match the fixed 
    reference image space, utilizing segmentation labels to improve accuracy across 
    different modalities.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow coregister {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--fixed-file{RESET}   : Path to the fixed/reference image (.nii.gz)
                     {MAGENTA}The target space for registration{RESET}
      {YELLOW}--moving-file{RESET}  : Path to the moving image to be registered (.nii.gz)
                     {MAGENTA}Will be transformed to match fixed image{RESET}
      {YELLOW}--output{RESET}       : Output path for the registered image (.nii.gz)
    
    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--fixed-segmentation{RESET}   : Path to fixed image segmentation (.nii.gz)
                            {MAGENTA}Auto-generated if not provided{RESET}
      {YELLOW}--moving-segmentation{RESET}  : Path to moving image segmentation (.nii.gz)
                            {MAGENTA}Auto-generated if not provided{RESET}
      {YELLOW}--output-segmentation{RESET}  : Path to save registered segmentation (.nii.gz)
      {YELLOW}--warp-file{RESET}            : Path to save forward warp field (.nii.gz)
                            {MAGENTA}Moving → Fixed transformation{RESET}
      {YELLOW}--affine-file{RESET}          : Path to save forward affine (.mat)
      {YELLOW}--rev-warp-file{RESET}        : Path to save reverse warp field (.nii.gz)
                            {MAGENTA}Fixed → Moving transformation{RESET}
      {YELLOW}--secondary-warp-file{RESET}  : Path for secondary warp (multi-step pipelines)
                            {MAGENTA}More accurate but complex to apply{RESET}
      {YELLOW}--secondary-rev-warp-file{RESET}: Path for secondary reverse warp
      {YELLOW}--linear-only{RESET}          : Only perform rigid + affine (no SyN)
                            {MAGENTA}Faster but less accurate{RESET}
      {YELLOW}--ants-threads{RESET}         : Number of threads for ANTs (default: 1)
      {YELLOW}--synthseg-threads{RESET}     : Number of threads for SynthSeg (default: 1)
    
    {CYAN}{BOLD}────────────────── EXAMPLE USAGE ────────────────────────{RESET}
    
    {BLUE}# Example 1: Basic nonlinear registration (automatic segmentation){RESET}
    micaflow coregister \\
      {YELLOW}--fixed-file{RESET} mni152.nii.gz \\
      {YELLOW}--moving-file{RESET} subject_t1w.nii.gz \\
      {YELLOW}--output{RESET} registered_t1w.nii.gz \\
      {YELLOW}--warp-file{RESET} warp.nii.gz \\
      {YELLOW}--affine-file{RESET} affine.mat
    
    {BLUE}# Example 2: With pre-computed segmentations (faster){RESET}
    micaflow coregister \\
      {YELLOW}--fixed-file{RESET} mni152.nii.gz \\
      {YELLOW}--moving-file{RESET} subject_t1w.nii.gz \\
      {YELLOW}--fixed-segmentation{RESET} mni152_seg.nii.gz \\
      {YELLOW}--moving-segmentation{RESET} subject_seg.nii.gz \\
      {YELLOW}--output{RESET} registered_t1w.nii.gz \\
      {YELLOW}--output-segmentation{RESET} registered_seg.nii.gz
    
    {BLUE}# Example 3: Linear-only registration (fast){RESET}
    micaflow coregister \\
      {YELLOW}--fixed-file{RESET} subject_t1w.nii.gz \\
      {YELLOW}--moving-file{RESET} subject_t2w.nii.gz \\
      {YELLOW}--output{RESET} t2w_in_t1w_space.nii.gz \\
      {YELLOW}--linear-only{RESET} \\
      {YELLOW}--affine-file{RESET} t2w_to_t1w.mat
    
    {BLUE}# Example 4: Multi-threaded registration{RESET}
    micaflow coregister \\
      {YELLOW}--fixed-file{RESET} mni152.nii.gz \\
      {YELLOW}--moving-file{RESET} subject_t1w.nii.gz \\
      {YELLOW}--output{RESET} registered_t1w.nii.gz \\
      {YELLOW}--ants-threads{RESET} 4 \\
      {YELLOW}--synthseg-threads{RESET} 2
    
    {BLUE}# Example 5: Multi-step registration with secondary warps{RESET}
    micaflow coregister \\
      {YELLOW}--fixed-file{RESET} mni152.nii.gz \\
      {YELLOW}--moving-file{RESET} subject_dwi.nii.gz \\
      {YELLOW}--output{RESET} dwi_in_mni.nii.gz \\
      {YELLOW}--warp-file{RESET} primary_warp.nii.gz \\
      {YELLOW}--secondary-warp-file{RESET} secondary_warp.nii.gz
    
    {CYAN}{BOLD}────────────────── REGISTRATION MODES ───────────────────{RESET}
    
    {GREEN}Full Nonlinear (Default):{RESET}
    {MAGENTA}•{RESET} Rigid → Affine → SyN deformation
    {MAGENTA}•{RESET} Best accuracy for inter-subject registration
    {MAGENTA}•{RESET} Processing time: ~5-15 minutes
    {MAGENTA}•{RESET} Use for: Normalizing to standard space (e.g., MNI152)
    
    {GREEN}Linear Only (--linear-only):{RESET}
    {MAGENTA}•{RESET} Rigid → Affine only (no nonlinear deformation)
    {MAGENTA}•{RESET} Faster but less accurate
    {MAGENTA}•{RESET} Processing time: ~1-3 minutes
    {MAGENTA}•{RESET} Use for: Intra-subject registration, motion correction
    
    {CYAN}{BOLD}────────────────────────── NOTES ───────────────────────{RESET}
    {MAGENTA}•{RESET} LAMAReg combines anatomical and label information for robust registration
    {MAGENTA}•{RESET} Segmentations are automatically generated using SynthSeg if not provided
    {MAGENTA}•{RESET} SynthSeg adds ~1-3 minutes per image to processing time
    {MAGENTA}•{RESET} Forward transforms: moving space → fixed space
    {MAGENTA}•{RESET} Reverse transforms: fixed space → moving space
    {MAGENTA}•{RESET} Save transforms to apply to other images later (use apply_warp)
    {MAGENTA}•{RESET} Multi-threading significantly reduces processing time
    {MAGENTA}•{RESET} Secondary warps enable greate accuracy due to skipping an interpolation step
    {MAGENTA}•{RESET} LAMAReg works across different MRI contrasts (T1w, T2w, FLAIR, etc.)
    {MAGENTA}•{RESET} Registration quality can be validated using calculate_dice
    
    {CYAN}{BOLD}────────────────────── EXIT CODES ───────────────────────{RESET}
    {GREEN}0{RESET} : Success - registration completed
    {RED}1{RESET} : Error - invalid inputs, file not found, or processing failure
    
    {CYAN}{BOLD}───────────────── COMMON ISSUES ─────────────────────────{RESET}
    {YELLOW}Issue:{RESET} Registration takes very long
    {GREEN}Solution:{RESET} Increase --ants-threads, provide pre-computed segmentations
    
    {YELLOW}Issue:{RESET} Poor registration quality
    {GREEN}Solution:{RESET} Ensure images have similar FOV, try providing segmentations
    
    {YELLOW}Issue:{RESET} "Out of memory" error
    {GREEN}Solution:{RESET} Reduce number of threads or image resolution
    """
    print(help_text)


def coregister(fixed_file, moving_file, output, fixed_segmentation=None,
               moving_segmentation=None, warp_file=None, affine_file=None,
               rev_warp_file=None, ants_threads=1,
               synthseg_threads=1, output_segmentation=None, linear_only=False, 
               secondary_warp_file=None, secondary_rev_warp_file=None):
    """
    Perform label-augmented image registration between two images.
    
    This function uses LAMAReg (Label-Augmented Modality-Agnostic Registration) to
    align a moving image with a fixed reference image. It can perform either full
    nonlinear registration (rigid + affine + SyN) or linear-only registration
    (rigid + affine). Segmentations can be provided or will be generated automatically
    using SynthSeg.
    
    Parameters
    ----------
    fixed_file : str
        Path to the fixed/reference image (.nii.gz).
        This defines the target space.
    moving_file : str
        Path to the moving image to be registered (.nii.gz).
        Will be transformed to match the fixed image.
    output : str
        Output path for the registered image (.nii.gz).
    fixed_segmentation : str, optional
        Path to fixed image segmentation (.nii.gz).
        If not provided, will be generated automatically using SynthSeg.
    moving_segmentation : str, optional
        Path to moving image segmentation (.nii.gz).
        If not provided, will be generated automatically using SynthSeg.
    warp_file : str, optional
        Path to save the forward warp field (.nii.gz).
        Transforms from moving to fixed space.
    affine_file : str, optional
        Path to save the forward affine transform (.mat).
        Linear component of moving to fixed transformation.
    rev_warp_file : str, optional
        Path to save the reverse warp field (.nii.gz).
        Transforms from fixed to moving space.
    rev_affine_file : str, optional
        Path to save the reverse affine transform (.mat).
        Linear component of fixed to moving transformation.
    ants_threads : int, optional
        Number of threads for ANTs registration operations. Default: 1.
        Higher values speed up processing on multi-core systems.
    synthseg_threads : int, optional
        Number of threads for SynthSeg segmentation operations. Default: 1.
    output_segmentation : str, optional
        Path to save the registered segmentation image (.nii.gz).
        Only used when segmentations are provided or generated.
    linear_only : bool, optional
        If True, only perform linear registration (rigid + affine) without 
        nonlinear SyN step. Faster but less accurate. Default: False.
    secondary_warp_file : str, optional
        Path to save secondary warp field for multi-step registration pipelines.
        More accurate but can be complex to apply.
    secondary_rev_warp_file : str, optional
        Path to save secondary reverse warp field.
        
    Returns
    -------
    str
        Path to the registered output image.
        
    Raises
    ------
    FileNotFoundError
        If input files cannot be found.
    RuntimeError
        If registration fails.
        
    Notes
    -----
    - Processing time varies: linear-only ~1-3 min, full nonlinear ~5-15 min
    - SynthSeg adds ~2-3 minutes per image when segmentations need generation
    - Multi-threading can reduce processing time significantly
    - Save transformations to apply them to other images later
    
    Examples
    --------
    >>> # Full nonlinear registration
    >>> output = coregister(
    ...     fixed_file="mni152.nii.gz",
    ...     moving_file="subject_t1w.nii.gz",
    ...     output="registered_t1w.nii.gz",
    ...     warp_file="warp.nii.gz",
    ...     affine_file="affine.mat",
    ...     ants_threads=4
    ... )
    >>> 
    >>> # Linear-only registration
    >>> output = coregister(
    ...     fixed_file="subject_t1w.nii.gz",
    ...     moving_file="subject_flair.nii.gz",
    ...     output="flair_in_t1w.nii.gz",
    ...     linear_only=True,
    ...     affine_file="flair_to_t1w.mat"
    ... )
    >>> 
    >>> # With pre-computed segmentations
    >>> output = coregister(
    ...     fixed_file="mni152.nii.gz",
    ...     moving_file="subject_t1w.nii.gz",
    ...     fixed_segmentation="mni152_seg.nii.gz",
    ...     moving_segmentation="subject_seg.nii.gz",
    ...     output="registered_t1w.nii.gz",
    ...     output_segmentation="registered_seg.nii.gz"
    ... )
    """
    # Validate input files exist
    for filepath, name in [(fixed_file, "Fixed"), (moving_file, "Moving")]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{name} file not found: {filepath}")
    
    if fixed_segmentation and not os.path.exists(fixed_segmentation):
        raise FileNotFoundError(f"Fixed segmentation not found: {fixed_segmentation}")
    
    if moving_segmentation and not os.path.exists(moving_segmentation):
        raise FileNotFoundError(f"Moving segmentation not found: {moving_segmentation}")
    
    if linear_only:
        print(f"{CYAN}Linear-only registration selected.{RESET}")
        print(f"{CYAN}Performing rigid + affine registration (no SyN)...{RESET}")
        
        moving = ants.image_read(moving_file)
        fixed = ants.image_read(fixed_file)
        
        print(f"  Fixed image: {fixed_file} (shape: {fixed.shape})")
        print(f"  Moving image: {moving_file} (shape: {moving.shape})")
        
        registration = ants.registration(
            fixed=fixed, 
            moving=moving, 
            type_of_transform='Affine'
        )
        
        registered_image = registration['warpedmovout']
        ants.image_write(registered_image, output)
        print(f"{GREEN}Registered image saved: {output}{RESET}")
        
        if affine_file:
            shutil.copy(registration['fwdtransforms'][0], affine_file)
            print(f"{GREEN}Affine transform saved: {affine_file}{RESET}")
            
        # Transform segmentation if provided
        if moving_segmentation and fixed_segmentation and output_segmentation:
            print(f"{CYAN}Transforming segmentation...{RESET}")
            moving_seg = ants.image_read(moving_segmentation)
            fixed_seg = ants.image_read(fixed_segmentation)
            transformed = ants.apply_transforms(
                fixed=fixed_seg,
                moving=moving_seg,
                transformlist=registration['fwdtransforms'],
                interpolator='nearestNeighbor'
            )
            ants.image_write(transformed, output_segmentation)
            print(f"{GREEN}Registered segmentation saved: {output_segmentation}{RESET}")
        
        # Cleanup temporary files
        for transform_file in registration['fwdtransforms'] + registration['invtransforms']:
            if os.path.exists(transform_file) and transform_file not in [affine_file]:
                try:
                    os.remove(transform_file)
                except:
                    pass
        
    elif fixed_segmentation and moving_segmentation:
        print(f"{CYAN}Using provided segmentation images.{RESET}")
        print(f"{CYAN}Performing full nonlinear registration (rigid + affine + SyN)...{RESET}")
        
        from lamareg.scripts.lamar import lamareg
        lamareg(
            input_image=moving_file,
            reference_image=fixed_file,
            output_image=output,
            input_parc=moving_segmentation,
            reference_parc=fixed_segmentation,
            output_parc=output_segmentation,
            affine_file=affine_file,
            warp_file=warp_file,
            inverse_warp_file=rev_warp_file,
            skip_moving_parc=True,
            skip_fixed_parc=True,
            skip_qc=True,
            ants_threads=ants_threads,
            synthseg_threads=synthseg_threads,
            secondary_warp_file=secondary_warp_file,
            inverse_secondary_warp_file=secondary_rev_warp_file
        )
        print(f"{GREEN}Registration complete!{RESET}")
        
    else:
        print(f"{YELLOW}No segmentations provided. Will generate using SynthSeg...{RESET}")
        print(f"{CYAN}Performing full nonlinear registration (rigid + affine + SyN)...{RESET}")
        
        from lamareg.scripts.lamar import lamareg
        
        # Generate paths for auto-generated segmentations
        auto_moving_segmentation = moving_file.replace('.nii.gz', '_parc.nii.gz')
        auto_fixed_segmentation = fixed_file.replace('.nii.gz', '_parc.nii.gz')
        
        print(f"  Auto-segmentation paths:")
        print(f"    Moving: {auto_moving_segmentation}")
        print(f"    Fixed: {auto_fixed_segmentation}")
        
        lamareg(
            input_image=moving_file,
            reference_image=fixed_file,
            output_image=output,
            input_parc=auto_moving_segmentation,
            reference_parc=auto_fixed_segmentation,
            output_parc=output_segmentation,
            affine_file=affine_file,
            warp_file=warp_file,
            inverse_warp_file=rev_warp_file,
            skip_moving_parc=False,
            skip_fixed_parc=False,
            skip_qc=True,
            ants_threads=ants_threads,
            synthseg_threads=synthseg_threads,
            secondary_warp_file=secondary_warp_file,
            inverse_secondary_warp_file=secondary_rev_warp_file
        )
        print(f"{GREEN}Registration complete!{RESET}")
        
    return output


if __name__ == "__main__":
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Run label-augmented modality-agnostic registration (LAMAReg) between two images.",
        add_help=False  # Use custom help
    )
    parser.add_argument("--fixed-file", required=True, 
                        help="Path to the fixed/reference image.")
    parser.add_argument("--moving-file", required=True, 
                        help="Path to the moving image to be registered.")
    parser.add_argument("--fixed-segmentation", 
                        help="Path to the fixed segmentation image. If not provided, it will be generated automatically.")
    parser.add_argument("--moving-segmentation", 
                        help="Path to the moving segmentation image. If not provided, it will be generated automatically.")
    parser.add_argument("--output", required=True,
                        help="Output path for the registered image.")
    parser.add_argument("--warp-file", default=None, 
                        help="Optional path to save the forward warp field (moving to fixed).")
    parser.add_argument("--affine-file", default=None,
                        help="Optional path to save the forward affine transform (moving to fixed).")
    parser.add_argument("--rev-warp-file", default=None,
                        help="Optional path to save the reverse warp field (fixed to moving).")
    parser.add_argument("--ants-threads", type=int, default=1, 
                        help="Number of threads for ANTs registration operations (default: 1).")
    parser.add_argument("--synthseg-threads", type=int, default=1, 
                        help="Number of threads for SynthSeg segmentation operations (default: 1).")
    parser.add_argument("--output-segmentation", default=None,
                        help="Optional path to save the output segmentation image alongside the registered image.")
    parser.add_argument("--linear-only", action='store_true',
                        help="If set, only perform linear registration (rigid + affine) without nonlinear SyN step.")
    parser.add_argument("--secondary-warp-file", 
                        help="If provided, will save a secondary warp file. More accurate but can be difficult to apply. If not provided, warpfields will be composed.")
    parser.add_argument("--secondary-rev-warp-file", 
                        help="If provided, will save a secondary reverse warp file. More accurate but can be difficult to apply. If not provided, warpfields will be composed.")
    
    args = parser.parse_args()
    
    try:
        # Call the coregister function with parsed arguments
        output_path = coregister(
            fixed_file=args.fixed_file,
            moving_file=args.moving_file,
            output=args.output,
            fixed_segmentation=args.fixed_segmentation,
            moving_segmentation=args.moving_segmentation,
            warp_file=args.warp_file,
            affine_file=args.affine_file,
            rev_warp_file=args.rev_warp_file,
            ants_threads=args.ants_threads,
            synthseg_threads=args.synthseg_threads,
            output_segmentation=args.output_segmentation,
            linear_only=args.linear_only,
            secondary_warp_file=args.secondary_warp_file,
            secondary_rev_warp_file=args.secondary_rev_warp_file
        )
        
        print(f"\n{GREEN}{BOLD}Registration successfully completed!{RESET}")
        print(f"  Output: {output_path}")
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\n{RED}{BOLD}File not found:{RESET}")
        print(f"  {str(e)}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n{RED}{BOLD}Error during registration:{RESET}")
        print(f"  {str(e)}")
        print(f"\n{YELLOW}Run 'micaflow coregister --help' for usage information.{RESET}")
        sys.exit(1)