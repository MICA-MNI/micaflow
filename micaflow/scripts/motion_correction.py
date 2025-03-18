import argparse
import ants
import numpy as np
import sys
from tqdm import tqdm
from colorama import init, Fore, Style

init()

def print_help_message():
    """Print a help message with formatted text."""
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
    ║                     MOTION CORRECTION                          ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script corrects for subject motion in diffusion-weighted images (DWI)
    by registering each volume to the first volume (typically a B0 image).
    It uses ANTs SyNRA registration which combines rigid, affine, and
    deformable transformations.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow motion_correction {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--denoised{RESET}   : Path to the input denoised DWI image (.nii.gz)
      {YELLOW}--bval{RESET}       : Path to the b-values file (.bval)
      {YELLOW}--bvec{RESET}       : Path to the b-vectors file (.bvec)
      {YELLOW}--output{RESET}     : Output path for the motion-corrected image (.nii.gz)
    
    {CYAN}{BOLD}─────────────────── EXAMPLE USAGE ───────────────────{RESET}
      micaflow motion_correction \\
        {YELLOW}--denoised{RESET} denoised_dwi.nii.gz \\
        {YELLOW}--bval{RESET} dwi.bval \\
        {YELLOW}--bvec{RESET} dwi.bvec \\
        {YELLOW}--output{RESET} motion_corrected_dwi.nii.gz
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    - The first volume is assumed to be a B0 image and used as the reference
    - Each subsequent volume is registered to this reference
    - The process can take significant time depending on volume count
    - Progress is displayed using a progress bar
    """
    print(help_text)

def run_motion_correction(dwi_path, bval_path, bvec_path, output):
    """Perform motion correction on diffusion-weighted images (DWI).
    
    This function corrects for subject motion in DWI data by registering each volume 
    to the first volume (assumed to be a B0 image). It uses ANTs SyNRA registration
    which combines rigid, affine, and deformable transformations to achieve robust 
    alignment between volumes.
    
    Args:
        dwi_path (str): Path to the input DWI NIfTI file.
        bval_path (str): Path to the b-values file (.bval). Currently unused but 
            included for API consistency.
        bvec_path (str): Path to the b-vectors file (.bvec). Currently unused but 
            included for API consistency.
        output (str): Path where the motion-corrected DWI will be saved.
        
    Returns:
        str: Path to the saved motion-corrected DWI image.
        
    Notes:
        The function assumes the first volume (index 0) is a B0 image that serves
        as the reference for registration. All other volumes are aligned to this
        reference using ANTs' SyNRA transformation. Progress is displayed using 
        a tqdm progress bar.
    """
    # Read the main DWI file using ANTs
    dwi_ants = ants.image_read(dwi_path)
    dwi_data = dwi_ants.numpy()

    # B0 is assumed to be the first volume (index 0)
    b0_data = dwi_data[..., 0]
    b0_ants = ants.from_numpy(
        b0_data, origin=dwi_ants.origin[:3], spacing=dwi_ants.spacing[:3]
    )

    registered_data = np.zeros_like(dwi_data)
    # Keep the original B0 in the first volume
    registered_data[..., 0] = b0_data

    # Register each shell to B0 using a quick approach
    for idx in tqdm(range(1, dwi_data.shape[-1]), desc="Registering volumes"):
        moving_data = dwi_data[..., idx]
        moving_ants = ants.from_numpy(
            moving_data, origin=dwi_ants.origin[:3], spacing=dwi_ants.spacing[:3]
        )

        # Non-linear registration (SyNOnly) using the rigid transform as initial
        quicksyn_reg = ants.registration(
            fixed=b0_ants,
            moving=moving_ants,
            type_of_transform="SyNRA",
        )

        # Place the registered volume in the output array
        warped_data = quicksyn_reg["warpedmovout"].numpy()
        registered_data[..., idx] = warped_data

    # Save the registered data
    registered_ants = ants.from_numpy(
        registered_data, origin=dwi_ants.origin, spacing=dwi_ants.spacing, direction=dwi_ants.direction
    )

    ants.image_write(registered_ants, output)

    print("Motion correction completed for all shells with QuickSyN registration.")
    return output


if __name__ == "__main__":
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="Perform motion correction on a DWI image using ANTs QuickSyN."
    )
    parser.add_argument(
        "--denoised",
        type=str,
        required=True,
        help="Path to the denoised DWI (NIfTI file).",
    )
    parser.add_argument(
        "--bval",
        type=str,
        required=True,
        help="Path to the bvals file. (Currently unused, but retained for consistency.)",
    )
    parser.add_argument(
        "--bvec",
        type=str,
        required=True,
        help="Path to the bvecs file. (Currently unused, but retained for consistency.)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the motion-corrected DWI.",
    )

    args = parser.parse_args()
    corrected_image = run_motion_correction(
        args.denoised, args.bval, args.bvec, args.output
    )
    print("Motion corrected image saved as:", corrected_image)