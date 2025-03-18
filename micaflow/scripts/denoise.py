import argparse
import nibabel as nib
import sys
from dipy.denoise.patch2self import patch2self
from dipy.io.gradients import read_bvals_bvecs
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
    ║                     DWI IMAGE DENOISING                        ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script denoises diffusion-weighted images (DWI) using the Patch2Self 
    algorithm, which leverages redundant information across diffusion gradients
    to remove noise without requiring additional reference scans.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow denoise {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--input{RESET}     : Path to the input DWI image (.nii.gz)
      {YELLOW}--bval{RESET}      : Path to the b-values file (.bval)
      {YELLOW}--bvec{RESET}      : Path to the b-vectors file (.bvec)
      {YELLOW}--output{RESET}    : Output path for the denoised image (.nii.gz)
    
    {CYAN}{BOLD}─────────────────── EXAMPLE USAGE ───────────────────{RESET}
      micaflow denoise \\
        {YELLOW}--input{RESET} raw_dwi.nii.gz \\
        {YELLOW}--bval{RESET} dwi.bval \\
        {YELLOW}--bvec{RESET} dwi.bvec \\
        {YELLOW}--output{RESET} denoised_dwi.nii.gz
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    - Patch2Self is a self-supervised learning method for denoising
    - Processing preserves anatomical structure while removing noise
    - The implementation uses OLS regression with b0 threshold of 50 s/mm²
    - B0 volumes are not denoised separately in this implementation
    """
    print(help_text)

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
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="Denoise a DWI image using patch2self."
    )
    parser.add_argument(
        "--input",
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
    parser.add_argument("--output", type=str, required=True, help="Output path for denoised image")

    args = parser.parse_args()
    output_path = run_denoise(args.input, args.bval, args.bvec, args.output)
    print("Denoised image saved as:", output_path)