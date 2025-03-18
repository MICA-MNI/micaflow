"""
compute_fa_md - Diffusion Tensor Imaging Metrics Calculator

Part of the micaflow processing pipeline for neuroimaging data.

This module computes diffusion tensor imaging (DTI) scalar metrics, specifically 
Fractional Anisotropy (FA) and Mean Diffusivity (MD), from preprocessed diffusion-weighted 
images (DWI). FA quantifies the directional preference of water diffusion, serving as a 
marker of white matter integrity, while MD represents the overall magnitude of diffusion. 
These metrics are essential for investigating white matter microstructure and are widely 
used in clinical and research neuroimaging.

Features:
--------
- Computes DTI model using robust tensor fitting from DIPY
- Generates both FA and MD maps in a single operation
- Supports masking to restrict calculations to brain tissue
- Compatible with standard neuroimaging file formats (NIfTI)
- Preserves image header and spatial information in output files

API Usage:
---------
micaflow compute_fa_md 
    --input <path/to/dwi.nii.gz>
    --mask <path/to/brain_mask.nii.gz>
    --bval <path/to/dwi.bval>
    --bvec <path/to/dwi.bvec>
    --output-fa <path/to/fa_map.nii.gz>
    --output-md <path/to/md_map.nii.gz>

Python Usage:
-----------
>>> from micaflow.scripts.compute_fa_md import compute_fa_md
>>> fa_path, md_path = compute_fa_md(
...     bias_corr_path="corrected_dwi.nii.gz",
...     mask_path="brain_mask.nii.gz",
...     moving_bval="dwi.bval",
...     moving_bvec="dwi.bvec",
...     fa_path="fa.nii.gz",
...     md_path="md.nii.gz"
... )

"""
import argparse
import sys
from dipy.reconst.dti import TensorModel
from dipy.core.gradients import gradient_table
import nibabel as nib
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
    ║                DIFFUSION TENSOR METRICS (FA/MD)                ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script computes Fractional Anisotropy (FA) and Mean Diffusivity (MD)
    maps from diffusion-weighted images using the diffusion tensor model.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow compute_fa_md {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--input{RESET}      : Path to the input DWI image (.nii.gz)
      {YELLOW}--mask{RESET}       : Path to the brain mask image (.nii.gz)
      {YELLOW}--bval{RESET}       : Path to the b-values file (.bval)
      {YELLOW}--bvec{RESET}       : Path to the b-vectors file (.bvec)
      {YELLOW}--output-fa{RESET}  : Output path for the FA map (.nii.gz)
      {YELLOW}--output-md{RESET}  : Output path for the MD map (.nii.gz)
    
    {CYAN}{BOLD}──────────────────── EXAMPLE USAGE ──────────────────────{RESET}
      micaflow compute_fa_md \\
        {YELLOW}--input{RESET} corrected_dwi.nii.gz \\
        {YELLOW}--mask{RESET} brain_mask.nii.gz \\
        {YELLOW}--bval{RESET} dwi.bval \\
        {YELLOW}--bvec{RESET} dwi.bvec \\
        {YELLOW}--output-fa{RESET} fa.nii.gz \\
        {YELLOW}--output-md{RESET} md.nii.gz
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    - FA (Fractional Anisotropy) values range from 0 (isotropic) to 1 (anisotropic)
    - MD (Mean Diffusivity) measures the overall magnitude of diffusion
    - Processing requires a brain mask to exclude non-brain regions
    
    """
    print(help_text)

# ----- Function: FA/MD Estimation -----
def compute_fa_md(bias_corr_path, mask_path, moving_bval, moving_bvec, fa_path, md_path):
    """Compute Fractional Anisotropy (FA) and Mean Diffusivity (MD) maps from diffusion-weighted images.
    
    This function takes a bias-corrected diffusion-weighted image (DWI) and a brain mask,
    creates a diffusion tensor model, and calculates FA and MD maps. The resulting
    maps are saved as NIfTI files at the specified output paths.
    
    Args:
        bias_corr_path (str): Path to the bias-corrected DWI image (NIfTI file).
        mask_path (str): Path to the brain mask image (NIfTI file).
        moving_bval (str): Path to the b-values file (.bval).
        moving_bvec (str): Path to the b-vectors file (.bvec).
        fa_path (str): Output path for the fractional anisotropy (FA) map.
        md_path (str): Output path for the mean diffusivity (MD) map.
        
    Returns:
        tuple: A tuple containing two strings (fa_path, md_path) - the paths to the 
              saved FA and MD NIfTI files.
    """
    bias_corr = nib.load(bias_corr_path)
    mask = nib.load(mask_path)
    masked_data = bias_corr.get_fdata() * mask.get_fdata()[..., None]
    gtab = gradient_table(moving_bval, moving_bvec)
    tensor_model = TensorModel(gtab)
    tensor_fit = tensor_model.fit(masked_data)
    fa = tensor_fit.fa
    md = tensor_fit.md
    nib.save(nib.Nifti1Image(fa, bias_corr.affine), fa_path)
    nib.save(nib.Nifti1Image(md, bias_corr.affine), md_path)
    return fa_path, md_path

if __name__ == "__main__":
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="Compute FA and MD maps using bias-corrected DWI and a brain mask."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the bias-corrected DWI image (NIfTI file).")
    parser.add_argument("--mask", type=str, required=True,
                        help="Path to the brain mask image (NIfTI file).")
    parser.add_argument("--bval", type=str, required=True,
                        help="Path to the bvals file.")
    parser.add_argument("--bvec", type=str, required=True,
                        help="Path to the bvecs file.")
    parser.add_argument("--output-fa", type=str, required=True,
                        help="Output path for the FA map.")
    parser.add_argument("--output-md", type=str, required=True,
                        help="Output path for the MD map.")
    args = parser.parse_args()
    
    fa_path, md_path = compute_fa_md(args.input, args.mask, args.bval, args.bvec, args.output_fa, args.output_md)
    print("FA map saved as:", fa_path)
    print("MD map saved as:", md_path)