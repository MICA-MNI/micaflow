"""
calculate_dice - Segmentation Overlap Measurement Tool

Part of the micaflow processing pipeline for neuroimaging data.

This module calculates the DICE between 
two segmentation volumes. The DICE score is a statistic used for comparing the 
similarity and diversity of sample sets, with values ranging from 0 (no overlap) to 
1 (perfect overlap). It is particularly useful for evaluating the quality of 
segmentations against a ground truth or comparing results from different methods.

Features:
--------
- Support for multi-label segmentations with per-ROI analysis
- Optional masking to restrict comparison to specific regions
- Configurable threshold for probabilistic segmentations
- CSV output format for easy integration with analysis workflows

API Usage:
---------
micaflow calculate_dice 
    --input <path/to/segmentation.nii.gz>
    --reference <path/to/ground_truth.nii.gz>
    --output <path/to/results.csv>
    [--mask <path/to/mask.nii.gz>]
    [--threshold <value>]

Python Usage:
-----------
>>> from micaflow.scripts.calculate_dice import main
>>> main(
...     image="segmentation.nii.gz",
...     reference="ground_truth.nii.gz",
...     output_file="dice_results.csv",
...     threshold=0.5,  # optional
...     mask_path="brain_mask.nii.gz"  # optional
... )

"""
import csv
import nibabel as nib
import argparse
import sys
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
    ║                     DICE SCORE CALCULATOR                      ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script calculates the DICE score (also known as the Sørensen-Dice coefficient)
    between two segmentation volumes, either globally or for each ROI.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow calculate_dice {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--input{RESET}, {YELLOW}-i{RESET}     : Path to the first input segmentation volume (.nii.gz)
      {YELLOW}--reference{RESET}, {YELLOW}-r{RESET} : Path to the reference segmentation volume (.nii.gz)
      {YELLOW}--output{RESET}, {YELLOW}-o{RESET}    : Output path for the CSV file with DICE scores

    {CYAN}{BOLD}──────────────────── EXAMPLE USAGE ──────────────────────{RESET}
      micaflow calculate_dice \\
        {YELLOW}--input{RESET} segmentation1.nii.gz \\
        {YELLOW}--reference{RESET} ground_truth.nii.gz \\
        {YELLOW}--output{RESET} dice_metrics.csv
      
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    - For multi-label segmentations, the DICE score is computed for each label
    - Values range from 0 (no overlap) to 1 (perfect overlap)
    """
    print(help_text)
from lamar.scripts.dice_compare import compare_parcellations_dice


if __name__ == "__main__":
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)
    
    parser = argparse.ArgumentParser(description="Calculate overlap metrics between two volumes")
    parser.add_argument("--input", "-i", required=True, help="First input volume")
    parser.add_argument("--reference", "-r", required=True, help="Reference volume to compare against")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file path")
    
    args = parser.parse_args()
    
    compare_parcellations_dice(args.input, args.reference, args.output)