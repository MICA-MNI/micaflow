"""
calculate_dice - Segmentation Overlap Measurement Tool

Part of the micaflow processing pipeline for neuroimaging data.

This module calculates the DICE coefficient (Sørensen-Dice coefficient) between
two segmentation volumes. The DICE score is a statistic used for comparing the
similarity and diversity of sample sets, with values ranging from 0 (no overlap) to
1 (perfect overlap). It is particularly useful for evaluating the quality of
segmentations against a ground truth or comparing results from different methods.

The DICE coefficient is defined as:

    DICE = 2 * |A ∩ B| / (|A| + |B|)

where A and B are the sets of voxels belonging to a particular label in each 
segmentation, |A ∩ B| is the number of voxels in common, and |A| + |B| is the
total number of labeled voxels in both segmentations.

This is a wrapper around the `compare_parcellations_dice` function from lamareg,
which computes DICE scores for each unique label/ROI in the segmentation volumes.

Features:
--------
- Computes DICE scores for multi-label segmentations
- Calculates per-ROI/per-label DICE coefficients
- Outputs results in CSV format for easy analysis
- Handles both binary and multi-class segmentations
- Background label (0) is typically excluded from calculations

Command-Line Usage:
------------------
micaflow calculate_dice \\
    --input <path/to/segmentation.nii.gz> \\
    --reference <path/to/ground_truth.nii.gz> \\
    --output <path/to/results.csv>

Python API Usage:
----------------
>>> from lamareg.scripts.dice_compare import compare_parcellations_dice
>>> compare_parcellations_dice(
...     "segmentation.nii.gz",
...     "ground_truth.nii.gz",
...     "dice_results.csv"
... )

Or use the command-line interface via subprocess:
>>> import subprocess
>>> result = subprocess.run([
...     "micaflow", "calculate_dice",
...     "--input", "segmentation.nii.gz",
...     "--reference", "ground_truth.nii.gz",
...     "--output", "dice_results.csv"
... ], check=True)

Output Format:
-------------
The output CSV file contains DICE scores for each label/ROI found in the segmentations:

Header row: "Label,DICE_Score"
Data format:
  - Column 1 (Label): Integer label/ROI identifier
  - Column 2 (DICE_Score): DICE coefficient (float, 0.0-1.0)

Example CSV output:
```
Label,DICE_Score
1,0.8523
2,0.9102
3,0.7845
...
```

One row is generated for each unique label found in either segmentation.

DICE Score Interpretation:
-------------------------
- 0.00 - 0.50: Poor overlap (needs improvement)
- 0.50 - 0.70: Moderate overlap (acceptable for some applications)
- 0.70 - 0.85: Good overlap (typical for automated methods)
- 0.85 - 1.00: Excellent overlap (near-manual quality)

Exit Codes:
----------
0 : Success - DICE scores computed and saved
1 : Error - invalid inputs, file not found, or processing failure

See Also:
--------
- synthseg : For generating segmentation/parcellation volumes
- bet : For creating brain masks from parcellations

"""

import csv
import nibabel as nib
import argparse
import sys
from colorama import init, Fore, Style
from lamareg.scripts.dice_compare import compare_parcellations_dice

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
    """Print comprehensive help message with examples and interpretation guidelines."""
    
    help_text = f"""
    {CYAN}{BOLD}╔════════════════════════════════════════════════════════════════╗
    ║                     DICE SCORE CALCULATOR                      ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script calculates the DICE coefficient (Sørensen-Dice coefficient)
    between two segmentation volumes. For multi-label segmentations, it computes
    the DICE score separately for each label/ROI.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow calculate_dice {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--input{RESET}, {YELLOW}-i{RESET}     : Path to the first input segmentation volume (.nii.gz)
                      Typically the predicted/test segmentation
      {YELLOW}--reference{RESET}, {YELLOW}-r{RESET} : Path to the reference/ground truth segmentation (.nii.gz)
                      The gold standard for comparison
      {YELLOW}--output{RESET}, {YELLOW}-o{RESET}    : Output path for the CSV file with DICE scores

    {CYAN}{BOLD}──────────────────── EXAMPLE USAGE ──────────────────────{RESET}
    
    {BLUE}# Compare automated vs manual segmentation{RESET}
    micaflow calculate_dice \\
      {YELLOW}--input{RESET} predicted_segmentation.nii.gz \\
      {YELLOW}--reference{RESET} ground_truth.nii.gz \\
      {YELLOW}--output{RESET} dice_scores.csv
    
    {BLUE}# Validate SynthSeg output against manual parcellation{RESET}
    micaflow calculate_dice \\
      {YELLOW}--input{RESET} subject_synthseg.nii.gz \\
      {YELLOW}--reference{RESET} manual_parcellation.nii.gz \\
      {YELLOW}--output{RESET} validation_metrics.csv
    
    {BLUE}# Compare two registration-based segmentations{RESET}
    micaflow calculate_dice \\
      {YELLOW}--input{RESET} atlas_warped_seg.nii.gz \\
      {YELLOW}--reference{RESET} subject_manual_seg.nii.gz \\
      {YELLOW}--output{RESET} registration_quality.csv
    
    {CYAN}{BOLD}──────────────────── OUTPUT FORMAT ──────────────────────{RESET}
    The output CSV file contains two columns with a header row:
    
    {MAGENTA}Header:{RESET} Label,DICE_Score
    {MAGENTA}Row format:{RESET}
      • {YELLOW}Label{RESET}       : Integer label/ROI identifier (e.g., 1, 2, 3...)
      • {YELLOW}DICE_Score{RESET}  : DICE coefficient (float, 0.0-1.0)
    
    Example output:
    {BLUE}Label,DICE_Score
    1,0.8523
    2,0.9102
    3,0.7845{RESET}
    
    {CYAN}{BOLD}───────────────── DICE COEFFICIENT FORMULA ───────────────{RESET}
    
    DICE = 2 × |A ∩ B| / (|A| + |B|)
    
    Where:
      • A = Set of voxels with label L in segmentation 1
      • B = Set of voxels with label L in segmentation 2
      • |A ∩ B| = Number of voxels in both A and B
      • |A| + |B| = Total count of labeled voxels
    
    {CYAN}{BOLD}─────────────────── DICE INTERPRETATION ──────────────────{RESET}
    {RED}0.00 - 0.50{RESET}: Poor overlap       {MAGENTA}→ Needs improvement{RESET}
    {YELLOW}0.50 - 0.70{RESET}: Moderate overlap  {MAGENTA}→ Acceptable for some tasks{RESET}
    {GREEN}0.70 - 0.85{RESET}: Good overlap      {MAGENTA}→ Typical for automated methods{RESET}
    {CYAN}0.85 - 1.00{RESET}: Excellent overlap {MAGENTA}→ Near-manual quality{RESET}
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    {MAGENTA}•{RESET} For multi-label segmentations, DICE is computed independently per label
    {MAGENTA}•{RESET} Labels present in one volume but not the other will have DICE = 0
    {MAGENTA}•{RESET} Background (label 0) is typically excluded from calculation
    {MAGENTA}•{RESET} Both input volumes must have the same dimensions and orientation
    {MAGENTA}•{RESET} If dimensions don't match, the script will fail with an error
    {MAGENTA}•{RESET} DICE is symmetric: DICE(A,B) = DICE(B,A)
    {MAGENTA}•{RESET} This is a wrapper around lamareg's compare_parcellations_dice function
    {MAGENTA}•{RESET} DICE is sensitive to small structures (can be low even with good alignment)
    
    {CYAN}{BOLD}────────────────────── EXIT CODES ───────────────────────{RESET}
    {GREEN}0{RESET} : Success - DICE scores computed and saved
    {RED}1{RESET} : Error - invalid inputs, file not found, or processing failure
    """
    print(help_text)


if __name__ == "__main__":
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Calculate DICE overlap coefficients between two segmentation volumes",
        add_help=False  # Use custom help
    )
    parser.add_argument(
        "--input", "-i", required=True, 
        help="Input segmentation volume (.nii.gz)"
    )
    parser.add_argument(
        "--reference", "-r", required=True, 
        help="Reference/ground truth segmentation volume (.nii.gz)"
    )
    parser.add_argument(
        "--output", "-o", required=True, 
        help="Output CSV file path for DICE scores"
    )

    args = parser.parse_args()

    try:
        print(f"{CYAN}Loading segmentation volumes...{RESET}")
        
        # Validate input files exist
        import os
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        if not os.path.exists(args.reference):
            raise FileNotFoundError(f"Reference file not found: {args.reference}")
        
        # Load images to validate dimensions
        input_img = nib.load(args.input)
        ref_img = nib.load(args.reference)
        
        print(f"  Input: {args.input} (shape: {input_img.shape})")
        print(f"  Reference: {args.reference} (shape: {ref_img.shape})")
        print(f"{CYAN}Computing DICE scores...{RESET}")
        
        # Call the actual comparison function from lamareg
        compare_parcellations_dice(args.input, args.reference, args.output)
        
        print(f"\n{GREEN}{BOLD}DICE scores successfully computed!{RESET}")
        print(f"  Results saved to: {args.output}")
        
        # Try to read and display a summary
        try:
            with open(args.output, 'r') as f:
                lines = f.readlines()
                num_labels = len(lines) - 1  # Subtract header
                if num_labels > 0:
                    print(f"  Number of labels evaluated: {num_labels}")
                    
                    # Calculate and display mean DICE
                    scores = [float(line.split(',')[1].strip()) for line in lines[1:]]
                    mean_dice = sum(scores) / len(scores)
                    print(f"  Mean DICE score: {mean_dice:.4f}")
        except Exception as e:
            pass  # Silently continue if we can't read the summary
        
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\n{RED}{BOLD}File not found:{RESET}")
        print(f"  {str(e)}")
        print(f"\n{YELLOW}Please check that the file paths are correct.{RESET}")
        sys.exit(1)
        
    except ValueError as e:
        print(f"\n{RED}{BOLD}Dimension mismatch:{RESET}")
        print(f"  {str(e)}")
        print(f"\n{YELLOW}Tip: Use 'micaflow apply_warp' to resample one segmentation to match the other.{RESET}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n{RED}{BOLD}Error during DICE calculation:{RESET}")
        print(f"  {str(e)}")
        print(f"\n{YELLOW}Run 'micaflow calculate_dice --help' for usage information.{RESET}")
        sys.exit(1)
