"""HD-BET (High-Definition Brain Extraction Tool) wrapper script.

This script provides a simplified command-line interface to the HD-BET brain extraction
tool, which performs accurate skull stripping on brain MR images using a deep learning approach.
It supports both CPU and GPU execution modes.

The script is a wrapper around the HD-BET entry_point.py script that simplifies the interface
and handles path resolution.

Example:
    python hdbet.py --input t1w.nii.gz --output t1w_brain.nii.gz
    python hdbet.py --input t1w.nii.gz --output t1w_brain.nii.gz --cpu

"""

import subprocess
import argparse
import os
import shutil
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform brain extraction using HD-BET")
    parser.add_argument("--input", "-i", required=True, help="Input MR image file")
    parser.add_argument(
        "--output", "-o", required=True, help="Output brain-extracted image file"
    )
    parser.add_argument(
        "--output-mask", "-m", required=True, help="Output brain-extracted mask image file"
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()
    input_abs_path = os.path.abspath(args.input)

    subprocess.run(
        "python3 scripts/HD_BET/entry_point.py -i "
        + input_abs_path
        + " -o "
        + args.output
        + " --save_bet_mask" + (" -device cpu --disable_tta" if args.cpu else ""),
        shell=True,
    )
    shutil.move(args.output.replace(".nii.gz", "") + "_bet" + ".nii.gz", args.output_mask)
