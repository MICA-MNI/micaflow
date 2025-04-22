"""
synthseg - Neural Network-Based Brain MRI Segmentation

Part of the micaflow processing pipeline for neuroimaging data.

This module provides an interface to SynthSeg, a deep learning-based tool for automated 
brain MRI segmentation that works across different MRI contrasts without retraining. 
SynthSeg segments brain anatomical structures in T1w, T2w, FLAIR, and other MR contrasts, 
providing both whole-brain segmentation and optional cortical parcellation.

Features:
--------
- Contrast-agnostic segmentation working across different MRI acquisition types
- Whole-brain anatomical structure segmentation with 37 labels
- Optional cortical parcellation (up to 95 additional regions)
- Multiple execution modes: standard, robust (higher quality), and fast
- Volumetric analysis with CSV output for region-wise measurements
- Quality control metrics for assessing segmentation reliability
- GPU acceleration with optional CPU-only execution

API Usage:
---------
micaflow synthseg 
    --i <path/to/image.nii.gz>
    --o <path/to/segmentation.nii.gz>
    [--parc]
    [--robust]
    [--fast]
    [--vol <path/to/volumes.csv>]
    [--qc <path/to/qc_scores.csv>]
    [--threads <num_threads>]

Python Usage:
-----------
>>> from micaflow.scripts.synthseg import main
>>> main({
...     'i': 'input_image.nii.gz',
...     'o': 'segmentation.nii.gz',
...     'parc': True,
...     'robust': False,
...     'fast': True,
...     'vol': 'volumes.csv',
...     'threads': 4
... })

"""

# python imports
import os
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from colorama import init, Fore, Style
from lamar.scripts.synthseg import main
init()

def print_extended_help():
    """Print extended help message with examples and usage instructions."""
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
    ║                         SYNTHSEG                               ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script runs the SynthSeg neural network-based tool for brain MRI
    segmentation. It provides automated segmentation of anatomical structures
    even across different contrasts and acquisition types.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow synthseg {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--i{RESET} PATH       : Input image(s) to segment (file or folder)
      {YELLOW}--o{RESET} PATH       : Output segmentation file(s) or folder
    
    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--parc{RESET}         : Enable cortical parcellation
      {YELLOW}--robust{RESET}       : Use robust mode (slower but better quality)
      {YELLOW}--fast{RESET}         : Faster processing (less postprocessing)
      {YELLOW}--threads{RESET} N    : Set number of CPU threads (default: 1)
      {YELLOW}--cpu{RESET}          : Force CPU processing (instead of GPU)
      {YELLOW}--vol{RESET} PATH     : Output volumetric CSV file
      {YELLOW}--qc{RESET} PATH      : Output quality control scores CSV file
      {YELLOW}--post{RESET} PATH    : Output posterior probability maps
      {YELLOW}--resample{RESET} PATH: Output resampled images
      {YELLOW}--crop{RESET} N [N ...]: Size of 3D patches to analyze (default: 192)
      {YELLOW}--ct{RESET}           : Clip intensities for CT scans [0,80]
      {YELLOW}--v1{RESET}           : Use SynthSeg 1.0 instead of 2.0
    
    {CYAN}{BOLD}────────────────── EXAMPLE USAGE ────────────────────────{RESET}
    
    {BLUE}# Basic segmentation{RESET}
    micaflow synthseg \\
      {YELLOW}--i{RESET} t1w_scan.nii.gz \\
      {YELLOW}--o{RESET} segmentation.nii.gz
    
    {BLUE}# With cortical parcellation{RESET}
    micaflow synthseg \\
      {YELLOW}--i{RESET} t1w_scan.nii.gz \\
      {YELLOW}--o{RESET} segmentation.nii.gz \\
      {YELLOW}--parc{RESET}
    
    {BLUE}# Batch processing with volume calculation{RESET}
    micaflow synthseg \\
      {YELLOW}--i{RESET} input_folder/ \\
      {YELLOW}--o{RESET} output_folder/ \\
      {YELLOW}--vol{RESET} volumes.csv
    
    {CYAN}{BOLD}────────────────────────── NOTES ───────────────────────{RESET}
    {MAGENTA}•{RESET} SynthSeg works with any MRI contrast without retraining
    {MAGENTA}•{RESET} GPU acceleration is used by default for faster processing
    {MAGENTA}•{RESET} The robust mode provides better quality but is slower
    {MAGENTA}•{RESET} For batch processing, input and output paths must be folders
    """
    print(help_text)
    
if __name__ == '__main__':
    # Check if help flags are provided or no arguments
  if len(sys.argv) == 1 or '-h' in sys.argv or '--help' in sys.argv:
      print_extended_help()
      sys.exit(0)

  # parse arguments
  parser = ArgumentParser(
      description="SynthSeg: Deep learning tool for brain MRI segmentation", 
      epilog="For more details see: https://github.com/BBillot/SynthSeg",
      formatter_class=RawDescriptionHelpFormatter
  )

  # input/outputs
  parser.add_argument("--i", help="Image(s) to segment. Can be a path to an image or to a folder.")
  parser.add_argument("--o", help="Segmentation output(s). Must be a folder if --i designates a folder.")
  parser.add_argument("--parc", action="store_true", help="(optional) Whether to perform cortex parcellation.")
  parser.add_argument("--robust", action="store_true", help="(optional) Whether to use robust predictions (slower).")
  parser.add_argument("--fast", action="store_true", help="(optional) Bypass some postprocessing for faster predictions.")
  parser.add_argument("--ct", action="store_true", help="(optional) Clip intensities to [0,80] for CT scans.")
  parser.add_argument("--vol", help="(optional) Path to output CSV file with volumes (mm3) for all regions and subjects.")
  parser.add_argument("--qc", help="(optional) Path to output CSV file with qc scores for all subjects.")
  parser.add_argument("--post", help="(optional) Posteriors output(s). Must be a folder if --i designates a folder.")
  parser.add_argument("--resample", help="(optional) Resampled image(s). Must be a folder if --i designates a folder.")
  parser.add_argument("--crop", nargs='+', type=int, help="(optional) Size of 3D patches to analyse. Default is 192.")
  parser.add_argument("--threads", type=int, default=1, help="(optional) Number of cores to be used. Default is 1.")
  parser.add_argument("--cpu", action="store_true", help="(optional) Enforce running with CPU rather than GPU.")
  parser.add_argument("--v1", action="store_true", help="(optional) Use SynthSeg 1.0 (updated 25/06/22).")

  # parse commandline
  args = vars(parser.parse_args())
  main(args)