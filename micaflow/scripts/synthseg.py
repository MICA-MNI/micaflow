"""
synthseg - Neural Network-Based Brain MRI Segmentation

Part of the micaflow processing pipeline for neuroimaging data.

This module provides an interface to SynthSeg, a deep learning-based tool for automated
brain MRI segmentation that works across different MRI contrasts without retraining.
SynthSeg segments brain anatomical structures in T1w, T2w, FLAIR, and other MR contrasts,
providing both whole-brain segmentation and optional cortical parcellation.

What is SynthSeg?
----------------
SynthSeg is a contrast-agnostic segmentation tool that:
- Segments brain MRI scans regardless of acquisition contrast
- Works with T1w, T2w, FLAIR, PD, and other MRI modalities
- Provides robust segmentation across different scanners and protocols

How It Works:
1. Load input MRI (any contrast)
2. Optional preprocessing (cropping, resampling)
3. Run 3D U-Net inference
4. Optional postprocessing (CRF, largest component)
5. Output segmentation labels
6. Optional: Compute volumes and QC scores

Segmentation Labels:
-------------------
Standard mode (37 labels):
- Cortical gray matter (left/right)
- White matter (left/right)
- Lateral ventricles (left/right)
- Cerebellum (left/right)
- Thalamus (left/right)
- Caudate (left/right)
- Putamen (left/right)
- Pallidum (left/right)
- Hippocampus (left/right)
- Amygdala (left/right)
- Brainstem
- ... and more subcortical structures

Parcellation mode (up to 132 labels):
- All standard labels
- Cortical parcellation (FreeSurfer-style)
- Frontal, parietal, temporal, occipital regions
- Gyral and sulcal subdivisions

Features:
--------
- Contrast-agnostic: Works with T1w, T2w, FLAIR, PD, etc.
- Whole-brain segmentation: 37 anatomical structures
- Cortical parcellation: Optional FreeSurfer-style parcellation
- Batch processing: Process multiple subjects efficiently
- Multi-threading: CPU parallelization support

Command-Line Usage:
------------------
# Basic segmentation
micaflow synthseg \\
    --i <path/to/image.nii.gz> \\
    --o <path/to/segmentation.nii.gz>

# With cortical parcellation (detailed)
micaflow synthseg \\
    --i <path/to/image.nii.gz> \\
    --o <path/to/segmentation.nii.gz> \\
    --parc

# Robust mode for highest quality
micaflow synthseg \\
    --i <path/to/image.nii.gz> \\
    --o <path/to/segmentation.nii.gz> \\
    --robust

# Fast mode for quick processing
micaflow synthseg \\
    --i <path/to/image.nii.gz> \\
    --o <path/to/segmentation.nii.gz> \\
    --fast

# With volume measurements
micaflow synthseg \\
    --i <path/to/image.nii.gz> \\
    --o <path/to/segmentation.nii.gz> \\
    --vol <path/to/volumes.csv>

# With quality control scores
micaflow synthseg \\
    --i <path/to/image.nii.gz> \\
    --o <path/to/segmentation.nii.gz> \\
    --qc <path/to/qc_scores.csv>

# Batch processing with all outputs
micaflow synthseg \\
    --i <path/to/input_folder/> \\
    --o <path/to/output_folder/> \\
    --parc \\
    --vol <path/to/volumes.csv> \\
    --qc <path/to/qc_scores.csv> \\
    --post <path/to/posteriors/> \\
    --threads 8

# CPU-only execution
micaflow synthseg \\
    --i <path/to/image.nii.gz> \\
    --o <path/to/segmentation.nii.gz> \\
    --cpu \\
    --threads 8

# CT scan segmentation
micaflow synthseg \\
    --i <path/to/ct_scan.nii.gz> \\
    --o <path/to/segmentation.nii.gz> \\
    --ct

Python API Usage:
----------------
>>> from micaflow.scripts.synthseg import main
>>> import sys
>>> 
>>> # Basic usage
>>> sys.argv = [
...     'synthseg',
...     '--i', 'T1w.nii.gz',
...     '--o', 'segmentation.nii.gz'
... ]
>>> main()
>>> 
>>> # With options (alternative approach)
>>> main({
...     'i': 'T1w.nii.gz',
...     'o': 'segmentation.nii.gz',
...     'parc': True,
...     'robust': True,
...     'vol': 'volumes.csv',
...     'qc': 'qc_scores.csv',
...     'threads': 8,
...     'cpu': False
... })

Pipeline Integration:
--------------------
SynthSeg is typically used early in the processing pipeline for:
1. Quality control of acquisitions
2. Brain masking
3. Registration targets
4. Volume-based analysis

Structural MRI Pipeline:
1. Segmentation (synthseg)
2. Surface reconstruction (optional)
3. Registration to template
4. Morphometric analysis

Multimodal Pipeline:
1. Structural segmentation (synthseg)
2. Functional/diffusion preprocessing
3. Coregistration to anatomical
4. ROI-based analysis using segmentation

Exit Codes:
----------
0 : Success - segmentation completed
1 : Error - invalid inputs, file not found, or processing failure

Limitations:
-----------
- Optimized for adult brain anatomy
- May struggle with severe pathology

See Also:
--------
- coregister : Register segmentation to functional data
- apply_warp : Apply transformations to segmentation
- extract_rois : Extract ROI time series using segmentation

References:
----------
1. Billot B, Greve DN, Puonti O, et al. SynthSeg: Segmentation of brain MRI scans 
   of any contrast and resolution without retraining. Med Image Anal. 
   2023;86:102789. doi:10.1016/j.media.2023.102789

"""

# python imports
import os
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from colorama import init, Fore, Style
from lamareg.scripts.synthseg import main

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


def print_extended_help():

    help_text = f"""
    {CYAN}{BOLD}╔════════════════════════════════════════════════════════════════╗
    ║                         SYNTHSEG                               ║
    ║              Contrast-Agnostic Brain Segmentation              ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script runs the SynthSeg neural network for automated brain MRI
    segmentation. It works across different MRI contrasts (T1w, T2w, FLAIR, etc.)
    without retraining, providing robust anatomical segmentation.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow synthseg {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--i{RESET} PATH       : Input image(s) to segment (file or folder)
      {YELLOW}--o{RESET} PATH       : Output segmentation file(s) or folder
    
    {CYAN}{BOLD}─────────────────── OPTIONAL ARGUMENTS ───────────────────{RESET}
      {YELLOW}--parc{RESET}         : Enable cortical parcellation (132 labels vs 37)
      {YELLOW}--robust{RESET}       : Use robust mode (higher quality, ~5x slower)
      {YELLOW}--fast{RESET}         : Faster processing (minimal postprocessing)
      {YELLOW}--threads{RESET} N    : Number of CPU threads (default: 1)
      {YELLOW}--cpu{RESET}          : Force CPU processing (instead of GPU)
      {YELLOW}--vol{RESET} PATH     : Output CSV file with region volumes
      {YELLOW}--qc{RESET} PATH      : Output CSV file with quality control scores
      {YELLOW}--post{RESET} PATH    : Output posterior probability maps
      {YELLOW}--resample{RESET} PATH: Output resampled images
      {YELLOW}--crop{RESET} N [N ...]: Size of 3D patches (default: 192)
      {YELLOW}--ct{RESET}           : Clip intensities [0,80] for CT scans
      {YELLOW}--v1{RESET}           : Use SynthSeg 1.0 (legacy version)
    
    {CYAN}{BOLD}────────────────── EXAMPLE USAGE ────────────────────────{RESET}
    
    {BLUE}# Example 1: Basic segmentation{RESET}
    micaflow synthseg \\
      {YELLOW}--i{RESET} T1w.nii.gz \\
      {YELLOW}--o{RESET} segmentation.nii.gz
    
    {BLUE}# Example 2: With cortical parcellation{RESET}
    micaflow synthseg \\
      {YELLOW}--i{RESET} T1w.nii.gz \\
      {YELLOW}--o{RESET} segmentation.nii.gz \\
      {YELLOW}--parc{RESET}
    
    {BLUE}# Example 3: Robust mode for research{RESET}
    micaflow synthseg \\
      {YELLOW}--i{RESET} T1w.nii.gz \\
      {YELLOW}--o{RESET} segmentation.nii.gz \\
      {YELLOW}--parc{RESET} \\
      {YELLOW}--robust{RESET}
    
    {BLUE}# Example 4: Batch processing with volumes{RESET}
    micaflow synthseg \\
      {YELLOW}--i{RESET} input_folder/ \\
      {YELLOW}--o{RESET} output_folder/ \\
      {YELLOW}--vol{RESET} volumes.csv \\
      {YELLOW}--qc{RESET} qc_scores.csv \\
      {YELLOW}--threads{RESET} 8
    
    {BLUE}# Example 5: Fast mode for exploration{RESET}
    micaflow synthseg \\
      {YELLOW}--i{RESET} T2w.nii.gz \\
      {YELLOW}--o{RESET} segmentation.nii.gz \\
      {YELLOW}--fast{RESET}
    
    {BLUE}# Example 6: CPU-only with all outputs{RESET}
    micaflow synthseg \\
      {YELLOW}--i{RESET} FLAIR.nii.gz \\
      {YELLOW}--o{RESET} segmentation.nii.gz \\
      {YELLOW}--parc{RESET} \\
      {YELLOW}--vol{RESET} volumes.csv \\
      {YELLOW}--qc{RESET} qc_scores.csv \\
      {YELLOW}--post{RESET} posteriors.nii.gz \\
      {YELLOW}--cpu{RESET} \\
      {YELLOW}--threads{RESET} 8
    
    {CYAN}{BOLD}───────────────── COMMON ISSUES ─────────────────────────{RESET}
    {YELLOW}Issue:{RESET} Poor segmentation quality
    {GREEN}Solution:{RESET} Try --robust mode, check image quality, verify contrast
    
    {YELLOW}Issue:{RESET} Out of memory error
    {GREEN}Solution:{RESET} Use --crop for smaller patches, or --cpu mode
    
    {YELLOW}Issue:{RESET} Very slow processing
    {GREEN}Solution:{RESET} Use GPU if available, or --fast mode for quick results
    
    {YELLOW}Issue:{RESET} Missing structures in segmentation
    {GREEN}Solution:{RESET} Check QC scores, use --robust mode, verify input quality
    """
    print(help_text)


if __name__ == "__main__":
    # Check if help flags are provided or no arguments
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_extended_help()
        sys.exit(0)

    # parse arguments
    parser = ArgumentParser(
        description="SynthSeg: Contrast-agnostic deep learning brain MRI segmentation",
        epilog="For more details see: https://github.com/BBillot/SynthSeg",
        formatter_class=RawDescriptionHelpFormatter,
        add_help=False  # Use custom help
    )

    # input/outputs
    parser.add_argument(
        "--i", 
        required=True,
        help="Image(s) to segment. Can be a path to an image or to a folder."
    )
    parser.add_argument(
        "--o",
        required=True,
        help="Segmentation output(s). Must be a folder if --i designates a folder.",
    )
    parser.add_argument(
        "--parc",
        action="store_true",
        help="(optional) Perform cortical parcellation (132 labels instead of 37).",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="(optional) Use robust mode with test-time augmentation (slower but higher quality).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="(optional) Bypass postprocessing for faster predictions (may reduce quality).",
    )
    parser.add_argument(
        "--ct",
        action="store_true",
        help="(optional) Clip intensities to [0,80] for CT scans.",
    )
    parser.add_argument(
        "--vol",
        help="(optional) Path to output CSV file with volumes (mm3) for all regions and subjects.",
    )
    parser.add_argument(
        "--qc",
        help="(optional) Path to output CSV file with quality control scores for all subjects.",
    )
    parser.add_argument(
        "--post",
        help="(optional) Path to save posterior probability maps. Must be a folder if --i is a folder.",
    )
    parser.add_argument(
        "--resample",
        help="(optional) Path to save resampled images. Must be a folder if --i is a folder.",
    )
    parser.add_argument(
        "--crop",
        nargs="+",
        type=int,
        help="(optional) Size of 3D patches to analyze. Default is 192.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="(optional) Number of CPU threads to use. Default is 1.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="(optional) Force CPU processing instead of GPU (slower).",
    )
    parser.add_argument(
        "--v1",
        action="store_true",
        help="(optional) Use SynthSeg 1.0 instead of 2.0 (legacy version).",
    )

    # parse commandline
    args = vars(parser.parse_args())
    
    try:
        # Validate required files exist
        if not os.path.isdir(args['i']) and not os.path.exists(args['i']):
            raise FileNotFoundError(f"Input file/folder not found: {args['i']}")
        
        # Show configuration
        print(f"{CYAN}Configuration:{RESET}")
        print(f"  Input: {args['i']}")
        print(f"  Output: {args['o']}")
        if args.get('parc'):
            print(f"  Parcellation: {GREEN}Enabled (132 labels){RESET}")
        else:
            print(f"  Parcellation: Standard (37 labels)")
        
        if args.get('robust'):
            print(f"  Mode: {YELLOW}Robust (slower, higher quality){RESET}")
        elif args.get('fast'):
            print(f"  Mode: {GREEN}Fast (minimal postprocessing){RESET}")
        else:
            print(f"  Mode: Standard")
        
        if args.get('cpu'):
            print(f"  Device: {YELLOW}CPU (forced){RESET}")
            print(f"  Threads: {args.get('threads', 1)}")
        else:
            print(f"  Device: {GREEN}GPU (if available){RESET}")
        
        if args.get('vol'):
            print(f"  Volume output: {args['vol']}")
        if args.get('qc'):
            print(f"  QC output: {args['qc']}")
        
        print(f"\n{CYAN}Starting segmentation...{RESET}\n")
        
        # Run SynthSeg
        main(args)
        
        print(f"\n{GREEN}{BOLD}Segmentation completed successfully!{RESET}")
        print(f"  Output: {args['o']}")
        if args.get('vol'):
            print(f"  Volumes: {args['vol']}")
        if args.get('qc'):
            print(f"  QC scores: {args['qc']}")
        print()
        
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\n{RED}{BOLD}File not found:{RESET}")
        print(f"  {str(e)}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n{RED}{BOLD}Error during segmentation:{RESET}")
        print(f"  {str(e)}")
        print(f"\n{YELLOW}Run 'micaflow synthseg --help' for usage information.{RESET}")
        sys.exit(1)
