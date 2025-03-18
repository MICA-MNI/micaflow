#!/usr/bin/env python3
import argparse
import os
import sys
import pkg_resources
import subprocess
import shutil

def get_snakefile_path():
    """Get the path to the Snakefile within the installed package."""
    return pkg_resources.resource_filename("micaflow", "resources/Snakefile")

def print_extended_help():
    """Print an extended help message with examples."""
    help_msg = """
    MicaFlow: MRI Processing Pipeline
    ================================
    
    USAGE:
      micaflow [options]
    
    REQUIRED PARAMETERS:
      --subject SUBJECT_ID           Subject ID
      --out-dir OUTPUT_DIR           Output directory
      --data-directory DATA_DIR      Input data directory 
      --t1w-file T1W_FILE            T1-weighted image file
    
    OPTIONAL PARAMETERS:
      --session SESSION_ID           Session ID (default: none)
      --flair-file FLAIR_FILE        FLAIR image file
      --run-dwi                      Enable diffusion processing
      --dwi-file DWI_FILE            Diffusion weighted image
      --bval-file BVAL_FILE          B-value file for DWI
      --bvec-file BVEC_FILE          B-vector file for DWI
      --inverse-dwi-file INV_FILE    Inverse (PA) DWI for distortion correction
      --threads N                    Number of threads (default: 1)
      --cpu                          Force CPU computation
      --cores N                      Number of CPU cores to use (default: 1)
      --dry-run, -n                  Dry run (don't execute commands)
      --config-file FILE             Path to a YAML configuration file
    
    EXAMPLES:
      # Process a single subject with T1w only
      micaflow --subject sub-001 --session ses-01 \\
        --data-directory /data --t1w-file sub-001_ses-01_T1w.nii.gz --out-dir /output --cores 4
    
      # Process with FLAIR
      micaflow --subject sub-001 --session ses-01 \\
        --data-directory /data --t1w-file sub-001_ses-01_T1w.nii.gz \\
        --flair-file sub-001_ses-01_FLAIR.nii.gz --out-dir /output --cores 4
    
      # Process with diffusion data
      micaflow --subject sub-001 --session ses-01 \\
        --data-directory /data --t1w-file sub-001_ses-01_T1w.nii.gz \\
        --run-dwi --dwi-file sub-001_ses-01_dwi.nii.gz \\
        --bval-file sub-001_ses-01_dwi.bval --bvec-file sub-001_ses-01_dwi.bvec \\
        --inverse-dwi-file sub-001_ses-01_acq-PA_dwi.nii.gz --out-dir /output --cores 4
        
      # Using a config file
      micaflow --config-file config.yaml --cores 4
    """
    print(help_msg)

def main():
    # If no arguments provided, show help and exit
    if len(sys.argv) == 1:
        print_extended_help()
        sys.exit(0)
        
    # Create custom formatter that includes our extended help
    class CustomHelpFormatter(argparse.HelpFormatter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, width=100)
        
        def format_help(self):
            # Include standard argparse help and our extended help
            standard_help = super().format_help()
            if "-h" in sys.argv or "--help" in sys.argv:
                return standard_help + "\n\n" + print_extended_help()
            return standard_help
    
    parser = argparse.ArgumentParser(
        description="Run the micaflow MRI processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    
    # Add arguments that match the config parameters
    # Pipeline command (default)
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the full micaflow pipeline")
    # Add pipeline arguments
    pipeline_parser.add_argument("--subject", help="Subject ID (e.g., sub-01)")
    pipeline_parser.add_argument("--session", help="Session ID (e.g., ses-01)")
    pipeline_parser.add_argument("--out-dir", help="Output directory")
    pipeline_parser.add_argument("--data-directory", default='', help="Data directory path")
    pipeline_parser.add_argument("--flair-file", help="Path to FLAIR image")
    pipeline_parser.add_argument("--t1w-file", help="Path to T1w image")
    pipeline_parser.add_argument("--dwi-file", help="Path to DWI image")
    pipeline_parser.add_argument("--bval-file", help="Path to bval file")
    pipeline_parser.add_argument("--bvec-file", help="Path to bvec file")
    pipeline_parser.add_argument("--inverse-dwi-file", help="Path to inverse DWI file")
    pipeline_parser.add_argument("--run-dwi", action="store_true", help="Run DWI processing")
    pipeline_parser.add_argument("--cpu", action="store_true", help="Use CPU computation")
    pipeline_parser.add_argument("--threads", type=int, default=1, help="Number of threads to use")
    pipeline_parser.add_argument("--dry-run", "-n", action="store_true", help="Dry run (don't execute commands)")
    pipeline_parser.add_argument("--cores", type=int, default=1, help="Number of CPU cores to use")
    pipeline_parser.add_argument("--config-file", help="Path to a YAML configuration file")
    pipeline_parser.add_argument("--snakemake-args", nargs=argparse.REMAINDER, help="Additional arguments to pass to Snakemake")
    
    
    # SynthSeg command
    synthseg_parser = subparsers.add_parser("synthseg", help="Run SynthSeg brain segmentation")
    synthseg_parser.add_argument("--i", help="Image(s) to segment. Can be a path to an image or to a folder.")
    synthseg_parser.add_argument("--o", help="Segmentation output(s). Must be a folder if --i designates a folder.")
    synthseg_parser.add_argument("--parc", action="store_true", help="(optional) Whether to perform cortex parcellation.")
    synthseg_parser.add_argument("--robust", action="store_true", help="(optional) Whether to use robust predictions (slower).")
    synthseg_parser.add_argument("--fast", action="store_true", help="(optional) Bypass some postprocessing for faster predictions.")
    synthseg_parser.add_argument("--ct", action="store_true", help="(optional) Clip intensities to [0,80] for CT scans.")
    synthseg_parser.add_argument("--vol", help="(optional) Path to output CSV file with volumes (mm3) for all regions and subjects.")
    synthseg_parser.add_argument("--qc", help="(optional) Path to output CSV file with qc scores for all subjects.")
    synthseg_parser.add_argument("--post", help="(optional) Posteriors output(s). Must be a folder if --i designates a folder.")
    synthseg_parser.add_argument("--resample", help="(optional) Resampled image(s). Must be a folder if --i designates a folder.")
    synthseg_parser.add_argument("--crop", nargs='+', type=int, help="(optional) Size of 3D patches to analyse. Default is 192.")
    synthseg_parser.add_argument("--threads", help="(optional) Number of cores to be used. Default is 1.")
    synthseg_parser.add_argument("--cpu", action="store_true", help="(optional) Enforce running with CPU rather than GPU.")
    synthseg_parser.add_argument("--v1", action="store_true", help="(optional) Use SynthSeg 1.0 (updated 25/06/22).")
    
    args = parser.parse_args()
    
    # If no command is provided, default to pipeline
    if not args.command:
        args.command = "pipeline"
    
    if args.command == "pipeline":
        # Get the path to the Snakefile
        snakefile = get_snakefile_path()
        
        # Build the snakemake command
        cmd = ["snakemake", "-s", snakefile]
        
        # Add config parameters if provided
        config = {}
        for param in ["subject", "session", "out_dir", "data_directory", "flair_file", 
                     "t1w_file", "dwi_file", "bval_file", "bvec_file", "inverse_dwi_file", 
                     "threads"]:
            if getattr(args, param.replace("-", "_"), None):
                config[param] = getattr(args, param.replace("-", "_"))
        
        if args.run_dwi:
            config["run_dwi"] = "True"
        if args.cpu:
            config["cpu"] = "True"
        
        # Add config parameters to command
        for key, value in config.items():
            cmd.extend(["--config", f"{key}={value}"])
        
        # Add config file if provided
        if args.config_file:
            cmd.extend(["--configfile", args.config_file])
        
        # Add other snakemake parameters
        if args.dry_run:
            cmd.append("-n")
        
        cmd.extend(["--cores", str(args.cores)])
        
        # Add any additional snakemake arguments
        if args.snakemake_args:
            cmd.extend(args.snakemake_args)
        
        print(f"Executing: {' '.join(cmd)}")
        
        # Execute the snakemake command
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running snakemake: {e}")
            sys.exit(1)
    
    elif args.command == "synthseg":
        # Prepare arguments for SynthSeg
        synthseg_args = []
        for arg_name, arg_value in vars(args).items():
            if arg_name != "command" and arg_value is not None:
                if isinstance(arg_value, bool):
                    if arg_value:
                        synthseg_args.append(f"--{arg_name}")
                elif isinstance(arg_value, list):
                    synthseg_args.append(f"--{arg_name}")
                    synthseg_args.extend([str(x) for x in arg_value])
                else:
                    synthseg_args.append(f"--{arg_name}")
                    synthseg_args.append(str(arg_value))
        subprocess.run(["python", "-m", "micaflow.scripts.util_synthseg"] + synthseg_args)


if __name__ == "__main__":
    main()