import os
import re
from pathlib import Path

# Directory where script module files are located
SCRIPTS_DIR = Path('../micaflow/scripts')

# Descriptions for each script
descriptions = {
    "bet": "Brain extraction using HD-BET.",
    "synthseg": "Deep learning segmentation with SynthSeg.",
    "coregister": "Image coregistration using ANTs.",
    "apply_warp": "Apply transformations to images.",
    "SDC": "Susceptibility distortion correction.",
    "apply_SDC": "Apply precomputed distortion correction.",
    "bias_correction": "N4 bias field correction.",
    "calculate_jaccard": "Calculate similarity between segmentations.",
    "compute_fa_md": "Compute DTI metrics (FA, MD).",
    "denoise": "Denoise diffusion-weighted images.",
    "motion_correction": "Motion correction for DWI.",
    "texture_generation": "Generate texture features."
}

def add_module_docstring(file_path, description):
    """Add a module-level docstring to a Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if the file already has a module docstring
    if re.search(r'""".*?"""', content, re.DOTALL):
        print(f"File {file_path} already has a docstring.")
        return
    
    # Find the insertion point (after imports and before first function/class definition)
    lines = content.split('\n')
    insertion_point = 0
    
    # Skip shebang line and imports
    for i, line in enumerate(lines):
        if line.startswith('#!') or line.startswith('import ') or line.startswith('from '):
            insertion_point = i + 1
        else:
            break
    
    # Insert the docstring
    docstring = f'"""{description}\n\nThis module provides the command-line interface for the {Path(file_path).stem} command.\n"""\n\n'
    lines.insert(insertion_point, docstring)
    
    # Write the file back
    with open(file_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Added docstring to {file_path}")

def main():
    """Add docstrings to all script modules."""
    for script_file in Path(SCRIPTS_DIR).glob("*.py"):
        if script_file.stem in ["__init__", "util_bids_pathing"]:
            continue
        
        description = descriptions.get(script_file.stem, f"MicaFlow {script_file.stem} utility.")
        add_module_docstring(script_file, description)
    
    print("Done!")

if __name__ == "__main__":
    main()