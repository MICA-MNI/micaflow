import os
import re
import inspect
import importlib
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ANSI color code pattern to remove
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

# Directory where script module files are located
SCRIPTS_DIR = Path('../micaflow/scripts')
OUTPUT_DIR = Path('./source/scripts')

def clean_help_text(text):
    """Remove ANSI color codes and escape RST special characters."""
    if text is None:
        return ""
    
    # Remove ANSI color codes
    text = ansi_escape.sub('', text)
    
    # Fix RST formatting issues
    text = text.replace('*', '\\*')
    
    return text

def extract_arguments_from_help(help_content):
    """Extract argument details from help text."""
    if not help_content:
        return {}
    
    arguments = {}
    
    # Find arguments sections in the help text
    arg_section_pattern = r'(?:REQUIRED ARGUMENTS|OPTIONAL ARGUMENTS|ARGUMENTS)[\s-]+(.*?)(?:(?=─+)|$)'
    arg_sections = re.findall(arg_section_pattern, help_content, re.DOTALL)
    
    # For each argument section, extract arguments
    for section in arg_sections:
        # Match argument patterns like:
        # --arg, -a PATH    : Description text
        # --arg PATH        : Description text
        arg_pattern = r'(?:--([a-zA-Z0-9_-]+)(?:,\s*-([a-zA-Z0-9]))?|(-[a-zA-Z0-9]),\s*--([a-zA-Z0-9_-]+))\s+([^\s:]*)?\s*:?\s*(.*?)\n'
        arg_matches = re.findall(arg_pattern, section, re.DOTALL)
        
        for match in arg_matches:
            # Handle both formats of arguments
            if match[0]:  # Format: --arg, -a
                arg_name = match[0]
                short_name = match[1] if match[1] else None
                arg_type = match[4] if match[4] else None
                description = match[5].strip()
            else:  # Format: -a, --arg
                arg_name = match[3]
                short_name = match[2] if match[2] else None
                arg_type = match[4] if match[4] else None
                description = match[5].strip()
            
            arguments[arg_name] = {
                'short_name': short_name,
                'type': arg_type,
                'description': description
            }
    
    return arguments

def extract_help_function_content(module_name):
    """Import a module and extract the content from its print_help_message function."""
    try:
        module = importlib.import_module(f"micaflow.scripts.{module_name}")
        
        # Look for the help function - different scripts use different names
        help_func_name = None
        for func_name in ["print_help_message", "print_extended_help"]:
            if hasattr(module, func_name):
                help_func_name = func_name
                break
        
        if help_func_name is None:
            print(f"Warning: No help function found in {module_name}")
            return None
        
        help_func = getattr(module, help_func_name)
        
        # If the function returns a string, capture it
        if inspect.signature(help_func).return_annotation != inspect.Signature.empty:
            return help_func()
        
        # Otherwise, temporarily capture stdout
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            help_func()
        
        return f.getvalue()
    
    except (ImportError, AttributeError) as e:
        print(f"Error processing {module_name}: {e}")
        return None

def create_script_rst_file(module_name, help_content):
    """Create an RST file for a module using the extracted help content."""
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Import the module to get its docstring
    try:
        module = importlib.import_module(f"micaflow.scripts.{module_name}")
        module_docstring = inspect.getdoc(module) or ""
    except (ImportError, AttributeError) as e:
        print(f"Error getting docstring for {module_name}: {e}")
        module_docstring = ""
    
    with open(OUTPUT_DIR / f"{module_name}.rst", "w") as f:
        title = module_name.replace('_', ' ').title()
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        
        # Add the module docstring at the top if it exists
        if module_docstring:
            f.write(f"{module_docstring}\n\n")
        # Add command line usage section
        f.write("Command Line Usage\n")
        f.write("-----------------\n\n")
        f.write(".. code-block:: bash\n\n")
        f.write(f"    micaflow {module_name} [options]\n\n")
        
        # Add source code links
        f.write("Source Code\n")
        f.write("-----------\n\n")
        f.write(f"View the source code: `GitHub Repository <https://github.com/MICA-LAB/micaflow/blob/main/micaflow/scripts/{module_name}.py>`_\n\n")
        
        # If we have help content, add it
        if help_content:
            # Clean the help text
            clean_content = clean_help_text(help_content)
            
            # Extract any available arguments from the help content
            arguments = extract_arguments_from_help(clean_content)
            
            # Add arguments section if we found any
            if arguments:
                f.write("Arguments\n")
                f.write("---------\n\n")
                
                # First, find required arguments
                required_args = {name: details for name, details in arguments.items() 
                               if "required" in details.get('description', '').lower() or
                                  "must be provided" in details.get('description', '').lower()}
                
                if required_args:
                    f.write("Required Arguments:\n\n")
                    f.write(".. list-table::\n")
                    f.write("   :widths: 20 10 70\n")
                    f.write("   :header-rows: 1\n\n")
                    f.write("   * - Argument\n")
                    f.write("     - Type\n")
                    f.write("     - Description\n")
                    
                    for arg_name, details in required_args.items():
                        arg_text = f"--{arg_name}"
                        if details.get('short_name'):
                            arg_text += f", -{details['short_name']}"
                        
                        f.write(f"   * - {arg_text}\n")
                        f.write(f"     - {details.get('type', '')}\n")
                        f.write(f"     - {details.get('description', '')}\n")
                    
                    f.write("\n")
                
                # Then add optional arguments
                optional_args = {name: details for name, details in arguments.items() 
                               if name not in required_args}
                
                if optional_args:
                    f.write("Optional Arguments:\n\n")
                    f.write(".. list-table::\n")
                    f.write("   :widths: 20 10 70\n")
                    f.write("   :header-rows: 1\n\n")
                    f.write("   * - Argument\n")
                    f.write("     - Type\n")
                    f.write("     - Description\n")
                    
                    for arg_name, details in optional_args.items():
                        arg_text = f"--{arg_name}"
                        if details.get('short_name'):
                            arg_text += f", -{details['short_name']}"
                        
                        f.write(f"   * - {arg_text}\n")
                        f.write(f"     - {details.get('type', '')}\n")
                        f.write(f"     - {details.get('description', '')}\n")
                    
                    f.write("\n")
            
            # Add description section
            f.write("Description\n")
            f.write("-----------\n\n")
            
            # Remove the fancy box header and extract just the description
            lines = clean_content.split('\n')
            description_started = False
            description_lines = []
            
            for line in lines:
                # Skip initial box drawing and header
                if "═══" in line or "╔" in line or "╚" in line:
                    continue
                
                # Start capturing after the title box
                if line.strip() and not description_started and not line.strip().isupper():
                    description_started = True
                
                if description_started:
                    # Stop when we hit a section header
                    if "─────" in line:
                        break
                    description_lines.append(line)
            
            # Join the description lines and add them to the RST file
            description = "\n".join(description_lines).strip()
            f.write(f"{description}\n\n")
            
            # Add the full help text in a code block
            f.write("Full Help\n")
            f.write("---------\n\n")
            f.write(".. code-block:: text\n\n")
            
            # Indent each line for the code block
            for line in clean_content.split('\n'):
                f.write(f"    {line}\n")
        
        # Move automodule directive to the end to avoid disrupting the flow

def create_scripts_index():
    """Create the index file that will list all script modules."""
    with open(OUTPUT_DIR / "index.rst", "w") as f:
        f.write("Scripts Reference\n")
        f.write("===============\n\n")
        f.write("This section provides detailed documentation for each command-line utility included in MicaFlow.\n\n")
        f.write("Each script can be run independently with the `micaflow [script_name]` command.\n\n")
        
        # Add a quick reference table
        f.write("Quick Reference\n")
        f.write("-------------\n\n")
        f.write(".. list-table::\n")
        f.write("   :widths: 30 70\n")
        f.write("   :header-rows: 1\n\n")
        f.write("   * - Script\n")
        f.write("     - Description\n")
        
        # List all script modules with short descriptions
        script_files = [p.stem for p in Path(SCRIPTS_DIR).glob("*.py") 
                       if p.stem not in ["__init__", "util_bids_pathing"]]
        
        for script in sorted(script_files):
            short_desc = get_short_description(script)
            f.write(f"   * - :doc:`{script}`\n")
            f.write(f"     - {short_desc}\n")
        
        f.write("\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 1\n")
        f.write("   :hidden:\n\n")
        
        # List all script modules
        for script in sorted(script_files):
            f.write(f"   {script}\n")

def get_short_description(script_name):
    """Extract a short description for a script."""
    descriptions = {
        "bet": "Brain extraction using HD-BET",
        "synthseg": "Deep learning segmentation with SynthSeg",
        "coregister": "Image coregistration using ANTs",
        "apply_warp": "Apply transformations to images",
        "SDC": "Susceptibility distortion correction",
        "apply_SDC": "Apply precomputed distortion correction",
        "bias_correction": "N4 bias field correction",
        "calculate_jaccard": "Calculate similarity between segmentations",
        "compute_fa_md": "Compute DTI metrics (FA, MD)",
        "denoise": "Denoise diffusion-weighted images",
        "motion_correction": "Motion correction for DWI",
        "texture_generation": "Generate texture features",
        "normalize": "Intensity normalization and clamping",
    }
    return descriptions.get(script_name, "MicaFlow utility")

def main():
    """Generate RST files for all script modules."""
    # Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process all Python files in the scripts directory
    script_files = [p.stem for p in Path(SCRIPTS_DIR).glob("*.py") 
                   if p.stem not in ["__init__", "util_bids_pathing"]]
    
    for script_name in script_files:
        print(f"Processing {script_name}...")
        help_content = extract_help_function_content(script_name)
        create_script_rst_file(script_name, help_content)
    
    print("Done!")

if __name__ == "__main__":
    main()