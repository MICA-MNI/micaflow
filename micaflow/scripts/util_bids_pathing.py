
import os
import sys
import torch
import tensorflow as tf
from colorama import init, Fore, Style

init()

# ANSI color codes for terminal output
CYAN = Fore.CYAN
GREEN = Fore.GREEN
YELLOW = Fore.YELLOW
BLUE = Fore.BLUE
MAGENTA = Fore.MAGENTA
BOLD = Style.BRIGHT
RED = Fore.RED
RESET = Style.RESET_ALL

def print_note(message, message2=None):
    """Print a note message in green."""
    print(f"{GREEN}[NOTE]{RESET} {message}")
    if message2:
        print(f"{GREEN}[NOTE]{RESET} {message2}")

def print_warning(message):
    """Print a warning message in yellow."""
    print(f"{YELLOW}[WARNING]{RESET} {message}")

def print_error(message):
    """Print an error message in red."""
    print(f"{RED}[ERROR]{RESET} {message}")

def check_paths(DATA_DIRECTORY, OUT_DIR, SUBJECT, SESSION, CPU, FLAIR_FILE, T1W_FILE, DWI_FILE, BVAL_FILE, BVEC_FILE, INVERSE_DWI_FILE, THREADS):
    """
    Check if the paths are valid and if the files exist.
    """
    if SUBJECT == '':
        print_error("Subject not provided.")
        sys.exit(1)
    if SESSION == '':
        print_warning("Session not provided, if the dataset does not have sessions, please set the SESSION argument to 'None'.")
    if SESSION == 'None':
        print_note("Session is set to None.")
        SESSION = None
    if CPU == '':
        print_note("CPU not provided, defaulting to running on CPU. To enable GPU computation, set this to False.")
        CPU = True
    elif CPU == 'False':
        print_note("GPU computation enabled.")
        CPU = False
    if THREADS is None:
        print_note("THREADS not provided, defaulting to single-threading.")
        THREADS = 1
    elif THREADS == '':
        print_note("THREADS not provided, defaulting to single-threading.")
        THREADS = 1
    else:
        print_note(f"THREADS set to {THREADS}.")
        THREADS = int(THREADS)
    if DWI_FILE != '':
        print_note("DWI file provided. Enabling diffusion pipeline...")
        if BVAL_FILE == '':
            print_error("BVAL file not provided. BVAL file is required for diffusion processing.")
            sys.exit(1)
        if BVEC_FILE == '':
            print_error("BVEC file not provided. BVEC file is required for diffusion processing.")
            sys.exit(1)
        if INVERSE_DWI_FILE == '':
            print_error("Inverse DWI file not provided. Inverse DWI file is required for distortion correction.")
            sys.exit(1)
        RUN_DWI = True
    else:
        RUN_DWI = False
        print_note("DWI file not provided, disabiling diffusion pipeline.")
    if FLAIR_FILE == '':
        print_note("FLAIR file not provided, disabling FLAIR processing.")
        RUN_FLAIR = False
    else:
        print_note("FLAIR file provided. Enabling FLAIR processing...")
        RUN_FLAIR = True

    print_note("Checking paths...")
    if DATA_DIRECTORY != '':
        if not os.path.exists(DATA_DIRECTORY):
            print_error("The data directory does not exist.")
            sys.exit(1)
        else:
            print_note("Data directory exists.")
            print_note("Data directory: ", DATA_DIRECTORY)
            print_note("Checking if provided subject exists in BIDS directory...")
            if SUBJECT != '':
                if os.path.exists(DATA_DIRECTORY + '/' + SUBJECT):
                    print_note(f"Subject {SUBJECT} exists.")
                    if SESSION != '':
                        if SESSION == None:
                            print_note("Session is set to None.")
                        elif os.path.exists(DATA_DIRECTORY + '/' + SUBJECT + '/' + SESSION):
                            print_note(f"Session {SESSION} exists.")
                        else:
                            print_error(f"Session {SESSION} does not exist.")
                            sys.exit(1)
                else:
                    print_error(f"Subject {SUBJECT} does not exist.")
                    sys.exit(1)
            else:
                print_error("Subject not provided.")
                sys.exit(1)
            if RUN_FLAIR:
                flair_path = (DATA_DIRECTORY + '/' + SUBJECT + '/anat/' + FLAIR_FILE) if SESSION is None else (DATA_DIRECTORY + '/' + SUBJECT + '/' + SESSION + '/anat/' + FLAIR_FILE)
                if os.path.exists(flair_path):
                    print_note(f"FLAIR file exists at path {flair_path}.")
                    FLAIR_FILE = flair_path
                else:
                    print_error(f"FLAIR file does not exist at path {flair_path}.")
                    sys.exit(1)
            else:
                print_note("FLAIR file not provided.")

            if T1W_FILE != '':
                # Construct path based on SESSION value
                t1w_path = (DATA_DIRECTORY + '/' + SUBJECT + '/anat/' + T1W_FILE) if SESSION is None else (DATA_DIRECTORY + '/' + SUBJECT + '/' + SESSION + '/anat/' + T1W_FILE)
                if os.path.exists(t1w_path):
                    print_note(f"T1w file exists at path {t1w_path}.")
                    T1W_FILE = t1w_path
                else:
                    print_error(f"T1w file does not exist at path {t1w_path}.")
                    sys.exit(1)
            else:
                print_error("T1w file not provided.")
                print_error("T1w file is required for registration.")
                sys.exit(1)

            if RUN_DWI:
                print_note('Checking diffusion data...')
                
                if DWI_FILE != '':
                    # Construct path based on SESSION value
                    dwi_path = (DATA_DIRECTORY + '/' + SUBJECT + '/dwi/' + DWI_FILE) if SESSION is None else (DATA_DIRECTORY + '/' + SUBJECT + '/' + SESSION + '/dwi/' + DWI_FILE)
                    if os.path.exists(dwi_path):
                        print_note(f"DWI file exists at path {dwi_path}.")
                        DWI_FILE = dwi_path
                    else:
                        print_error(f"DWI file does not exist at path {dwi_path}.")
                        sys.exit(1)
                else:
                    print_error("DWI file not provided.")
                    sys.exit(1)
                    
                if BVAL_FILE != '':
                    # Construct path based on SESSION value
                    bval_path = (DATA_DIRECTORY + '/' + SUBJECT + '/dwi/' + BVAL_FILE) if SESSION is None else (DATA_DIRECTORY + '/' + SUBJECT + '/' + SESSION + '/dwi/' + BVAL_FILE)
                    if os.path.exists(bval_path):
                        print_note(f"BVAL file exists at path {bval_path}.")
                        BVAL_FILE = bval_path
                    else:
                        print_error(f"BVAL file does not exist at path {bval_path}.")
                        sys.exit(1)
                else:
                    print_error("BVAL file not provided.")
                    sys.exit(1)
                    
                if BVEC_FILE != '':
                    # Construct path based on SESSION value
                    bvec_path = (DATA_DIRECTORY + '/' + SUBJECT + '/dwi/' + BVEC_FILE) if SESSION is None else (DATA_DIRECTORY + '/' + SUBJECT + '/' + SESSION + '/dwi/' + BVEC_FILE)
                    if os.path.exists(bvec_path):
                        print_note(f"BVEC file exists at path {bvec_path}.")
                        BVEC_FILE = bvec_path
                    else:
                        print_error(f"BVEC file does not exist at path {bvec_path}.")
                        sys.exit(1)
                else:
                    print_error("BVEC file not provided.")
                    sys.exit(1)
                    
                if INVERSE_DWI_FILE != '':
                    # Construct path based on SESSION value
                    inverse_dwi_path = (DATA_DIRECTORY + '/' + SUBJECT + '/dwi/' + INVERSE_DWI_FILE) if SESSION is None else (DATA_DIRECTORY + '/' + SUBJECT + '/' + SESSION + '/dwi/' + INVERSE_DWI_FILE)
                    if os.path.exists(inverse_dwi_path):
                        print_note(f"Inverse DWI file exists at path {inverse_dwi_path}.")
                        INVERSE_DWI_FILE = inverse_dwi_path
                    else:
                        print_error(f"Inverse DWI file does not exist at path {inverse_dwi_path}.")
                        sys.exit(1)
                else:
                    print_error("Inverse DWI file not provided.")
                    sys.exit(1)
            else:
                print_note(f"Diffusion data not provided, diffusion pipeline is disabled.")
    else:
        print_note("Data directory not provided, file paths are assumed to be absolute paths to the relevant files.")
        if RUN_FLAIR:
            if os.path.exists(FLAIR_FILE):
                print_note(f"FLAIR file exists at path {FLAIR_FILE}.")
            else:
                print_error(f"FLAIR file does not exist at path {FLAIR_FILE}.")
                sys.exit(1)
        else:
            print_note("FLAIR will not be processed.")
        if T1W_FILE != '':
            if os.path.exists(T1W_FILE):
                print_note(f"T1w file exists at path {T1W_FILE}.")
            else:
                print_error(f"T1w file does not exist at path {T1W_FILE}.")
                sys.exit(1)
        else:
            print_error("T1w file not provided.")
            print_error("T1w file is required for registration.")
            sys.exit(1)
        if RUN_DWI:
            print_note('Checking diffusion data...')
            if DWI_FILE != '':
                if os.path.exists(DWI_FILE):
                    print_note(f"DWI file exists at path {DWI_FILE}.")
                else:
                    print_error(f"DWI file does not exist at path {DWI_FILE}.")
                    sys.exit(1)
            else:
                print_error("DWI file not provided.")
                sys.exit(1)
            if BVAL_FILE != '': 
                if os.path.exists(BVAL_FILE):
                    print_note(f"BVAL file exists at path {BVAL_FILE}.")
                else:
                    print_error(f"BVAL file does not exist at path {BVAL_FILE}.")
                    sys.exit(1)
            else:
                print_error("BVAL file not provided.")
                sys.exit(1)
            if BVEC_FILE != '':
                if os.path.exists(BVEC_FILE):
                    print_note(f"BVEC file exists at path {BVEC_FILE}.")
                else:
                    print_error(f"BVEC file does not exist at path {BVEC_FILE}.")
                    sys.exit(1)
            else:
                print_error("BVEC file not provided.")
                sys.exit(1)
            if INVERSE_DWI_FILE != '':
                if os.path.exists(INVERSE_DWI_FILE):
                    print_note(f"Inverse DWI file exists at path {INVERSE_DWI_FILE}.")
                else:
                    print_error(f"Inverse DWI file does not exist at path {INVERSE_DWI_FILE}.")
                    sys.exit(1)
            else:
                print_error("Inverse DWI file not provided.")
                sys.exit(1)
        else:
            print_note(f"Diffusion data not provided, diffusion pipeline is disabled.")
    print_note("All paths are valid.")
    print_note("Checking output directory...")
    if OUT_DIR != '':
        if not os.path.exists(OUT_DIR):
            print_note("Output directory does not exist, creating output directory...")
            os.makedirs(OUT_DIR)
            print_note(f"Output directory created at {OUT_DIR}.")
        else:
            print_note(f"Output directory exists at {OUT_DIR}.")
    else:
        print_note("Output directory not provided, output will be saved in the current working directory.")
    
    print_note("Checking hardware resources...")
    if CPU:
        print_note("CPU computation enabled.")
    else:

        if not torch.cuda.is_available():
            print_warning("GPU computation not enabled on PyTorch, running on CPU.")
            CPU = True
        else:
            print_note("GPU computation enabled from PyTorch.")

        if not tf.test.is_gpu_available():
            print_warning("GPU computation not enabled on Tensorflow, running on CPU.")
            CPU = True
        else: 
            print_note("GPU computation enabled from Tensorflow.")
    
    print_note("Checking number of threads...")
    print_note(f"Number of threads: {THREADS}")
    
    return DATA_DIRECTORY, OUT_DIR, SUBJECT, SESSION, RUN_DWI, CPU, FLAIR_FILE, T1W_FILE, DWI_FILE, BVAL_FILE, BVEC_FILE, INVERSE_DWI_FILE, THREADS, RUN_FLAIR
    