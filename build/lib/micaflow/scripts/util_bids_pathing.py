
import os
import sys
import glob
import shutil
import subprocess
import torch
import tensorflow as tf
def check_paths(DATA_DIRECTORY, OUT_DIR, SUBJECT, SESSION, RUN_DWI, CPU, FLAIR_FILE, T1W_FILE, DWI_FILE, BVAL_FILE, BVEC_FILE, INVERSE_DWI_FILE, THREADS):
    """
    Check if the paths are valid and if the files exist.
    """
    if SUBJECT == '':
        print("Subject not provided.")
        sys.exit(1)
    if SESSION == '':
        print("Session not provided, if the dataset does not have sessions, please set the SESSION argument to 'None'.")
    if SESSION == 'None':
        print("Session is set to None.")
        SESSION = None
    if RUN_DWI == '':
        print("RUN_DWI not provided, defaulting to True.")
        RUN_DWI = True
    if CPU == '':
        print("CPU not provided, defaulting to running on CPU. To enable GPU computation, set this to False.")
        CPU = True
    if THREADS == '':
        print("THREADS not provided, defaulting to single-threading.")
        THREADS = 1
    
    print("Checking paths...")
    if DATA_DIRECTORY != '':
        if not os.path.exists(DATA_DIRECTORY):
            print("The data directory does not exist.")
            sys.exit(1)
        else:
            print("Data directory exists.")
            print("Data directory: ", DATA_DIRECTORY)
            print("Checking if provided subject exists in BIDS directory...")
            if SUBJECT != '':
                if os.path.exists(DATA_DIRECTORY + '/' + SUBJECT):
                    print(f"Subject {SUBJECT} exists.")
                    if SESSION != '':
                        if os.path.exists(DATA_DIRECTORY + '/' + SUBJECT + '/' + SESSION):
                            print(f"Session {SESSION} exists.")
                        elif SESSION == None:
                            print("Session is set to None.")
                        else:
                            print(f"Session {SESSION} does not exist.")
                            sys.exit(1)
                else:
                    print(f"Subject {SUBJECT} does not exist.")
                    sys.exit(1)
            else:
                print("Subject not provided.")
                sys.exit(1)
            if FLAIR_FILE != '':
                # Construct path based on SESSION value
                flair_path = (DATA_DIRECTORY + '/' + SUBJECT + '/anat/' + FLAIR_FILE) if SESSION is None else (DATA_DIRECTORY + '/' + SUBJECT + '/' + SESSION + '/anat/' + FLAIR_FILE)
                if os.path.exists(flair_path):
                    print(f"FLAIR file exists at path {flair_path}.")
                    FLAIR_FILE = flair_path
                else:
                    print(f"FLAIR file does not exist at path {flair_path}.")
                    sys.exit(1)
            else:
                print("FLAIR file not provided.")

            if T1W_FILE != '':
                # Construct path based on SESSION value
                t1w_path = (DATA_DIRECTORY + '/' + SUBJECT + '/anat/' + T1W_FILE) if SESSION is None else (DATA_DIRECTORY + '/' + SUBJECT + '/' + SESSION + '/anat/' + T1W_FILE)
                if os.path.exists(t1w_path):
                    print(f"T1w file exists at path {t1w_path}.")
                    T1W_FILE = t1w_path
                else:
                    print(f"T1w file does not exist at path {t1w_path}.")
                    sys.exit(1)
            else:
                print("T1w file not provided.")
                print("T1w file is required for registration.")
                sys.exit(1)

            if RUN_DWI:
                print('RUN_DWI is set to true.')
                print('Checking diffusion data...')
                
                if DWI_FILE != '':
                    # Construct path based on SESSION value
                    dwi_path = (DATA_DIRECTORY + '/' + SUBJECT + '/dwi/' + DWI_FILE) if SESSION is None else (DATA_DIRECTORY + '/' + SUBJECT + '/' + SESSION + '/dwi/' + DWI_FILE)
                    if os.path.exists(dwi_path):
                        print(f"DWI file exists at path {dwi_path}.")
                        DWI_FILE = dwi_path
                    else:
                        print(f"DWI file does not exist at path {dwi_path}.")
                        sys.exit(1)
                else:
                    print("DWI file not provided.")
                    sys.exit(1)
                    
                if BVAL_FILE != '':
                    # Construct path based on SESSION value
                    bval_path = (DATA_DIRECTORY + '/' + SUBJECT + '/dwi/' + BVAL_FILE) if SESSION is None else (DATA_DIRECTORY + '/' + SUBJECT + '/' + SESSION + '/dwi/' + BVAL_FILE)
                    if os.path.exists(bval_path):
                        print(f"BVAL file exists at path {bval_path}.")
                        BVAL_FILE = bval_path
                    else:
                        print(f"BVAL file does not exist at path {bval_path}.")
                        sys.exit(1)
                else:
                    print("BVAL file not provided.")
                    sys.exit(1)
                    
                if BVEC_FILE != '':
                    # Construct path based on SESSION value
                    bvec_path = (DATA_DIRECTORY + '/' + SUBJECT + '/dwi/' + BVEC_FILE) if SESSION is None else (DATA_DIRECTORY + '/' + SUBJECT + '/' + SESSION + '/dwi/' + BVEC_FILE)
                    if os.path.exists(bvec_path):
                        print(f"BVEC file exists at path {bvec_path}.")
                        BVEC_FILE = bvec_path
                    else:
                        print(f"BVEC file does not exist at path {bvec_path}.")
                        sys.exit(1)
                else:
                    print("BVEC file not provided.")
                    sys.exit(1)
                    
                if INVERSE_DWI_FILE != '':
                    # Construct path based on SESSION value
                    inverse_dwi_path = (DATA_DIRECTORY + '/' + SUBJECT + '/dwi/' + INVERSE_DWI_FILE) if SESSION is None else (DATA_DIRECTORY + '/' + SUBJECT + '/' + SESSION + '/dwi/' + INVERSE_DWI_FILE)
                    if os.path.exists(inverse_dwi_path):
                        print(f"Inverse DWI file exists at path {inverse_dwi_path}.")
                        INVERSE_DWI_FILE = inverse_dwi_path
                    else:
                        print(f"Inverse DWI file does not exist at path {inverse_dwi_path}.")
                        sys.exit(1)
                else:
                    print("Inverse DWI file not provided.")
                    sys.exit(1)
            else:
                print(f"RUN_DWI is set to false, diffusion data will not be processed.")
    else:
        print("Data directory not provided, file paths are assumed to be absolute paths to the relevant files.")
        
        if FLAIR_FILE != '':
            if os.path.exists(FLAIR_FILE):
                print(f"FLAIR file exists at path {FLAIR_FILE}.")
            else:
                print(f"FLAIR file does not exist at path {FLAIR_FILE}.")
                sys.exit(1)
        else:
            print("FLAIR file not provided.")
        if T1W_FILE != '':
            if os.path.exists(T1W_FILE):
                print(f"T1w file exists at path {T1W_FILE}.")
            else:
                print(f"T1w file does not exist at path {T1W_FILE}.")
                sys.exit(1)
        else:
            print("T1w file not provided.")
            print("T1w file is required for registration.")
            sys.exit(1)
        if RUN_DWI:
            print('RUN_DWI is set to true.')
            print('Checking diffusion data...')
            if DWI_FILE != '':
                if os.path.exists(DWI_FILE):
                    print(f"DWI file exists at path {DWI_FILE}.")
                else:
                    print(f"DWI file does not exist at path {DWI_FILE}.")
                    sys.exit(1)
            else:
                print("DWI file not provided.")
                sys.exit(1)
            if BVAL_FILE != '': 
                if os.path.exists(BVAL_FILE):
                    print(f"BVAL file exists at path {BVAL_FILE}.")
                else:
                    print(f"BVAL file does not exist at path {BVAL_FILE}.")
                    sys.exit(1)
            else:
                print("BVAL file not provided.")
                sys.exit(1)
            if BVEC_FILE != '':
                if os.path.exists(BVEC_FILE):
                    print(f"BVEC file exists at path {BVEC_FILE}.")
                else:
                    print(f"BVEC file does not exist at path {BVEC_FILE}.")
                    sys.exit(1)
            else:
                print("BVEC file not provided.")
                sys.exit(1)
            if INVERSE_DWI_FILE != '':
                if os.path.exists(INVERSE_DWI_FILE):
                    print(f"Inverse DWI file exists at path {INVERSE_DWI_FILE}.")
                else:
                    print(f"Inverse DWI file does not exist at path {INVERSE_DWI_FILE}.")
                    sys.exit(1)
            else:
                print("Inverse DWI file not provided.")
                sys.exit(1)
        else:
            print(f"RUN_DWI is set to false, diffusion data will not be processed.")
    print("All paths are valid.")
    print("Checking output directory...")
    if OUT_DIR != '':
        if not os.path.exists(OUT_DIR):
            print("Output directory does not exist, creating output directory...")
            os.makedirs(OUT_DIR)
            print(f"Output directory created at {OUT_DIR}.")
        else:
            print(f"Output directory exists at {OUT_DIR}.")
    else:
        print("Output directory not provided, output will be saved in the current working directory.")
    print("Checking hardware resources...")
    if CPU:
        print("CPU computation enabled.")
    else:

        if not torch.cuda.is_available():
            print("GPU computation not enabled on PyTorch, running on CPU.")
            CPU = True
        else:
            print("GPU computation enabled from PyTorch.")

        if not tf.test.is_gpu_available():
            print("GPU computation not enabled on Tensorflow, running on CPU.")
            CPU = True
        else: 
            print("GPU computation enabled from Tensorflow.")
    print("Checking number of threads...")
    print(f"Number of threads: {THREADS}")
    return DATA_DIRECTORY, OUT_DIR, SUBJECT, SESSION, RUN_DWI, CPU, FLAIR_FILE, T1W_FILE, DWI_FILE, BVAL_FILE, BVEC_FILE, INVERSE_DWI_FILE, THREADS
    