"""
texture_generation - MRI Texture Feature Extraction for Radiomics

Part of the micaflow processing pipeline for neuroimaging data.

This module computes advanced texture features from MRI data that can be used for
tissue characterization, lesion analysis, radiomics applications, and quantitative
imaging biomarker development. It performs automatic tissue segmentation and extracts
quantitative imaging features including gradient magnitude and relative intensity maps,
which capture local intensity variations and tissue contrast properties respectively.

What are Texture Features?
--------------------------
Texture features quantify spatial patterns and intensity variations in medical images:
- Provide information beyond simple intensity statistics
- Capture tissue heterogeneity and structural organization
- Enable quantitative characterization of pathology
- Support radiomics and machine learning applications
- Complement traditional volumetric measurements

Computed Features:
-----------------
1. Gradient Magnitude:
   - Measures local intensity changes
   - Highlights edges and boundaries
   - Captures tissue transitions
   - Sensitive to structural organization
   - Range: 0 to maximum gradient

2. Relative Intensity (RI):
   - Normalized intensity relative to tissue peaks
   - Accounts for scanner/protocol variations
   - Centers around GM-WM boundary
   - Range: Typically 0-200 (100 = background)
   - Provides contrast-normalized values

How It Works:
------------
1. Load input MRI and brain mask
2. Segment brain into GM, WM, CSF (Atropos K-means)
3. Find GM and WM intensity peaks
4. Compute gradient magnitude (edge detection)
5. Calculate relative intensity (normalized contrast)
6. Apply smoothing to relative intensity map
7. Save all feature maps as NIfTI files

Relative Intensity Calculation:
------------------------------
1. Find GM peak intensity (mode of GM histogram)
2. Find WM peak intensity (mode of WM histogram)
3. Compute background: BG = 0.5 × (GM_peak + WM_peak)
4. For voxels < BG: RI = 100 × (1 - (BG - I) / BG)
5. For voxels > BG: RI = 100 × (1 + (I - BG) / BG)
6. Smooth with Gaussian (σ=3mm FWHM)

Command-Line Usage:
------------------
# Basic usage
micaflow texture_generation \\
    --input <path/to/image.nii.gz> \\
    --mask <path/to/brain_mask.nii.gz> \\
    --output <path/to/output_prefix>

# Example with T1w image
micaflow texture_generation \\
    --input T1w_preprocessed.nii.gz \\
    --mask brain_mask.nii.gz \\
    --output subject01_textures

# Example with short flags
micaflow texture_generation \\
    -i T1w.nii.gz \\
    -m mask.nii.gz \\
    -o output/features

Python API Usage:
----------------
>>> from micaflow.scripts.texture_generation import run_texture_pipeline
>>> 
>>> # Basic usage
>>> run_texture_pipeline(
...     input="preprocessed_t1w.nii.gz",
...     mask="brain_mask.nii.gz",
...     output_dir="output_texture_maps"
... )
>>> 
>>> # With variables
>>> input_file = "data/T1w.nii.gz"
>>> mask_file = "data/mask.nii.gz"
>>> output_prefix = "results/subject01"
>>> run_texture_pipeline(input_file, mask_file, output_prefix)

Pipeline Integration:
--------------------
Texture generation typically follows preprocessing:

Structural MRI Pipeline:
1. Preprocessing (N4 bias correction, denoising)
2. Brain extraction (skull stripping)
3. Tissue segmentation (optional: use synthseg instead)
4. Texture feature extractiom
5. Statistical analysis or machine learning

Radiomics Pipeline:
1. Image acquisition and quality control
2. Preprocessing and standardization
3. ROI/lesion segmentation
4. Feature extraction (texture_generation)
5. Feature selection
6. Model training/prediction

Exit Codes:
----------
0 : Success - texture features computed successfully
1 : Error - invalid inputs, file not found, or processing failure

See Also:
--------
- synthseg : Alternative segmentation method
- n4_bias_correction : Recommended preprocessing
- denoise : Noise reduction before feature extraction

"""

import argparse
import os
import random
import string
from collections import Counter
import ants
import numpy as np
import time
import sys
from colorama import init, Fore, Style

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
    """
    Print comprehensive help message with examples and usage instructions.
    
    This function displays detailed information about texture feature extraction including:
    - What texture features are and why they matter
    - Available features (gradient magnitude, relative intensity)
    - Command-line options and usage
    - Multiple examples for different scenarios
    - Technical details about algorithms
    - Quality control recommendations
    - Output file descriptions
    
    The help message uses color-coded sections for better readability.
    
    Examples
    --------
    >>> # Display help message
    >>> print_help_message()
    
    >>> # Help is shown automatically with --help, -h, or no arguments
    >>> # micaflow texture_generation --help
    
    Notes
    -----
    - Called automatically when script run with --help, -h, or no arguments
    - Provides more detail than standard argparse help
    - Uses ANSI color codes for visual organization
    """
    
    help_text = f"""
    {CYAN}{BOLD}╔════════════════════════════════════════════════════════════════╗
    ║                  TEXTURE FEATURE EXTRACTION                    ║
    ║                        (Radiomics)                             ║
    ╚════════════════════════════════════════════════════════════════╝{RESET}
    
    This script generates quantitative texture feature maps from MRI data
    for radiomics, tissue characterization, and quantitative imaging biomarker
    development. Features include gradient magnitude and relative intensity.
    
    {CYAN}{BOLD}────────────────────────── USAGE ──────────────────────────{RESET}
      micaflow texture_generation {GREEN}[options]{RESET}
    
    {CYAN}{BOLD}─────────────────── REQUIRED ARGUMENTS ───────────────────{RESET}
      {YELLOW}--input{RESET}, {YELLOW}-i{RESET}   : Path to the input MRI image (.nii.gz)
      {YELLOW}--mask{RESET}, {YELLOW}-m{RESET}    : Path to the binary brain mask (.nii.gz)
      {YELLOW}--output{RESET}, {YELLOW}-o{RESET}  : Output prefix for texture feature maps
    
    {CYAN}{BOLD}──────────────────── EXAMPLE USAGE ──────────────────────{RESET}
    
    {BLUE}# Example 1: Basic usage{RESET}
    micaflow texture_generation \\
      {YELLOW}--input{RESET} T1w_preprocessed.nii.gz \\
      {YELLOW}--mask{RESET} brain_mask.nii.gz \\
      {YELLOW}--output{RESET} subject01_textures
    
    {BLUE}# Example 2: With short flags{RESET}
    micaflow texture_generation \\
      {YELLOW}-i{RESET} T1w.nii.gz \\
      {YELLOW}-m{RESET} mask.nii.gz \\
      {YELLOW}-o{RESET} output/features
    
    {BLUE}# Example 3: From preprocessed data{RESET}
    micaflow texture_generation \\
      {YELLOW}-i{RESET} data/preprocessed/T1w_n4.nii.gz \\
      {YELLOW}-m{RESET} data/masks/brain_mask.nii.gz \\
      {YELLOW}-o{RESET} results/radiomics/sub-01
    
    {CYAN}{BOLD}────────── WHAT ARE TEXTURE FEATURES? ──────────────────{RESET}
    
    {GREEN}Texture features quantify spatial patterns in medical images:{RESET}
    {MAGENTA}•{RESET} Go beyond simple intensity statistics
    {MAGENTA}•{RESET} Capture tissue heterogeneity and organization
    {MAGENTA}•{RESET} Provide quantitative imaging biomarkers
    {MAGENTA}•{RESET} Enable radiomics and machine learning
    {MAGENTA}•{RESET} Support precision medicine approaches
    
    {GREEN}Common applications:{RESET}
    {MAGENTA}•{RESET} Tumor characterization and grading
    {MAGENTA}•{RESET} Neurodegenerative disease staging
    {MAGENTA}•{RESET} White matter lesion analysis
    {MAGENTA}•{RESET} Age-related changes
    {MAGENTA}•{RESET} Treatment response monitoring
    
    {CYAN}{BOLD}─────────────── COMPUTED FEATURES ──────────────────────{RESET}
    
    {GREEN}1. Gradient Magnitude:{RESET}
    {MAGENTA}•{RESET} Measures local intensity changes
    {MAGENTA}•{RESET} Highlights edges and tissue boundaries
    {MAGENTA}•{RESET} Sensitive to structural organization
    {MAGENTA}•{RESET} Range: 0 to maximum gradient
    {MAGENTA}•{RESET} Higher values at GM/WM boundaries
    
    {GREEN}2. Relative Intensity (RI):{RESET}
    {MAGENTA}•{RESET} Normalized intensity relative to tissue peaks
    {MAGENTA}•{RESET} Accounts for scanner/protocol variations
    {MAGENTA}•{RESET} Centers around GM-WM boundary (value = 100)
    {MAGENTA}•{RESET} Range: Typically 50-150
    {MAGENTA}•{RESET} Provides contrast-normalized values
    
    {GREEN}3. Tissue Segmentation (intermediate):{RESET}
    {MAGENTA}•{RESET} Gray matter mask
    {MAGENTA}•{RESET} White matter mask
    {MAGENTA}•{RESET} CSF identification
    {MAGENTA}•{RESET} Used for feature computation
    
    {CYAN}{BOLD}────────────────── HOW IT WORKS ────────────────────────{RESET}
    
    {GREEN}Processing pipeline:{RESET}
    {MAGENTA}1.{RESET} Load input MRI and brain mask
    {MAGENTA}2.{RESET} Segment brain into GM, WM, CSF (Atropos K-means)
    {MAGENTA}3.{RESET} Find GM and WM intensity peaks
    {MAGENTA}4.{RESET} Compute gradient magnitude (edge detection)
    {MAGENTA}5.{RESET} Calculate relative intensity (normalized contrast)
    {MAGENTA}6.{RESET} Apply smoothing (σ=3mm FWHM)
    {MAGENTA}7.{RESET} Save feature maps as NIfTI files
    
    {GREEN}Atropos segmentation:{RESET}
    {MAGENTA}•{RESET} K-means clustering (3 tissue classes)
    {MAGENTA}•{RESET} MRF spatial prior (0.2 weight, 1x1x1 neighborhood)
    {MAGENTA}•{RESET} 3 iterations for convergence
    {MAGENTA}•{RESET} Masked to brain tissue only
    {MAGENTA}•{RESET} Output: 1=CSF, 2=GM, 3=WM
    
    {CYAN}{BOLD}────────────────────── OUTPUT FILES ─────────────────────{RESET}
    
    Given output prefix {GREEN}"subject01_textures"{RESET}, creates:
    
    {YELLOW}subject01_textures_gradient-magnitude.nii.gz{RESET}
      - Edge and boundary detection map
      - Higher values at tissue transitions
      - Range: 0 to ~50 (depends on contrast)
    
    {YELLOW}subject01_textures_relative-intensity.nii.gz{RESET}
      - Normalized contrast map
      - Centered at GM-WM boundary (100)
      - Range: Typically 50-150
      - Smoothed with σ=3mm FWHM
    
    {CYAN}{BOLD}────────────────────────── NOTES ─────────────────────────{RESET}
    {MAGENTA}•{RESET} Preprocessing recommended: N4 bias correction, denoising
    {MAGENTA}•{RESET} Processing time: 2-5 minutes per subject
    {MAGENTA}•{RESET} Memory: ~2-4 GB RAM
    {MAGENTA}•{RESET} All operations masked to brain only
    {MAGENTA}•{RESET} No resampling (preserves input resolution)
    {MAGENTA}•{RESET} Values outside mask are zero
    {MAGENTA}•{RESET} Gradient uses first-order finite differences
    {MAGENTA}•{RESET} Peak finding uses 1st-99.5th percentile range
    
    {CYAN}{BOLD}─────────────── PIPELINE POSITION ───────────────────────{RESET}
    {BLUE}Structural MRI Pipeline:{RESET}
    1. Preprocessing (N4, denoising)
    2. Brain extraction (skull stripping)
    3. Tissue segmentation (optional)
    {GREEN}4. Texture feature extraction{RESET} {MAGENTA}← You are here{RESET}
    5. Statistical analysis or ML
    
    {BLUE}Radiomics Pipeline:{RESET}
    1. Image acquisition and QC
    2. Preprocessing and standardization
    3. ROI/lesion segmentation
    {GREEN}4. Feature extraction{RESET} {MAGENTA}← You are here{RESET}
    5. Feature selection
    6. Model training/prediction
    
    {CYAN}{BOLD}────────────────────── EXIT CODES ───────────────────────{RESET}
    {GREEN}0{RESET} : Success - texture features computed
    {RED}1{RESET} : Error - invalid inputs, file not found, or processing failure
    
    {CYAN}{BOLD}───────────── QUALITY CONTROL ───────────────────────────{RESET}
    {YELLOW}Visual inspection:{RESET}
    {MAGENTA}1.{RESET} Check gradient magnitude highlights tissue boundaries
    {MAGENTA}2.{RESET} Verify relative intensity centered around 100
    {MAGENTA}3.{RESET} Ensure smooth transitions (no edge artifacts)
    {MAGENTA}4.{RESET} Confirm brain mask coverage is complete
    {MAGENTA}5.{RESET} Look for segmentation errors
    
    {YELLOW}Expected ranges:{RESET}
    {MAGENTA}•{RESET} Gradient magnitude: 0 to ~50
    {MAGENTA}•{RESET} Relative intensity: 50-150 (100 = background)
    {MAGENTA}•{RESET} Values outside mask: 0
    
    {CYAN}{BOLD}───────────────── COMMON ISSUES ─────────────────────────{RESET}
    {YELLOW}Issue:{RESET} Poor segmentation quality
    {GREEN}Solution:{RESET} Use N4 bias correction first, verify mask quality
    
    {YELLOW}Issue:{RESET} Extreme gradient values
    {GREEN}Solution:{RESET} Check image quality, reduce noise with denoising
    
    {YELLOW}Issue:{RESET} Relative intensity not centered at 100
    {GREEN}Solution:{RESET} Verify tissue segmentation, check for artifacts
    
    {YELLOW}Issue:{RESET} Processing takes very long
    {GREEN}Solution:{RESET} Normal for high-resolution images (2-5 min typical)
    """
    print(help_text)


def write_nifti(input, id, output_dir, type):
    """
    Write ANTsImage to NIfTI file with standardized naming.
    
    Parameters
    ----------
    input : ANTsImage
        Image data to write to file.
    id : str
        Subject or case identifier for filename.
    output_dir : str
        Output directory path.
    type : str
        Feature type suffix (e.g., 'gradient-magnitude', 'relative-intensity').
    
    Returns
    -------
    None
        Writes file to disk: <output_dir>/<id>_<type>.nii.gz
    
    Examples
    --------
    >>> gradient_img = ants.image_read("gradient.nii.gz")
    >>> write_nifti(gradient_img, "sub-01", "/output", "gradient-magnitude")
    # Creates: /output/sub-01_gradient-magnitude.nii.gz
    """
    output_fname = os.path.join(output_dir, id + '_' + type + '.nii.gz')
    ants.image_write(input, output_fname)


def compute_RI(image, bg, mask):
    """
    Compute relative intensity map normalized to background intensity.
    
    Relative intensity (RI) normalizes voxel intensities relative to a background
    reference (typically the GM-WM boundary), accounting for scanner and protocol
    variations. This creates contrast-normalized values centered at 100.
    
    Formula:
    - For voxels < bg: RI = 100 × (1 - (bg - I) / bg)
    - For voxels > bg: RI = 100 × (1 + (I - bg) / bg)
    
    Parameters
    ----------
    image : ndarray
        Input intensity image as numpy array.
    bg : float
        Background intensity reference (typically 0.5 × (GM_peak + WM_peak)).
    mask : ndarray
        Binary mask defining brain tissue (1=brain, 0=background).
    
    Returns
    -------
    ndarray
        Relative intensity map with same shape as input.
        Values: ~50-150 (100 = background reference)
        Outside mask: 0
    
    Notes
    -----
    - Based on Nyúl et al. (2000) intensity normalization
    - Centers values at 100 for interpretability
    - Preserves relative contrast relationships
    - Accounts for inter-scanner variability
    
    Examples
    --------
    >>> img = np.array([50, 75, 100, 125, 150])
    >>> bg = 100
    >>> mask = np.ones(5)
    >>> ri = compute_RI(img, bg, mask)
    >>> print(ri)
    [50. 75. 100. 125. 150.]
    
    References
    ----------
    Nyúl LG, Udupa JK, Zhang X. New variants of a method of MRI scale 
    standardization. IEEE Trans Med Imaging. 2000;19(2):143-150.
    """
    ri = np.zeros_like(image)
    
    # Find voxels below background (typically CSF and dark GM)
    bgm = np.stack(np.where(np.logical_and(image < bg, mask == 1)), axis=1)
    bgm_ind = bgm[:, 0], bgm[:, 1], bgm[:, 2]
    
    # Find voxels above background (typically WM and bright GM)
    bgp = np.stack(np.where(np.logical_and(image > bg, mask == 1)), axis=1)
    bgp_ind = bgp[:, 0], bgp[:, 1], bgp[:, 2]

    # Compute relative intensity (centered at 100)
    ri[bgm_ind] = 100 * (1 - (bg - image[bgm_ind]) / bg)
    ri[bgp_ind] = 100 * (1 + (bg - image[bgp_ind]) / bg)
    
    return ri


def peakfinder(gm, wm, lower_q, upper_q):
    """
    Find intensity peaks for gray matter and white matter, compute background.
    
    Uses histogram mode (most common intensity) to find representative peaks
    for GM and WM tissues. The background reference is computed as the midpoint
    between these peaks, representing the GM-WM boundary.
    
    Parameters
    ----------
    gm : ANTsImage
        Gray matter masked image.
    wm : ANTsImage
        White matter masked image.
    lower_q : float
        Lower percentile for thresholding (e.g., 1 = 1st percentile).
    upper_q : float
        Upper percentile for thresholding (e.g., 99.5 = 99.5th percentile).
    
    Returns
    -------
    float
        Background intensity: 0.5 × (GM_peak + WM_peak)
        Represents the GM-WM boundary intensity.
    
    Notes
    -----
    - Uses mode (most common value) rather than mean
    - More robust to outliers and partial volume effects
    - Percentile thresholding removes extreme values
    - Intensities are rounded before finding mode
    
    Examples
    --------
    >>> gm_img = ants.image_read("gm_masked.nii.gz")
    >>> wm_img = ants.image_read("wm_masked.nii.gz")
    >>> bg = peakfinder(gm_img, wm_img, 1, 99.5)
    >>> print(f"Background: {bg}")
    Background: 542.5
    """
    # Find mode of GM intensities (within percentile range)
    gm_peak = Counter(threshold_percentile(gm, lower_q, upper_q)).most_common(1)[0][0]
    
    # Find mode of WM intensities (within percentile range)
    wm_peak = Counter(threshold_percentile(wm, lower_q, upper_q)).most_common(1)[0][0]
    
    # Compute background as midpoint (GM-WM boundary)
    bg = 0.5 * (gm_peak + wm_peak)
    
    return bg


def threshold_percentile(x, lower_q, upper_q):
    """
    Threshold image intensities to percentile range and round values.
    
    Removes extreme values using percentile thresholding, which improves
    robustness of peak finding by excluding outliers, artifacts, and noise.
    
    Parameters
    ----------
    x : ANTsImage
        Input image to threshold.
    lower_q : float
        Lower percentile threshold (0-100).
    upper_q : float
        Upper percentile threshold (0-100).
    
    Returns
    -------
    ndarray
        Flattened array of rounded intensities within percentile range.
    
    Notes
    -----
    - Values are rounded to integers before return
    - Flattened to 1D array for histogram computation
    - Removes extreme outliers that could bias peak finding
    - Typical range: 1st to 99.5th percentile
    
    Examples
    --------
    >>> img = ants.image_read("brain.nii.gz")
    >>> vals = threshold_percentile(img, 1, 99.5)
    >>> print(f"Shape: {vals.shape}, Range: {vals.min()}-{vals.max()}")
    Shape: (245632,), Range: 15-1250
    """
    # Convert to numpy array
    x = x.numpy()
    
    # Compute percentile thresholds
    lq = np.percentile(x, lower_q)
    uq = np.percentile(x, upper_q)
    
    # Keep only values within range
    x = x[np.logical_and(x > lq, x <= uq)]
    
    # Round and flatten for histogram
    return x.flatten().round()


def find_logger_basefilename(logger):
    """
    Find the base filename of a logger's file handler.
    
    Note: This function appears to be unused legacy code.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger object with file handler.
    
    Returns
    -------
    str or None
        Base filename of the first file handler, or None if no handler.
    """
    log_file = None
    handler = logger.handlers[0]
    log_file = handler.baseFilename
    return log_file


def random_case_id():
    """
    Generate a random case identifier string.
    
    Creates a unique identifier in format: abc_1234
    (3 lowercase letters + underscore + 4 digits)
    
    Note: This function appears to be unused in the current implementation.
    
    Returns
    -------
    str
        Random identifier string (e.g., "xyz_5678").
    
    Examples
    --------
    >>> case_id = random_case_id()
    >>> print(case_id)
    'abc_1234'
    >>> print(len(case_id))
    8
    """
    letters = ''.join(random.choices(string.ascii_letters, k=16))
    digits = ''.join(random.choices(string.digits, k=16))
    x = letters[:3].lower() + '_' + digits[:4]
    return x


class noelTexturesPy:
    """
    Texture feature extraction pipeline for neuroimaging data.
    
    This class implements a complete pipeline for computing texture features
    from MRI scans, including tissue segmentation, gradient magnitude, and
    relative intensity calculations.
    
    Parameters
    ----------
    id : str
        Subject or case identifier for output files.
    output_dir : str, optional
        Directory prefix for output files.
    input : str, optional
        Path to input MRI image (.nii.gz).
    mask : str, optional
        Path to brain mask image (.nii.gz).
    
    Attributes
    ----------
    _id : str
        Subject identifier.
    _outputdir : str
        Output directory prefix.
    input : str
        Input image path.
    mask : str
        Mask image path.
    _input : ANTsImage
        Loaded input image.
    _mask : ANTsImage
        Loaded mask image.
    _segm : ANTsImage
        Tissue segmentation (1=CSF, 2=GM, 3=WM).
    _gm : ndarray
        Gray matter binary mask.
    _wm : ndarray
        White matter binary mask.
    _grad_input : ANTsImage
        Gradient magnitude map.
    _ri : ANTsImage
        Relative intensity map.
    
    Methods
    -------
    load_nifti_file()
        Load input image and mask from disk.
    segmentation()
        Segment brain into GM, WM, CSF using Atropos.
    gradient_magnitude()
        Compute and save gradient magnitude feature.
    relative_intensity()
        Compute and save relative intensity feature.
    file_processor()
        Execute complete processing pipeline.
    
    Examples
    --------
    >>> pipeline = noelTexturesPy(
    ...     id='sub-01',
    ...     output_dir='results/textures',
    ...     input='T1w.nii.gz',
    ...     mask='mask.nii.gz'
    ... )
    >>> pipeline.file_processor()
    loading nifti files
    computing GM, WM, CSF segmentation
    computing gradient magnitude
    computing relative intensity
    pipeline processing time elapsed: 183.4 seconds
    
    Notes
    -----
    - Uses Atropos K-means for tissue segmentation
    - Gradient computed with finite differences
    - Relative intensity smoothed with σ=3mm FWHM
    - All outputs are NIfTI format (.nii or .nii.gz)
    """
    
    def __init__(
        self,
        id,
        output_dir=None,
        input=None,
        mask=None,
    ):
        """Initialize texture extraction pipeline."""
        super().__init__()
        self._id = id
        self._outputdir = output_dir
        self.input = input
        self.mask = mask

    def load_nifti_file(self):
        """
        Load input MRI image and brain mask from disk.
        
        Reads NIfTI files into memory as ANTsImage objects for processing.
        
        Raises
        ------
        FileNotFoundError
            If input or mask file does not exist.
        RuntimeError
            If file loading fails.
        
        Notes
        -----
        - Prints status message during loading
        - Sets self._input and self._mask attributes
        - Files must be in NIfTI format (.nii or .nii.gz)
        """
        # load nifti data to memory
        print(f'{CYAN}Loading NIfTI files...{RESET}')
        self._input = ants.image_read(self.input)
        self._mask = ants.image_read(self.mask)
        print(f'  {GREEN}Input: {self.input}{RESET}')
        print(f'  {GREEN}Mask: {self.mask}{RESET}')

    def segmentation(self):
        """
        Segment brain tissue into gray matter, white matter, and CSF.
        
        Uses Atropos K-means clustering with MRF spatial prior to segment
        the brain into three tissue classes. Creates binary masks for GM
        and WM used in subsequent feature computations.
        
        Algorithm:
        - K-means clustering with 3 classes
        - MRF smoothing (weight=0.2, neighborhood=1x1x1)
        - 3 iterations for convergence
        - Masked to brain tissue only
        
        Sets:
        ----
        self._segm : ANTsImage
            Segmentation labels (1=CSF, 2=GM, 3=WM)
        self._gm : ndarray
            Binary gray matter mask (1=GM, 0=other)
        self._wm : ndarray
            Binary white matter mask (1=WM, 0=other)
        
        Notes
        -----
        - CSF typically has label 1 (darkest)
        - GM typically has label 2 (intermediate)
        - WM typically has label 3 (brightest)
        - Assumes T1w-like contrast
        
        References
        ----------
        Avants BB, Tustison NJ, Wu J, et al. An open source multivariate 
        framework for n-tissue segmentation with evaluation on public data. 
        Neuroinformatics. 2011;9(4):381-400.
        """
        print(f'{CYAN}Computing GM, WM, CSF segmentation...{RESET}')
        
        # Run Atropos segmentation
        segm = ants.atropos(
            a=self._input,              # Input image
            i='Kmeans[3]',              # K-means with 3 classes
            m='[0.2,1x1x1]',            # MRF: weight=0.2, neighborhood=1x1x1
            c='[3,0]',                  # 3 iterations, no initialization
            x=self._mask,               # Brain mask
        )
        
        self._segm = segm['segmentation']
        
        # Create binary tissue masks
        self._gm = np.where((self._segm.numpy() == 2), 1, 0).astype('float32')
        self._wm = np.where((self._segm.numpy() == 3), 1, 0).astype('float32')
        
        print(f'  {GREEN}Segmentation completed{RESET}')
        print(f'    CSF: label 1')
        print(f'    GM:  label 2 ({np.sum(self._gm)} voxels)')
        print(f'    WM:  label 3 ({np.sum(self._wm)} voxels)')

    def gradient_magnitude(self):
        """
        Compute gradient magnitude for edge and boundary detection.
        
        Calculates the magnitude of the intensity gradient at each voxel,
        which highlights edges and tissue boundaries. Higher values indicate
        sharp intensity transitions (e.g., GM-WM boundary).
        
        Method:
        - First-order finite differences
        - Magnitude: sqrt(gx² + gy² + gz²)
        - Preserves spatial resolution
        
        Saves:
        -----
        <output_dir>_gradient-magnitude.nii
            Gradient magnitude map
            Range: 0 to maximum gradient (~50 for typical T1w)
            Higher values at tissue boundaries
        
        Notes
        -----
        - Uses ANTs iMath 'Grad' operation
        - Sigma parameter = 1 (minimal smoothing)
        - Values are absolute (always positive)
        - No masking applied to gradient itself
        """
        print(f'{CYAN}Computing gradient magnitude...{RESET}')

        # Compute gradient using iMath
        self._grad_input = ants.iMath(self._input, 'Grad', 1)

        # Save gradient magnitude map
        output_file = self._outputdir + '_gradient-magnitude.nii'
        ants.image_write(self._grad_input, output_file)
        
        # Compute statistics for reporting
        grad_data = self._grad_input.numpy()[self._mask.numpy() == 1]
        print(f'  {GREEN}Gradient magnitude computed{RESET}')
        print(f'    Range: {grad_data.min():.2f} to {grad_data.max():.2f}')
        print(f'    Mean: {grad_data.mean():.2f}')
        print(f'    Saved: {output_file}')

    def relative_intensity(self):
        """
        Compute relative intensity map with intensity normalization.
        
        Calculates intensity values normalized relative to the GM-WM boundary,
        accounting for scanner and protocol variations. This creates contrast-
        normalized values centered at 100, improving cross-scanner consistency.
        
        Steps:
        1. Mask input to GM and WM tissues
        2. Find intensity peaks for GM and WM (histogram mode)
        3. Compute background: BG = 0.5 × (GM_peak + WM_peak)
        4. Calculate relative intensity for each voxel:
           - Below BG: RI = 100 × (1 - (BG - I) / BG)
           - Above BG: RI = 100 × (1 + (I - BG) / BG)
        5. Apply Gaussian smoothing (σ=3mm FWHM)
        
        Saves:
        -----
        <output_dir>_relative-intensity.nii
            Relative intensity map
            Range: Typically 50-150
            Value of 100 represents GM-WM boundary
            Smoothed with σ=3mm FWHM
        
        Notes
        -----
        - Based on Nyúl et al. intensity standardization
        - Percentile range: 1st to 99.5th (removes outliers)
        - Smoothing reduces noise while preserving features
        - Values outside mask remain at 0
        
        References
        ----------
        Nyúl LG, Udupa JK, Zhang X. New variants of a method of MRI scale 
        standardization. IEEE Trans Med Imaging. 2000;19(2):143-150.
        """
        print(f'{CYAN}Computing relative intensity...{RESET}')

        # Mask input to GM and WM
        input_n4_gm = self._input * self._input.new_image_like(self._gm)
        input_n4_wm = self._input * self._input.new_image_like(self._wm)
        
        # Find intensity peaks and compute background
        bg_input = peakfinder(input_n4_gm, input_n4_wm, 1, 99.5)
        print(f'  Background intensity: {bg_input:.2f}')
        
        # Compute relative intensity
        input_ri = compute_RI(self._input.numpy(), bg_input, self._mask.numpy())
        
        # Create ANTs image and smooth
        tmp = self._input.new_image_like(input_ri)
        self._ri = ants.smooth_image(tmp, sigma=3, FWHM=True)
        
        # Save relative intensity map
        output_file = self._outputdir + '_relative-intensity.nii'
        ants.image_write(self._ri, output_file)
        
        # Compute statistics for reporting
        ri_data = self._ri.numpy()[self._mask.numpy() == 1]
        print(f'  {GREEN}Relative intensity computed{RESET}')
        print(f'    Range: {ri_data.min():.2f} to {ri_data.max():.2f}')
        print(f'    Mean: {ri_data.mean():.2f}')
        print(f'    Saved: {output_file}')

    def file_processor(self):
        """
        Execute complete texture feature extraction pipeline.
        
        Runs all processing steps in sequence:
        1. Load input files
        2. Tissue segmentation (GM, WM, CSF)
        3. Gradient magnitude computation
        4. Relative intensity computation
        
        Prints processing time upon completion.
        
        Raises
        ------
        FileNotFoundError
            If input or mask file not found.
        RuntimeError
            If any processing step fails.
        
        Notes
        -----
        - Processing time typically 2-5 minutes
        - All outputs saved automatically
        - Progress messages printed throughout
        - Total time includes loading and all computations
        """
        start = time.time()
        
        print(f'\n{CYAN}{BOLD}Starting texture feature extraction pipeline{RESET}\n')
        
        self.load_nifti_file()
        self.segmentation()
        self.gradient_magnitude()
        self.relative_intensity()
        
        end = time.time()
        elapsed = np.round(end - start, 1)
        
        print(f'\n{GREEN}{BOLD}Pipeline completed successfully!{RESET}')
        print(f'  Processing time: {elapsed} seconds ({elapsed/60:.1f} minutes)')
        print(f'  Output files:')
        print(f'    {self._outputdir}_gradient-magnitude.nii')
        print(f'    {self._outputdir}_relative-intensity.nii\n')


def run_texture_pipeline(input, mask, output_dir):
    """
    Run the neuroimaging texture feature extraction pipeline.
    
    This function initializes and executes a texture analysis pipeline on a
    neuroimaging volume. The pipeline computes gradient magnitude and relative
    intensity features from the input image within regions defined by the mask.
    Results are saved to the specified output directory.
    
    The pipeline includes:
    1. Automatic tissue segmentation (GM, WM, CSF)
    2. Gradient magnitude computation (edge detection)
    3. Relative intensity calculation (contrast normalization)
    
    Parameters
    ----------
    input : str
        Path to the input MRI image file (.nii.gz).
        Typically a preprocessed T1w volume with bias correction.
    mask : str
        Path to the binary brain mask file (.nii.gz).
        Defines regions for texture analysis (1=brain, 0=background).
    output_dir : str
        Output prefix for computed texture feature maps.
        Creates files: <output_dir>_gradient-magnitude.nii.gz
                      <output_dir>_relative-intensity.nii.gz
    
    Returns
    -------
    None
        The function saves texture feature maps to disk but does not return values.
    
    Raises
    ------
    FileNotFoundError
        If input or mask file does not exist.
    RuntimeError
        If any processing step fails.
    
    Examples
    --------
    >>> # Basic usage
    >>> run_texture_pipeline(
    ...     input="T1w_preprocessed.nii.gz",
    ...     mask="brain_mask.nii.gz",
    ...     output_dir="subject01_textures"
    ... )
    
    >>> # With full paths
    >>> run_texture_pipeline(
    ...     input="/data/preprocessed/sub-01_T1w.nii.gz",
    ...     mask="/data/masks/sub-01_mask.nii.gz",
    ...     output_dir="/results/radiomics/sub-01"
    ... )
    
    Notes
    -----
    - The function relies on the noelTexturesPy class
    - Processing time: 2-5 minutes per subject
    - Memory requirement: ~2-4 GB RAM
    - Preprocessing recommended: N4 bias correction, denoising
    - All features saved in NIfTI format
    - Progress messages printed throughout execution
    
    See Also
    --------
    noelTexturesPy : Class implementing the texture extraction pipeline
    """
    pipeline = noelTexturesPy(
        id='textures',
        output_dir=output_dir,
        input=input,
        mask=mask,
    )
    pipeline.file_processor()


if __name__ == "__main__":
    # Check if no arguments were provided or help was requested
    if len(sys.argv) == 1 or "-h" in sys.argv or "--help" in sys.argv:
        print_help_message()
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="Generate texture features from neuroimaging data",
        add_help=False  # Use custom help
    )
    parser.add_argument("--input", "-i", required=True, 
                       help="Path to the input MRI image file (.nii.gz)")
    parser.add_argument("--mask", "-m", required=True, 
                      help="Path to the binary brain mask file (.nii.gz)")
    parser.add_argument("--output", "-o", required=True, 
                      help="Output prefix for texture feature maps")
    
    args = parser.parse_args()
    
    try:

        # Validate input files exist
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        if not os.path.exists(args.mask):
            raise FileNotFoundError(f"Mask file not found: {args.mask}")
        
        print(f"{CYAN}Configuration:{RESET}")
        print(f"  Input: {args.input}")
        print(f"  Mask: {args.mask}")
        print(f"  Output prefix: {args.output}")
        print()
        
        # Run pipeline
        run_texture_pipeline(args.input, args.mask, args.output)
        
        sys.exit(0)
        
    except FileNotFoundError as e:
        print(f"\n{RED}{BOLD}File not found:{RESET}")
        print(f"  {str(e)}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n{RED}{BOLD}Error during texture extraction:{RESET}")
        print(f"  {str(e)}")
        print(f"\n{YELLOW}Run 'micaflow texture_generation --help' for usage information.{RESET}")
        sys.exit(1)