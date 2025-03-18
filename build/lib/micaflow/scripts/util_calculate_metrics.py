import csv
import nibabel as nib
import argparse
import numpy as np
    
def apply_threshold(image_path, reference_path):
    """
    Round the values in image_path to the nearest integer values present in reference_path.
    
    Args:
        image_path (str): Path to the image to be rounded
        reference_path (str): Path to the reference image with target integer values
        
    Returns:
        str: Path to the new thresholded image
    """

    # Load both images
    img = nib.load(image_path)
    ref_img = nib.load(reference_path)
    
    # Get data
    data = img.get_fdata()
    ref_data = ref_img.get_fdata()
    
    # Get unique integer values in reference (excluding 0)
    ref_values = np.unique(ref_data)
    ref_values = ref_values[ref_values > 0]  # Remove background (0)
    
    if len(ref_values) > 0:
        # Vectorized approach - create a lookup table for all possible input values
        # First determine range of input values to create an efficient lookup table
        non_zero_data = data[data > 0]
        if len(non_zero_data) > 0:
            min_val = int(np.floor(np.min(non_zero_data)))
            max_val = int(np.ceil(np.max(non_zero_data)))
            
            # Create lookup table (much faster than calculating for each voxel)
            lookup_range = np.arange(min_val, max_val + 1)
            # Find nearest reference value for each possible input value
            lookup_values = np.zeros(len(lookup_range))
            for i, val in enumerate(lookup_range):
                if val <= 0:
                    lookup_values[i] = 0
                else:
                    idx = np.abs(ref_values - val).argmin()
                    lookup_values[i] = ref_values[idx]
                    
            # Create a mapping function using the lookup table
            def map_to_nearest(val):
                if val <= 0:
                    return 0
                idx = int(val) - min_val
                if 0 <= idx < len(lookup_values):
                    return lookup_values[idx]
                # Fallback for out-of-range values
                idx = np.abs(ref_values - val).argmin()
                return ref_values[idx]
                
            # Apply mapping - use vectorize for speed
            vfunc = np.vectorize(map_to_nearest)
            rounded_data = vfunc(data)
        else:
            # No non-zero values in input
            rounded_data = np.zeros_like(data)
    else:
        # If no reference values, just binarize the image
        rounded_data = (data > 0).astype(int)
    
    # Save the result
    new_image_path = image_path.replace(".nii", "_thr.nii")
    if ".nii" not in new_image_path:
        new_image_path = new_image_path + "_thr.nii"
        
    nib.save(nib.Nifti1Image(rounded_data.astype(np.int32), img.affine), new_image_path)
    return new_image_path

def Overlap(volume1_path, volume2_path, mask_path=None):
    """
    Calculate Jaccard index between two segmented volumes.
    
    Args:
        volume1_path (str): Path to first volume
        volume2_path (str): Path to second volume
        mask_path (str, optional): Path to mask volume
        
    Returns:
        dict: Dictionary containing ROI-wise Jaccard indices
    """
    import numpy as np
    
    # Load volumes
    vol1_img = nib.load(volume1_path)
    vol2_img = nib.load(volume2_path)
    
    vol1_data = vol1_img.get_fdata()
    vol2_data = vol2_img.get_fdata()
    
    # Apply mask if provided
    if mask_path:
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata().astype(bool)
        vol1_data = np.logical_and(vol1_data, mask_data)
        vol2_data = np.logical_and(vol2_data, mask_data)
    
    # Get unique ROIs (assuming ROIs are represented by integer values)
    roi_values = np.unique(np.where(vol1_data > 0, vol1_data, 0))
    roi_values = roi_values[roi_values > 0]  # Remove background (0)
    
    # Calculate Jaccard for each ROI
    roi_ji = []
    
    if len(roi_values) > 0:
        for roi in roi_values:
            roi1 = vol1_data == roi
            roi2 = vol2_data == roi
            
            intersection = np.logical_and(roi1, roi2).sum()
            union = np.logical_or(roi1, roi2).sum()
            
            # Calculate Jaccard index
            jaccard = intersection / union if union > 0 else 0
            roi_ji.append(jaccard)
        # Calculate global Jaccard if no ROIs found
        intersection = np.logical_and(vol1_data, vol2_data).sum()
        union = np.logical_or(vol1_data, vol2_data).sum()
        jaccard = intersection / union if union > 0 else 0
        roi_ji.append(jaccard)
    else:
        # Calculate global Jaccard if no ROIs found
        intersection = np.logical_and(vol1_data, vol2_data).sum()
        union = np.logical_or(vol1_data, vol2_data).sum()
        jaccard = intersection / union if union > 0 else 0
        roi_ji.append(jaccard)
    
    # Create a results object similar to nipype's Overlap
    class Results:
        class Outputs:
            def __init__(self, roi_ji):
                self.roi_ji = roi_ji
        
        def __init__(self, roi_ji):
            self.outputs = self.Outputs(roi_ji)
            
    return Results(roi_ji)

def main(image, reference, output_file, threshold=0.5, mask_path=None):
    # Apply threshold and use the new file paths

    # Use our custom Overlap function instead of nipype
    if mask_path:
        res = Overlap(image, reference, mask_path)
    else:
        res = Overlap(image, reference)

    # Print the number of ROIs
    num_rois = len(res.outputs.roi_ji)
    print("Number of ROIs:", num_rois)


    with open(output_file, "w", newline="") as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(["ROI", "Jaccard Index"])
        for i, ji in enumerate(res.outputs.roi_ji):
            csvwriter.writerow([i + 1, ji])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Calculate overlap metrics between two volumes")
    parser.add_argument("--input", "-i", required=True, help="First input volume")
    parser.add_argument("--reference", "-r", required=True, help="Reference volume to compare against")
    parser.add_argument("--output", "-o", required=True, help="Output CSV file path")
    parser.add_argument("--mask", "-m", help="Optional mask volume")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Threshold value (default: 0.5)")
    
    args = parser.parse_args()
    
    main(args.input, args.reference, args.output, threshold=args.threshold, mask_path=args.mask)