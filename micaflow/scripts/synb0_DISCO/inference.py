import numpy as np
import torch
import nibabel as nib
import argparse
from .model import UNet3D
from . import util

def inference(T1_path, b0_d_path, model, device):
    # Eval mode
    model.eval()
    
    # Disable gradient computation for inference
    with torch.no_grad():
        # Get image
        img_T1 = np.expand_dims(util.get_nii_img(T1_path), axis=3)
        img_b0_d = np.expand_dims(util.get_nii_img(b0_d_path), axis=3)
        print('T1 shape: ' + str(img_T1.shape))
        print('b0_d shape: ' + str(img_b0_d.shape))

        # Convert to torch img format
        img_T1 = util.nii2torch(img_T1)
        img_b0_d = util.nii2torch(img_b0_d)

        # Normalize data
        img_T1 = util.normalize_img(img_T1, 150, 0, 1, -1)
        max_img_b0_d = np.percentile(img_b0_d, 99)
        min_img_b0_d = 0
        img_b0_d = util.normalize_img(img_b0_d, max_img_b0_d, min_img_b0_d, 1, -1)

        # Calculate padding needed to make dimensions divisible by 8
        original_shape = img_T1.shape[2:5]
        pad_dims = []
        for i in range(3):  # For the 3 spatial dimensions
            dim_size = original_shape[i]
            remainder = dim_size % 8
            if remainder != 0:
                padding_needed = 8 - remainder
                # Distribute padding as evenly as possible
                pad_before = padding_needed // 2
                pad_after = padding_needed - pad_before
                pad_dims.append((pad_before, pad_after))
            else:
                # No padding needed for this dimension
                pad_dims.append((0, 0))
        
        # Apply padding if needed
        if any(sum(p) > 0 for p in pad_dims):
            print(f"Padding dimensions to be divisible by 8: {pad_dims}")
            # FIX: Properly structure padding as a tuple of tuples: ((before,after), (before,after), ...)
            padding = ((0, 0), (0, 0),  # No padding for batch and channel dims
                     (pad_dims[0][0], pad_dims[0][1]),  # Depth padding
                     (pad_dims[1][0], pad_dims[1][1]),  # Height padding
                     (pad_dims[2][0], pad_dims[2][1]))  # Width padding
            
            # Apply same padding to both images
            img_T1 = np.pad(img_T1, padding, 'constant')
            img_b0_d = np.pad(img_b0_d, padding, 'constant')
            print(f"Padded shape: {img_T1.shape[2:5]}")

        # Set "data"
        img_data = np.concatenate((img_b0_d, img_T1), axis=1)

        # Send data to device
        img_data = torch.from_numpy(img_data).float().to(device)
        print("Passing to model...")
        
        # Pass through model
        img_model = model(img_data)
        
        print("Model complete")
        # Unnormalize model
        img_model = util.unnormalize_img(img_model, max_img_b0_d, min_img_b0_d, 1, -1)

        # Remove padding if added
        if any(sum(p) > 0 for p in pad_dims):
            # Remove the padding we added earlier
            img_model = img_model[:, :, 
                       pad_dims[0][0]:(None if pad_dims[0][1]==0 else -pad_dims[0][1]),
                       pad_dims[1][0]:(None if pad_dims[1][1]==0 else -pad_dims[1][1]),
                       pad_dims[2][0]:(None if pad_dims[2][1]==0 else -pad_dims[2][1])]
            print(f"Removed padding. Final shape: {img_model.shape[2:5]}")
        else:
            # If dimensions were already divisible by 8, use the old hard-coded padding removal
            img_model = img_model[:, :, 2:-1, 2:-1, 3:-2]
            print("Using default padding removal")

        # Return model
        return img_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic B0 image')
    parser.add_argument('t1_path', help='Path to T1w input image')
    parser.add_argument('b0_path', help='Path to distorted B0 input image')
    parser.add_argument('output_path', help='Path for output synthetic B0')
    parser.add_argument('model_path', help='Path to model weights')
    parser.add_argument('--threads', type=int, default=None, 
                      help='Number of CPU threads to use (default: system setting)')
    parser.add_argument('--cpu', action='store_true',
                      help='Force CPU usage even if GPU is available')
    args = parser.parse_args()
    
    # Set thread count if specified
    if args.threads is not None:
        torch.set_num_threads(args.threads)
        print(f"Setting thread count to {args.threads}")
    
    # Determine device
    if not args.cpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print('T1 input path: ' + args.t1_path)
    print('b0 input path: ' + args.b0_path)
    print('b0 output path: ' + args.output_path)
    print('Model path: ' + args.model_path)

    # Run code ---------------------------------------------#

    # Get model
    model = UNet3D(2, 1).to(device)
    model.load_state_dict(torch.load(args.model_path))

    # Inference
    img_model = inference(args.t1_path, args.b0_path, model, device)

    # Save
    nii_template = nib.load(args.b0_path)
    nii = nib.Nifti1Image(util.torch2nii(img_model.detach().cpu()), nii_template.affine, nii_template.header)
    nib.save(nii, args.output_path)
