import os.path
from multiprocessing import Pool
import sys
import SimpleITK as sitk
import torch
from batchgenerators.utilities.file_and_folder_operations import (
    nifti_files,
    join,
    maybe_mkdir_p,
    isdir,
)

sys.stdout = open(os.devnull, "w")
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

sys.stdout = sys.__stdout__
from paths import folder_with_parameter_files
import tempfile
import shutil
import os
import shutil
import glob


def move_files_by_extension(source_dir, target_dir, extension):
    """
    Move all files with a specific extension from source_dir to target_dir.

    Args:
        source_dir: Source directory containing the files
        target_dir: Target directory where files will be moved
        extension: File extension to match (e.g., '.nii.gz')
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Find all files with the specified extension
    for file_path in glob.glob(os.path.join(source_dir, f"*{extension}")):
        print(f"Processing: {file_path}")
        # Get just the filename
        filename = os.path.basename(file_path)
        # Move the file
        shutil.move(file_path, os.path.join(target_dir, filename))
        print(f"Moved: {filename}")


def apply_bet(img, bet, out_fname):
    img_itk = sitk.ReadImage(img)
    img_npy = sitk.GetArrayFromImage(img_itk)
    img_bet = sitk.GetArrayFromImage(sitk.ReadImage(bet))
    img_npy[img_bet == 0] = 0
    out = sitk.GetImageFromArray(img_npy)
    out.CopyInformation(img_itk)
    sitk.WriteImage(out, out_fname)


def get_hdbet_predictor(
    use_tta: bool = False,
    device: torch.device = torch.device("cuda"),
    verbose: bool = False,
):
    os.environ["nnUNet_compile"] = "F"
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=use_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=verbose,
        verbose_preprocessing=verbose,
    )
    predictor.initialize_from_trained_model_folder(folder_with_parameter_files, "all")
    if device == torch.device("cpu"):
        torch.set_num_threads(os.cpu_count())
    return predictor


def hdbet_predict(
    input_file_or_folder: str,
    output_file_or_folder: str,
    predictor: nnUNetPredictor,
    keep_brain_mask: bool = False,
    compute_brain_extracted_image: bool = True,
):
    temp_dir = tempfile.mkdtemp(prefix="hdbet_temp_")
    original_cwd = os.getcwd()
    # find input file or files
    if os.path.isdir(input_file_or_folder):
        input_files = nifti_files(input_file_or_folder)
        # output_file_or_folder must be folder in this case
        maybe_mkdir_p(output_file_or_folder)
        output_files = [
            join(output_file_or_folder, os.path.basename(i)) for i in input_files
        ]
        brain_mask_files = [i[:-7] + "_bet.nii.gz" for i in output_files]
    else:
        assert not isdir(
            output_file_or_folder
        ), "If input is a single file then output must be a filename, not a directory"
        assert output_file_or_folder.endswith(
            ".nii.gz"
        ), "Output file must end with .nii.gz"
        input_files = [input_file_or_folder]
        output_files = [join(os.path.curdir, output_file_or_folder)]
        brain_mask_files = [
            join(os.path.curdir, output_file_or_folder[:-7] + "_bet.nii.gz")
        ]
    try:
        # Change to temp directory for the prediction
        os.chdir(temp_dir)
        # we first just predict the brain masks using the standard nnU-Net inference
        predictor.predict_from_files(
            [[i] for i in input_files],
            brain_mask_files,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=4,
            num_processes_segmentation_export=8,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )

        if compute_brain_extracted_image:
            # now brain extract the images
            res = []
            with Pool(4) as p:
                for im, bet, out in zip(input_files, brain_mask_files, output_files):
                    res.append(p.starmap_async(apply_bet, ((im, bet, out),)))
                [i.get() for i in res]
        print(output_file_or_folder)
        print(
            "files: ",
            os.listdir(os.path.join(temp_dir, os.path.dirname(output_file_or_folder))),
        )
        os.makedirs(
            os.path.join(original_cwd, os.path.dirname(output_file_or_folder)),
            exist_ok=True,
        )
        move_files_by_extension(
            os.path.join(temp_dir, os.path.dirname(output_file_or_folder)),
            os.path.join(original_cwd, os.path.dirname(output_file_or_folder)),
            ".nii.gz",
        )
        if not keep_brain_mask:
            [os.remove(i) for i in brain_mask_files]
    except Exception as e:
        print(e)
    finally:
        # Always restore original directory
        os.chdir(original_cwd)
        # Clean up temp directory when done
        shutil.rmtree(temp_dir, ignore_errors=True)
