import argparse
import os
import random
import string
from collections import Counter
import ants
import numpy as np
import time

def write_nifti(input, id, output_dir, type):
    output_fname = os.path.join(output_dir, id + '_' + type + '.nii.gz')
    ants.image_write(input, output_fname)


def compute_RI(image, bg, mask):
    ri = np.zeros_like(image)
    bgm = np.stack(np.where(np.logical_and(image < bg, mask == 1)), axis=1)
    bgm_ind = bgm[:, 0], bgm[:, 1], bgm[:, 2]
    bgp = np.stack(np.where(np.logical_and(image > bg, mask == 1)), axis=1)
    bgp_ind = bgp[:, 0], bgp[:, 1], bgp[:, 2]

    ri[bgm_ind] = 100 * (1 - (bg - image[bgm_ind]) / bg)
    ri[bgp_ind] = 100 * (1 + (bg - image[bgp_ind]) / bg)
    return ri


def peakfinder(gm, wm, lower_q, upper_q):
    gm_peak = Counter(threshold_percentile(gm, lower_q, upper_q)).most_common(1)[0][0]
    wm_peak = Counter(threshold_percentile(wm, lower_q, upper_q)).most_common(1)[0][0]
    bg = 0.5 * (gm_peak + wm_peak)
    return bg


def threshold_percentile(x, lower_q, upper_q):
    x = x.numpy()
    lq = np.percentile(x, lower_q)
    uq = np.percentile(x, upper_q)
    x = x[np.logical_and(x > lq, x <= uq)]
    return x.flatten().round()


def find_logger_basefilename(logger):
    """Finds the logger base filename(s) currently there is only one"""
    log_file = None
    handler = logger.handlers[0]
    log_file = handler.baseFilename
    return log_file


def random_case_id():
    letters = ''.join(random.choices(string.ascii_letters, k=16))
    digits = ''.join(random.choices(string.digits, k=16))
    x = letters[:3].lower() + '_' + digits[:4]
    return x

class noelTexturesPy:
    def __init__(
        self,
        id,
        output_dir=None,
        input=None,
        mask=None,
    ):
        super().__init__()
        self._id = id
        self._outputdir = output_dir
        self.input = input
        self.mask = mask

    def load_nifti_file(self):
        # load nifti data to memory
        print('loading nifti files')
        self._input = ants.image_read(self.input)
        self._mask = ants.image_read(self.mask)


    def segmentation(self):
        print('computing GM, WM, CSF segmentation')
        segm = ants.atropos(
            a=self._input,
            i='Kmeans[3]',
            m='[0.2,1x1x1]',
            c='[3,0]',
            x=self._mask,
        )
        self._segm = segm['segmentation']
        self._gm = np.where((self._segm.numpy() == 2), 1, 0).astype('float32')
        self._wm = np.where((self._segm.numpy() == 3), 1, 0).astype('float32')


    def gradient_magnitude(self):
        print('computing gradient magnitude')

        self._grad_input = ants.iMath(self._input, 'Grad', 1)

        ants.image_write(
            self._grad_input,
            self._outputdir + '_gradient-magnitude.nii',
        )


    def relative_intensity(self):
        print('computing relative intensity')

        input_n4_gm = self._input * self._input.new_image_like(self._gm)
        input_n4_wm = self._input * self._input.new_image_like(self._wm)
        bg_input = peakfinder(input_n4_gm, input_n4_wm, 1, 99.5)
        input_ri = compute_RI(self._input.numpy(), bg_input, self._mask.numpy())
        tmp = self._input.new_image_like(input_ri)
        self._ri = ants.smooth_image(tmp, sigma=3, FWHM=True)
        ants.image_write(
            self._ri,
            self._outputdir + '_relative-intensity.nii',
        )


    def file_processor(self):
        start = time.time()
        self.load_nifti_file()
        self.segmentation()
        self.gradient_magnitude()
        self.relative_intensity()
        # self.create_zip_archive()
        end = time.time()
        print(
            'pipeline processing time elapsed: {} seconds'.format(
                np.round(end - start, 1)
            )
        )

def run_texture_pipeline(input, mask, output_dir):
    """Run the neuroimaging texture feature extraction pipeline.
    
    This function initializes and executes a texture analysis pipeline on a neuroimaging volume.
    The pipeline computes various texture features (e.g., gradient magnitude, relative intensity,
    local binary patterns) from the input image within the regions defined by the mask.
    Results are saved to the specified output directory.
    
    Parameters
    ----------
    input : str
        Path to the input image file (typically a preprocessed MRI volume).
    mask : str
        Path to the binary mask file that defines regions of interest for texture analysis.
    output_dir : str
        Directory where the computed texture feature maps will be saved.
    
    Returns
    -------
    None
        The function saves texture feature maps to the output directory but does not return values.
        
    Notes
    -----
    The function relies on the noelTexturesPy class which implements multiple texture
    feature extraction algorithms specifically designed for neuroimaging data.
    """
    pipeline = noelTexturesPy(
        id='textures',
        output_dir=output_dir,
        input=input,
        mask=mask,
    )
    pipeline.file_processor()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run texture pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input image file")
    parser.add_argument("--mask", "-m", required=True, help="Input mask file")
    parser.add_argument(
        "--output", "-o", required=True, help="Output corrected image file"
    )
    args = parser.parse_args()
    run_texture_pipeline(args.input, args.mask, args.output)
