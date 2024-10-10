import os
import argparse
import numpy as np
import nibabel as nib
from Hessian_3D import vesselness3d
import SimpleITK as sitk
from skimage import exposure
import warnings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", help="where the T2/T1 files in")
    parser.add_argument("--input", help="Name of Input Subject")
    parser.add_argument(
        "--save_dir", help="Histgram enhanced file and output of frangi filter are saved in this directory.")

    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in true_divide")

    # sigma = [0.01, 0.025, 0.05, 0.075, 0.1]
    sigma = [0.01]
    a, b, c = 2, 0.5, 100

    # enhanced_image_name = args.input + 'T2/T1 file suffix'
    enhanced_image_name = args.input + '.nii.gz'
    input_path = os.path.join(args.path, enhanced_image_name)
    hist_enhanced_image_name = args.input + '_contrast.nii.gz'
    hist_enhanced_image_path = os.path.join(
        args.save_dir, hist_enhanced_image_name)

    # histgram contrast enhance
    t2_file = nib.load(input_path)
    t2_data = t2_file.get_fdata()
    t2_affine = t2_file.affine

    t_max, t_min = t2_data.max(), t2_data.min()
    t2_data = (t2_data - t_min) / (t_max - t_min)

    p10, p98 = np.percentile(t2_data, (10, 98))
    t2_data = exposure.rescale_intensity(t2_data, in_range=(p10, p98))

    nib.Nifti1Image(t2_data, t2_affine).to_filename(hist_enhanced_image_path)

    # frangi filter for histgram enhanced filter
    output_path = hist_enhanced_image_path.replace(
        '_contrast.nii.gz', '_contrast_out.nii.gz')

    img = sitk.ReadImage(hist_enhanced_image_path)
    img_data = sitk.GetArrayFromImage(img)
    space = img.GetSpacing()
    direction = img.GetDirection()
    origin = img.GetOrigin()
    img_data = 200-img_data
    v = vesselness3d(img_data, sigma, list(space), a, b, c)
    image_data = v.vesselness3d()
    img = sitk.GetImageFromArray(image_data)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    img.SetSpacing(space)
    sitk.WriteImage(img, output_path)
    
    img = nib.load(output_path)
    nib.Nifti1Image(img.get_fdata(), img.affine).to_filename(output_path)