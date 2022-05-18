import imreg_dft as ird
from scipy.stats import gaussian_kde
import numpy as np
from tqdm import tqdm
from utils import region
from PIL import Image
from skimage.color import rgb2hed
import SimpleITK as sitk


def extract_H(img_he, split=False):
    if split:  # rgb2hed for large image
        he_hed = rgb2hed_split(img_he)[:, :, 0]
    else:
        he_rgb = np.array(img_he)[:, :, :3]
        he_hed = rgb2hed(he_rgb)[:, :, 0]
    region_he = he_hed - np.min(he_hed)
    region_he = region_he * 255 / np.max(region_he)
    region_he = 255 - np.asarray(region_he, dtype='uint8')

    return region_he


def zeropad(img, H, W):
    # zero padding when the image sizes are different in the first step
    h2, w2 = img.shape

    img_new = np.zeros((H, W))

    img_new[:h2, :w2] = img

    return img_new


class Registration:
    def __init__(self):
        self.tx = []
        self.ty = []
        self.posx = []
        self.posy = []
        self.rposx = []
        self.rposy = []
        self.images = {}

        self.first = None
        self.second = None
        self.nonrigid = None

        self.x_init_offset = 0
        self.y_init_offset = 0

    def set_all(self, tvec, posx, posy, rposx, rposy, image):
        self.tx.append(tvec[0])
        self.ty.append(tvec[1])
        self.posx.append(posx)
        self.posy.append(posy)
        self.rposx.append(rposx)
        self.rposy.append(rposy)
        self.images["{}:{}".format(posx, posy)] = image

    def find_peak(self, scale):
        # peak finder
        xy = np.vstack([self.tx, self.ty])
        kde = gaussian_kde(xy)
        density = kde(xy)
        (ymax, xmax) = xy.T[np.argmax(density)]
        self.xshift = int((2 ** scale) * xmax + self.x_init_offset)
        self.yshift = int((2 ** scale) * ymax + self.y_init_offset)

    def get_shift(self):
        return self.xshift, self.yshift

    def get_init_shift(self):
        return (self.x_init_offset, self.y_init_offset)

    def rigid_registration(self, region_ihc, region_he, scale,
                           only_return=False, is_show=False):
        maxH = max(region_ihc.shape[0], region_he.shape[0])
        maxW = max(region_ihc.shape[1], region_he.shape[1])
        region_ihc = zeropad(region_ihc, maxH, maxW)
        region_he = zeropad(region_he, maxH, maxW)

        result = ird.translation(region_ihc, region_he)

        if only_return:
            return result

        if is_show:
            timg = ird.transform_img(region_he, scale=1.0, angle=result['angle'], tvec=result['tvec'])
            ird.imshow(region_ihc, region_he, timg)
            print(result['tvec'])

        self.x_init_offset, \
        self.y_init_offset = int(result['tvec'][1] * (2 ** scale)), \
                             int(result['tvec'][0] * (2 ** scale))

    def peak_rigid_registration(self, op_ihc, op_he, scale=1, step_img=25, imgsize=500):
        shift = (self.x_init_offset, self.y_init_offset)

        width, height = op_he.dimensions
        for ii, xx in enumerate(tqdm(range(int(width / step_img), width - imgsize, int(width / step_img)))):
            for jj, yy in enumerate(range(int(width / step_img), height - imgsize, int(height / step_img))):
                region_ihc, region_he = region.get_region(op_ihc, op_he, pos=(xx, yy),
                                                          size=(imgsize, imgsize), scale=scale,
                                                          shift=shift)

                if np.mean(region_he) > 5:
                    result = self.first_registration(region_ihc, region_he, None, only_return=True)
                    self.set_all(result['tvec'], ii, jj, xx, yy, region_he)

        self.find_peak(scale)

    def nonrigid_registration(self, region_HE, region_dapi, region_target):

        he_h = extract_H(Image.fromarray(region_HE)).astype(np.float32)
        ihc_dapi = region_dapi.astype(np.float32)
        ihc_target = region_target.astype(np.float32)

        # from ndarray to sitk_image
        he_h = sitk.GetImageFromArray(he_h)
        ihc_dapi = sitk.GetImageFromArray(ihc_dapi)
        ihc_target = sitk.GetImageFromArray(ihc_target)

        # histgram matching
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(1024)  # original=1024
        matcher.SetNumberOfMatchPoints(7)  # original=7
        matcher.ThresholdAtMeanIntensityOn()
        he_h_matched = matcher.Execute(he_h, ihc_dapi)  # matcher.Execute(moving, fixed)

        # demons_registration
        registration_method = sitk.ImageRegistrationMethod()

        # Create initial identity transformation.
        transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
        transform_to_displacment_field_filter.SetReferenceImage(he_h_matched)
        # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
        initial_transform = sitk.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute(
            sitk.TranslationTransform(2)))  # original = Execute(sitk.Transform())

        # Regularization (update field - viscous, total field - elastic).
        initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0)

        registration_method.SetInitialTransform(initial_transform)

        registration_method.SetMetricAsDemons(10)  # intensities are equal if the difference is less than 10HU

        # Multi-resolution framework.
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1]) #4,2,1
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[12, 8, 4, 0]) #8,4,0
        registration_method.SetInterpolator(sitk.sitkLinear)

        # If you have time, run this code as is, otherwise switch to the gradient descent optimizer
        registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20,
                                                                      convergenceMinimumValue=1e-6,
                                                                      convergenceWindowSize=10)
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=20,
                                                          convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        displacementField = registration_method.Execute(he_h_matched, ihc_dapi)  # .Execute(fixed, moving)
        outTx = sitk.DisplacementFieldTransform(displacementField)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(he_h_matched)
        resampler.SetInterpolator(sitk.sitkBSpline)  # original=(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(100)  # original=100
        resampler.SetTransform(outTx)
        # Execute
        ihc_dapi = resampler.Execute(ihc_dapi)
        ihc_target = resampler.Execute(ihc_target)

        ihc_dapi_reg = sitk.GetArrayFromImage(ihc_dapi).astype(np.uint8)
        ihc_target_reg = sitk.GetArrayFromImage(ihc_target).astype(np.uint8)

        return ihc_target_reg, ihc_dapi_reg
