import sys
import numpy as np
import cv2
import logging
from typing import Tuple

from .configuration import Config



import sys
import logging
from typing import Tuple
import numpy as np
import cv2


DEFAULT_CLAHE_LIMIT: float = 2000.
DEFAULT_CLAHE_COEF: float = 500.
DEFAULT_POLAR_SHAPE: Tuple[int, int] = (512, 1024)
MAX_GRID_CACHE_SIZE: int = 5
DEFAULT_BEAM_CENTER: Tuple[float, float] = (0, 0)
DEFAULT_ALGORITHM: int = cv2.INTER_CUBIC
DEFAULT_CLAHE_LIMIT: float = 2000.
DEFAULT_CLAHE_COEF: float = 500.

def normalize_with_std_mean(img: np.array, mean= None, std= None):
    if mean is None:
        mean = 0.485
    if std is None:
        std = 0.229
    return (img - mean) / std

def normalize(img: np.ndarray, nonzero_indices: np.ndarray) -> np.ndarray:
    return (img[nonzero_indices] - np.nanmin(img[nonzero_indices])) / (np.nanmax(img[nonzero_indices]) - np.nanmin(img[nonzero_indices]))


def normalize_image(image, mean=0.5, std=0.1):
    # Convert image to float32
    image = image.astype(np.float32)
    
    # Normalize to range [0, 1], ignoring zeros
    nonzero_indices = image != 0    
    #image = (image - image.min()) / (image.max() - image.min())
    nonzero_values = image[nonzero_indices]
    min_nonzero = np.min(nonzero_values)
    max_nonzero = np.max(nonzero_values)
    image[nonzero_indices] = (image[nonzero_indices] - min_nonzero) / (max_nonzero - min_nonzero)
    
    # Normalize to desired mean and std
    image[nonzero_indices] = image[nonzero_indices] * std + (mean-(std/2))
    
    return image

def clahe_func(img, limit: float = DEFAULT_CLAHE_LIMIT):

    return cv2.createCLAHE(clipLimit=limit, tileGridSize=(1, 1)).apply(np.clip(img, 0, 65535).astype('uint16')).astype(np.float32)

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

def _contrast_correction(
        config: Config = None,    
        img: np.array = None, 
        limit: float = DEFAULT_CLAHE_LIMIT,
        coef: float = DEFAULT_CLAHE_COEF,
        clahe: bool = True,
        log: bool = True,
        linear_normalization = False,
        linear_perc_997 = False
):

    if config is not None:
        linear_normalization = config.PREPROCESSING_LINEAR_CONTRAST
        linear_perc_997 = config.PREPROCESSING_LINEAR_PERC_977
        if config.PREPROCESSING_NO_CONTRASTCORRECTION:
            linear_normalization = False
            log = False
            clahe = False

    mask = ~np.isnan(img) & (img > 0)

    if linear_normalization:
        if linear_perc_997:
            upper_clip_limit = np.percentile(img[mask],97.0)
        else:
            upper_clip_limit = np.percentile(img[mask],99.9)

        lower_clip_limit = np.percentile(img[mask],5)

        img[mask] = np.clip(img[mask], lower_clip_limit, upper_clip_limit)
        img[mask] =  normalize(img, mask)
        img = img *255

        img = cv2.equalizeHist(img.astype(np.uint8))

        img = img /255
        img = img.astype(np.float32)
        img[~mask] = 0
        return img, mask

    if log:
        img = np.log10(img * coef + 1)
        img[mask] =  normalize(img, mask)

    if clahe:
        img = clahe_func(img * coef, limit)
        img[mask] = normalize(img,mask)
        img[~mask] = 0

    return img, mask

def contrast_correction(config, raw_polar_img: np.array):
    return _contrast_correction(config, raw_polar_img)


def add_batch_and_color_channel(img: np.array):
    img = np.repeat(img[ np.newaxis, :, :], 1, axis=0)
    return np.repeat(img[ np.newaxis, :, :], 1, axis=0)

def grayscale_to_color(img: np.array):
    return np.concatenate((img,)*3, axis=1)

def _set_q_max(config: Config):
    config.GEO_QMAX = np.sqrt(((config.GEO_RECIPROCAL_SHAPE[0] / config.GEO_PIXELPERANGSTROEM) ** 2) + ((config.GEO_RECIPROCAL_SHAPE[1] / config.GEO_PIXELPERANGSTROEM) ** 2))

def get_q_max(config: Config):
    _set_q_max(config)
    return config.GEO_QMAX

def _get_quazipolar_grid(config, beam_center: Tuple[float, float] = DEFAULT_BEAM_CENTER,
                         shape: Tuple[int, int] = (10, 10),
                         polar_shape: Tuple[int, int] = DEFAULT_POLAR_SHAPE,
                         coef: float = 0.6,
                         ):


    if config.PREPROCESSING_CUDA:
        xp = cp
    else:
        xp = np

    y0, z0 = beam_center
    y = np.arange(shape[1], dtype=np.float32) - y0
    z = np.arange(shape[0], dtype=np.float32) - z0
    zz, yy = np.meshgrid(z, y)  # meshgrid order: (z, y)

    rr = np.sqrt(yy ** 2 + zz ** 2)
    phi = np.arctan2(zz, yy)
    r_range = rr.min(), rr.max()
    phi_range = phi.min(), phi.max()

    phi = np.linspace(*phi_range, polar_shape[0], dtype=np.float32)
    r = np.linspace(*r_range, polar_shape[1], dtype=np.float32)

    r_matrix = np.repeat(r[None, :], polar_shape[0], axis=0)
    p_matrix = np.repeat(phi[:, None], polar_shape[1], axis=1)

    p_coef = coef / (1e-4 + r_matrix / r_matrix.max())
    p_matrix = np.minimum(p_matrix * p_coef, np.pi)

    polar_yy = r_matrix * np.cos(p_matrix) + y0
    polar_zz = r_matrix * np.sin(p_matrix) + z0

    return polar_yy, polar_zz


def _get_polar_grid(config,
        img_shape: Tuple[int, int],
        polar_shape: Tuple[int, int],
        beam_center: Tuple[float, float],
):

    if config.PREPROCESSING_CUDA:
        xp = cp
    else:
        xp = np

    y0, z0 = beam_center

    y = (xp.arange(img_shape[1]) - y0)
    z = (xp.arange(img_shape[0]) - z0)

    yy, zz = xp.meshgrid(y, z)
    rr = xp.sqrt(yy ** 2 + zz ** 2)
    phi = xp.arctan2(zz, yy)
    r_range = (rr.min(), rr.max())
    phi_range = phi.min(), phi.max()

    phi = xp.linspace(*phi_range, polar_shape[0])
    r = xp.linspace(*r_range, polar_shape[1])

    r_matrix = r[xp.newaxis, :].repeat(polar_shape[0], axis=0)
    p_matrix = phi[:, xp.newaxis].repeat(polar_shape[1], axis=1)

    polar_yy = r_matrix * xp.cos(p_matrix) + y0
    polar_zz = r_matrix * xp.sin(p_matrix) + z0


    return polar_yy, polar_zz


def _calc_polar_img(config, img: np.ndarray, yy: np.ndarray, zz: np.ndarray, algorithm: int) -> np.ndarray or None:
    try:
        if config.PREPROCESSING_CUDA:
            return cv2.cuda.remap(img,
                            yy,
                            zz,
                            interpolation=algorithm)
        else:
            return cv2.remap(img.astype(np.float32),
                yy.astype(np.float32),
                zz.astype(np.float32),
                interpolation=algorithm)

    except cv2.error:
        logging.error("Error in polar conversion!")
        sys.exit()


def calc_quazipolar_image(config, img: np.ndarray,
                          beam_center: Tuple[float, float] = DEFAULT_BEAM_CENTER,
                          polar_shape: Tuple[int, int] = DEFAULT_POLAR_SHAPE,
                          algorithm=cv2.INTER_LINEAR, coef: float = 0.6) -> np.ndarray or None:

    # Detect if input is CUDA or numpy and extract shape
    if isinstance(img, cv2.cuda_GpuMat):
        height, width = img.size()
        img_shape = (height, width)
    elif isinstance(img, np.ndarray):
        img_shape = img.shape

    yy, zz = _get_quazipolar_grid(config, beam_center, img_shape, polar_shape, coef=coef)

    return _calc_polar_img(config, img, yy, zz, algorithm)

def calc_polar_image(config,
        img: np.ndarray,
        polar_shape: Tuple[int, int] = DEFAULT_POLAR_SHAPE,
        beam_center: Tuple[float, float] = DEFAULT_BEAM_CENTER,
        algorithm: int = DEFAULT_ALGORITHM,
) -> np.ndarray or None:
    yy, zz = _get_polar_grid(config, img.shape, polar_shape, beam_center)

    return _calc_polar_img(config, img, yy, zz, algorithm)


def preprocess_geometry(config, raw_reciprocal_img: np.array):
    get_q_max(config)
    if config.PREPROCESSING_QUAZIPOLAR:
        return calc_quazipolar_image(config, raw_reciprocal_img, polar_shape=config.PREPROCESSING_POLAR_SHAPE)
    else:
        return calc_polar_image(config, raw_reciprocal_img, polar_shape=config.PREPROCESSING_POLAR_SHAPE)

def standard_preprocessing(config, raw_reciprocal_img: np.array, counter = None):

    config.GEO_RECIPROCAL_SHAPE = list(raw_reciprocal_img.shape)

    if config.PREPROCESSING_CUDA:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(raw_reciprocal_img)
        raw_reciprocal_img = gpu_img
        
    #conversion to polar coordinates
    if config.PREPROCESSING_POLAR_CONVERSION:
        raw_polar_img = preprocess_geometry(config, raw_reciprocal_img)
    else:
        raw_polar_img = raw_reciprocal_img

    equalized_polar, mask = contrast_correction(config, raw_polar_img)
    equalized_polar = add_batch_and_color_channel(equalized_polar)
    mask = add_batch_and_color_channel(mask)

    #reshape for detr model
    if config.MODEL_TYPE == 'detr':
        equalized_polar = grayscale_to_color(equalized_polar)
        equalized_polar = equalized_polar[:,:,:,:]
        #equalized_polar = np.pad(equalized_polar, ((0,0),(0,0,),(0,496), (0,0)))

    if config.PREPROCESSING_CUDA:
        equalized_polar = cp.asnumpy(equalized_polar)
        raw_polar_img = cp.asnumpy(raw_polar_img)
        mask = cp.asnumpy(mask)

    return equalized_polar, raw_polar_img, mask