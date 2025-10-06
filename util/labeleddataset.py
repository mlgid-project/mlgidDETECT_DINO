"""EXAMPLE USAGE:
data = H5GIWAXSDataset(dataset, buffer_size=5, unskewed_polar=True)
    for i, giwaxs_img_container in enumerate(data.iter_images()):

        giwaxs_img = giwaxs_img_container.converted_polar_image
        raw_giwaxs_img = giwaxs_img_container.raw_polar_image
        labels = giwaxs_img_container.polar_labels
        fits = giwaxs_img_container.fits
"""

import sys
sys.path.append("..")
import re
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
from typing import Tuple
import cv2 as cv
from h5py import File, Group
from typing import Union
from torch import Tensor
from multiprocessing import Process, Queue, Value
from typing import Iterator
from .exp_preprocess import preprocess_geometry, contrast_correction
from .imgcontainer import ImageContainer, Labels

DEFAULT_CLAHE_LIMIT: float = 2000.
DEFAULT_CLAHE_COEF: float = 500.
DEFAULT_POLAR_SHAPE: Tuple[int, int] = (512, 512)
MAX_GRID_CACHE_SIZE: int = 5
DEFAULT_BEAM_CENTER: Tuple[float, float] = (0, 0)
DEFAULT_ALGORITHM: int = cv.INTER_CUBIC


@dataclass
class H5GIWAXSDataset():
    """Container for a GIWAXS dataset. Contains multiple images with Labels.
    the images are loaded in a serarate process and stored in image_queue.
    The dataset can be iterated over with the function iter_images:
    e.g.

    data = H5GIWAXSDataset(dataset, buffer_size=5, unskewed_polar=True)
    for i, giwaxs_img_container in enumerate(data.iter_images()):

        giwaxs_img = giwaxs_img_container.converted_polar_image
        raw_giwaxs_img = giwaxs_img_container.raw_polar_image
        labels = giwaxs_img_container.polar_labels
        fits = giwaxs_img_container.fits
    """

    # Path to the dataset
    config: 'Config' = None
    path: Path = None
    preprocess_func: callable = None

    file_keys: list = field(default_factory=list)
    file_group: Group = None
    #min confidence of loaded boxes
    min_confidence: float = None
    #shape of the desired polar images
    polar_img_shape: tuple = DEFAULT_POLAR_SHAPE


    load_worker: Process = None
    #buffer size of the worker. E.g. 3 are loaded in advance by default.
    buffer_size: int = 3
    image_queue: Queue = Queue()
    image_metrics: list = field(default_factory=list)


    def __post_init__(self):
        if self.config is not None:
            self.polar_img_shape: tuple = self.config.PREPROCESSING_POLAR_SHAPE        
        with File(self.path, 'r') as f:
            self.file_keys = sum([
                ['/'.join([key, sub_key]) for sub_key in f[key].keys() if sub_key != 'metadata']
                for key in f.keys()
            ], [])
        self.load_worker_finished = Value('i',False)
        self.image_queue = Queue(self.buffer_size)
        self.load_worker = Process(target=load_worker, args=[self],daemon=True)
        self.load_worker.start()


    def iter_images(self) -> Iterator:
        """Iterator for Images, usage:
        for img in data.iter_images():
        

        Returns:
            iterator: Iterator of GIWAXSImage-objects
        """
        return iter(self.image_queue.get,None)
    
    def __iter__(self):
        return self

    def __next__(self):
        queue_object = self.image_queue.get()
        if queue_object is not None:
            return queue_object
        else:
            raise StopIteration

    def create_boxes(self, img: ImageContainer) -> None:
        """Calculates the coordinates of the boxes in the polar coordinates

        Args:
            img (GIWAXSImage): ImageObject of GIWAXS image
        """
        polar_shape = self.polar_img_shape
        reciprocal_labels = img.reciprocal_labels
        polar_labels = img.polar_labels
        
        max_radius = np.sqrt((np.array(img.reciprocal_img_shape) ** 2).sum())
        r_scale = polar_shape[1] / max_radius
        a_scale = polar_shape[0] / 90

        radii = reciprocal_labels.radii * r_scale
        widths = reciprocal_labels.widths  * r_scale / 2
        angles = reciprocal_labels.angles * a_scale
        angles_std = reciprocal_labels.angles_std * a_scale / 2

        boxes = np.stack([
            radii - widths, angles - angles_std, radii + widths, angles + angles_std
        ], -1)

        if self.min_confidence is not None:
            boxes = boxes[reciprocal_labels.confidences >= self.min_confidence]

        if  self.config.PREPROCESSING_QUAZIPOLAR:
            rs = ((boxes[:, 0] + boxes[:, 2]) / 2) / polar_shape[1]

            coef = 0.6 / (1e-4 + rs)

            boxes[:, 1::2] /= coef[:, None]
        
        polar_labels.boxes = boxes
        polar_labels.radii = radii
        polar_labels.widths = widths
        polar_labels.angles = angles
        polar_labels.angles_std = angles_std
        polar_labels.confidences = reciprocal_labels.confidences
        polar_labels.intensities = reciprocal_labels.intensities
        polar_labels.img_nr = reciprocal_labels.img_nr
        polar_labels.img_name = reciprocal_labels.img_name
    
    def close(self):
        pass

    

def load_worker(data_loader: H5GIWAXSDataset):
    """worker for the interaction with the H5 file and a automatic conversion to polar coordinates
        Intended to be spawned as a separate process and enqueue the results

    Args:
        data_loader (H5GIWAXSDataset): Dataset for which
    """

    for counter, key in enumerate(data_loader.file_keys):
        image = ImageContainer(data_loader.config)
        image.config = data_loader.config
        image.reciprocal_labels = Labels()
        with File(data_loader.path, 'r') as f:
            group = f[key]
            for h5key, python_key in zip(['confidence_level','radius','width','angle','angle_std','peak height','background (level)','background (slope)',''],
                                            ['confidences','radii','widths','angles','angles_std','intensities','background_levels','background_slopes']):
                setattr(image.reciprocal_labels, python_key, group['roi_data/' + h5key][()])
            image.reciprocal_labels.is_ring = [type ==1 for type in group['roi_data/type']]
            try:
                image.q_range = [range_param for range_param in group['metadata/qz_qxy_range_[A-1]']]
                #solution for q-range as string
                if len(image.q_range) == 1:
                    image.q_range =  re.findall(r"[-+]?(?:\d*\.*\d+)", str(image.q_range[0]))
                    image.q_range = [float(i) for i in image.q_range]
            except:
                pass
            image.raw_reciprocal = group['image'][()]
            image.reciprocal_img_shape = image.raw_reciprocal.shape
            image.reciprocal_labels.img_nr = counter
            image.reciprocal_labels.img_name = key
            data_loader.create_boxes(image)
            image.converted_polar_image, image.raw_polar_image, image.converted_mask = data_loader.preprocess_func(data_loader.config, image.raw_reciprocal, counter)
            data_loader.image_queue.put(image)    
    data_loader.image_queue.put(None)