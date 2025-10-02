from dataclasses import dataclass, field
from typing import Tuple
import numpy as np

DEFAULT_POLAR_SHAPE: Tuple[int, int] = (512, 512)

@dataclass
class Labels():
    """Container for the labeled peaks. Exists for reciprocal- and polar shape."""

    boxes: list = field(default_factory=list)
    split_boxes: list = field(default_factory=list)
    radii: list = field(default_factory=list)
    widths: list = field(default_factory=list)
    angles: list = field(default_factory=list)
    angles_std: list = field(default_factory=list)
    confidences: list = field(default_factory=list)
    intensities: list = field(default_factory=list)
    background_levels: list = field(default_factory=list)
    background_slopes: list = field(default_factory=list)
    is_ring: list = field(default_factory=list)
    img_nr: int = 0
    img_name: str = None
    peak_height: int = 0

    
    def get_num_low_conf(self):
        return np.count_nonzero(self.confidences == 0.1)
    
    def get_num_med_conf(self):
        return np.count_nonzero(self.confidences == 0.5)
    
    def get_num_high_conf(self):
        return np.count_nonzero(self.confidences == 1)

    def __len__(self):
        return len(self.boxes)

@dataclass
class ImageContainer:
    """Container for GIWAXS images. Contains the image in raw and polar coordinates."""
    config = None
    raw_reciprocal: np.ndarray = None
    raw_polar_image: np.ndarray = field(init=False)
    converted_polar_image: np.ndarray = field(init=False)
    split_polar_images: np.array = field(init=False)
    #image mask, necessary to calculate the linear profile
    converted_mask: np.ndarray = field(init=False)
    polar_img_shape: tuple = DEFAULT_POLAR_SHAPE
    q_z: float = None
    q_xy: float = None
    beam_center_x = None
    beam_center_y = None
    h5_group = None
    nr: int = None
    qzqxyboxes : np.array = None
    boxes: np.array = None
    scores: np.array = None
    radius: np.array = None
    radius_width: np.array = None
    angle: np.array = None
    angle_width = None
    reciprocal_labels: Labels = field(default_factory=Labels)
    polar_labels: Labels = field(default_factory=Labels)

    def __post_init__(self):
        if self.config is not None:
            self.polar_img_shape: tuple = self.config.PREPROCESSING_POLAR_SHAPE