"""Read-only loader for pyGID/NeXus structured GIWAXS files (labeled evaluation).

Ported from ``mlgidDETECT/mlgiddetect/dataloader/pygidloader.py`` (the ONNX write-back path
is intentionally dropped here: the DINO repo evaluates with the live PyTorch model, so only the
image-read + ground-truth-label half is needed).

A pyGID file stores, per top-level group, the reciprocal-space frames at ``<group>/data/img_gid_q``
and the ground-truth peaks at ``<group>/data/analysis/frameNNNNN/fitted_peaks``. This differs from
the older ``roi_data``-style files handled by ``H5GIWAXSDataset`` in ``util/labeleddataset.py``.

EXAMPLE USAGE:
    data = PyGIDDataset(config, preprocess_func=standard_preprocessing, buffer_size=5, load_labels=True)
    for i, img_container in enumerate(data.iter_images()):
        giwaxs_img = img_container.converted_polar_image
        labels = img_container.polar_labels
"""

import sys
sys.path.append("..")
from dataclasses import dataclass, field
from typing import Tuple, Iterator
import logging
import numpy as np
from h5py import File, Group
from multiprocessing import Process, Queue, Value
from .imgcontainer import ImageContainer, Labels

DEFAULT_POLAR_SHAPE: Tuple[int, int] = (512, 512)


def detect_dataset_type(path) -> str:
    """Auto-detect the labeled-dataset layout of an h5 file.

    Returns 'pygid' for pyGID/NeXus files (any top-level group exposing ``data/img_gid_q``,
    e.g. organic_labeled.h5), otherwise 'h5giwaxs' for the roi_data-style files. Defaulting to
    'h5giwaxs' keeps every existing file routing through H5GIWAXSDataset unchanged.
    """
    with File(path, 'r') as f:
        for key in f.keys():
            grp = f[key]
            if isinstance(grp, Group) and 'data' in grp:
                data = grp['data']
                if isinstance(data, Group) and 'img_gid_q' in data:
                    return 'pygid'
    return 'h5giwaxs'


@dataclass
class PyGIDDataset():
    """Container for a labeled pyGID dataset. Loads frames + ground-truth peaks in a separate
    process and stores them in image_queue. Iterate via ``iter_images()`` or directly:

        for img_container in PyGIDDataset(config, preprocess_func=..., load_labels=True):
            ...
    """

    config: 'Config' = None
    path: str = None
    preprocess_func: callable = None
    #min confidence of loaded boxes
    min_confidence: float = None
    #shape of the desired polar images
    polar_img_shape: tuple = DEFAULT_POLAR_SHAPE
    #load ground-truth peaks from data/analysis/frameXXXXX/fitted_peaks (labeled evaluation only)
    load_labels: bool = False

    load_worker: Process = None
    #buffer size of the worker. E.g. 5 are loaded in advance by default.
    buffer_size: int = 5
    image_queue: Queue = Queue()
    image_metrics: list = field(default_factory=list)

    def __post_init__(self):
        if self.config is not None:
            self.polar_img_shape: tuple = self.config.PREPROCESSING_POLAR_SHAPE
            if self.path is None:
                self.path = self.config.INPUT_DATASET
        self.load_worker_finished = Value('i', False)
        self.image_queue = Queue(self.buffer_size)
        self.load_worker = Process(target=load_worker, args=[self], daemon=True)
        self.load_worker.start()

    def iter_images(self) -> Iterator:
        """Iterator for images, usage: for img in data.iter_images():"""
        return iter(self.image_queue.get, None)

    def __iter__(self):
        return self

    def __next__(self):
        queue_object = self.image_queue.get()
        if queue_object is not None:
            return queue_object
        else:
            raise StopIteration

    def close(self):
        pass


def _load_fittedpeaks(f, key: str, frame_nr: int, data_loader: 'PyGIDDataset') -> Labels:
    """Read GT peaks from data/analysis/frameXXXXX/fitted_peaks and return a populated Labels object.

    The peak confidence is derived from the discrete ``visibility`` level (3->1.0, 2->0.5, 1->0.1)
    so it lands in the same {0.1, 0.5, 1.0} space the evaluator expects from H5GIWAXSDataset. The
    raw ``score`` field is used as a fallback for any other visibility value.

    Args:
        f: open h5py File handle (read mode)
        key: top-level H5 group key for the sample
        frame_nr: zero-based frame index
        data_loader: PyGIDDataset instance (provides config and min_confidence)

    Returns:
        Labels with .boxes (N x 4, polar pixel coords), .confidences, .is_ring and .visibility populated.
    """
    labels = Labels()
    frame_path = f'{key}/data/analysis/frame{str(frame_nr).zfill(5)}/fitted_peaks'

    if frame_path not in f:
        logging.debug('No fitted_peaks found at %s, returning empty labels.', frame_path)
        return labels

    peaks = f[frame_path][()]
    if len(peaks) == 0:
        return labels

    polar_shape = data_loader.polar_img_shape
    q_max = data_loader.config.GEO_QMAX

    radius = peaks['radius'].astype(np.float64)             # centre, q units
    radius_width = peaks['radius_width'].astype(np.float64) # full width, q units
    angle = peaks['angle'].astype(np.float64)              # centre, degrees (0-90)
    angle_width = peaks['angle_width'].astype(np.float64)  # full width, degrees
    is_ring = peaks['is_ring']
    visibility = peaks['visibility'].astype(np.int32)

    # Visibility level -> confidence in the {0.1, 0.5, 1.0} space used for metric stratification.
    confidences = np.select(
        [visibility == 3, visibility == 2, visibility == 1],
        [1.0, 0.5, 0.1],
        default=peaks['score'].astype(np.float32),
    ).astype(np.float32)

    # Convert to polar image pixel coordinates (q -> x on polar_shape[1], angle -> y on polar_shape[0]).
    radius_pixel = radius / q_max * polar_shape[1]
    radius_half_pixel = radius_width / q_max * polar_shape[1] / 2
    angle_pixel = angle * polar_shape[0] / 90
    angle_half_pixel = angle_width * polar_shape[0] / 90 / 2

    boxes = np.stack([
        radius_pixel - radius_half_pixel,  # x1
        angle_pixel - angle_half_pixel,    # y1
        radius_pixel + radius_half_pixel,  # x2
        angle_pixel + angle_half_pixel,    # y2
    ], axis=-1).astype(np.float32)

    # Exclude visibility=0 peaks (not reliably visible) from the ground truth.
    keep = visibility != 0
    if data_loader.min_confidence is not None:
        keep &= confidences >= data_loader.min_confidence
    boxes = boxes[keep]
    confidences = confidences[keep]
    is_ring = is_ring[keep]
    visibility = visibility[keep]

    labels.boxes = boxes
    labels.confidences = confidences
    labels.is_ring = list(is_ring)
    labels.visibility = visibility
    return labels


def load_worker(data_loader: PyGIDDataset):
    """Worker for reading the pyGID H5 file and converting frames to polar coordinates.
    Intended to be spawned as a separate process; enqueues ImageContainer objects.

    Args:
        data_loader (PyGIDDataset): dataset being loaded
    """
    with File(data_loader.path, 'r') as f:
        file_keys = [key for key in f.keys()]

    for counter, key in enumerate(file_keys):
        with File(data_loader.path, 'r') as f:
            try:
                img_nrs = range(len(f[key]['data/img_gid_q'][()]))
                data_loader.config.GEO_QMAX = np.sqrt(
                    (f[key]['data/q_z'][-1]) ** 2 + (f[key]['data/q_xy'][-1]) ** 2
                )
                data_loader.config.GEO_PIXELPERANGSTROEM = (
                    f[key]['data/img_gid_q'][0].shape[0] / f[key]['data/q_z'][-1]
                )
            except Exception:
                continue

        for i in img_nrs:
            img_container = ImageContainer(data_loader.config)
            img_container.config = data_loader.config
            with File(data_loader.path, 'r') as f:
                group = f[key]
                raw_img = group['data/img_gid_q'][i]
                img_container.h5_group = group.name
                img_container.q_z = group['data/q_z'][-1]
                img_container.q_xy = group['data/q_xy'][-1]
                img_container.nr = i
                img_container.raw_reciprocal = np.nan_to_num(raw_img)
                img_container.reciprocal_img_shape = img_container.raw_reciprocal.shape
                img_container.converted_polar_image, img_container.raw_polar_image, img_container.converted_mask = \
                    data_loader.preprocess_func(data_loader.config, img_container.raw_reciprocal, counter)
                if data_loader.load_labels:
                    img_container.polar_labels = _load_fittedpeaks(f, key, i, data_loader)
            data_loader.image_queue.put(img_container)
    data_loader.image_queue.put(None)
