# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import NamedTuple
from math import pi
import random

import numpy as np
from typing import Union
import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import nms, masks_to_boxes
import torchvision
from math import pi, sin, cos
from dataclasses import dataclass


""" from gixd_detectron.img_processing import (
    normalize,
    torch_he,
    with_probability
)

from gixd_detectron.noise import perlin

from gixd_detectron.simulations.angle_limits import AngleLimits
from gixd_detectron.simulations.misc import clamp_boxes """

HEIGHT = 512
WIDTH = 1024

def normalize(img: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    return (img - img.min()) / (img.max() - img.min())

def interp1d(x: Tensor, y: Tensor, x_new: Tensor) -> Tensor:
    eps = torch.finfo(y.dtype).eps
    ind = torch.searchsorted(x.contiguous(), x_new.contiguous())
    ind = torch.clamp(ind - 1, 0, x.shape[0] - 2)
    slopes = (y[1:] - y[:-1]) / (eps + (x[1:] - x[:-1]))
    return y[ind] + slopes[ind] * (x_new - x[ind])

@torch.no_grad()
def torch_he(img: Tensor, bins: int = 1000):
    """
    Histogram equalization implemented in pytorch
    :param img: target image
    :param bins: number of bins
    :return: image with corrected intensity
    """
    bin_edges = torch.linspace(img.min(), img.max(), bins + 1, device=img.device)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    img_flat = img.view(-1)
    hist = torch.histc(img_flat, bins=bins)
    cdf = torch.cumsum(hist, 0)
    cdf = cdf / cdf[-1]
    res = interp1d(bin_centers, cdf, img_flat)
    return res.view(img.shape)

def with_probability(probability: float = 1.):
    def wrapper(func):
        def new_func(img, *args, **kwargs):
            prob = kwargs.pop('prob') if 'prob' in kwargs else probability
            return func(img, *args, **kwargs) if random.random() < prob else img

        return new_func

    return wrapper

def perlin_octave(width, height, x_scale, y_scale, device='cuda'):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, x_scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, y_scale + 1)[None, :-1].to(device)

    wx = 1 - interp(xs)
    wy = 1 - interp(ys)

    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))

    return dots.permute(0, 2, 1, 3).contiguous().view(width * y_scale, height * x_scale)


def interp(t):
    return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

def perlin(octave_rates: tuple = (1, 2, 3, 4),
           weights: tuple = None,
           amp: float = 1., size: int = 512, device='cuda'):
    weights = weights or [1] * len(octave_rates)

    p = 0

    for rate, weight in zip(octave_rates, weights):
        octave = 2 ** rate
        p += perlin_octave(octave, octave, WIDTH//octave, HEIGHT//octave,device=device) * weight
    return ((p - p.min()) / (p.max() - p.min()) - 0.5) * amp + 1

class AngleLimits(object):
    def __init__(self,
                 slope_range: tuple = (0, 0.1),
                 size_ratio_range: tuple = (-0.15, .15),
                 r_size: int = 512,
                 phi_size: int = 512
                 ):
        self.r_size = r_size
        self.phi_size = phi_size
        self.slope_range = slope_range
        self.size_ratio_range = size_ratio_range
        self.slope, self.size_ratio = None, None
        self._x_size, self._y_size = None, None
        self._quazipolar: bool = False
        self._quazipolar_coef: float = 1.
        self._quazipolar_range: tuple = (0.1, 1.2)
        self.update_params()

    def max(self, r: Tensor) -> Tensor:
        return (
                       (r <= self._y_size) +
                       (r > self._y_size) *
                       torch.nan_to_num(torch.arcsin(self._y_size / r)) / pi * 2
               ) * self.phi_size

    def min(self, r: Tensor) -> Tensor:
        dark_area = r * self.slope
        geometry_area = (r > self._x_size) * torch.nan_to_num(torch.arccos(self._x_size / r)) / pi * 2 * self.phi_size
        min_angles = torch.maximum(geometry_area, dark_area)
        min_angles = geometry_area
        if self._quazipolar:
            min_angles = torch.maximum(min_angles, r * self._quazipolar_coef - 150)
        return min_angles

    def update_params(self):
        self.slope = random.uniform(*self.slope_range)
        self.size_ratio = random.uniform(*self.size_ratio_range)
        #self.size_ratio = -.15
        x_weight = pi / 4 - self.size_ratio

        self._x_size = self.r_size * sin(x_weight)
        self._y_size = self.phi_size * cos(x_weight)
        self._quazipolar =  bool(random.random() <= 0.3)
        self._quazipolar = False
        self._quazipolar_coef = random.uniform(*self._quazipolar_range)

@dataclass
class SimulationConfig():
    
    obj_num: tuple = (2, 200)
    width_central: tuple = (1., 5.)
    ring_width_central: tuple = (2., 15.)
    widths_std: float = 150
    pos: tuple = (.048*WIDTH, .98*WIDTH)
    a_pos: tuple = (0, 1.1*HEIGHT)
    a_seg_widths_central: tuple = (.1, 10.)
    a_seg_widths_std: float = 50
    a_ring_widths_central: tuple = (50, 10000)
    a_ring_widths_std: float = 50
    ring_intensity_range: tuple = (2, 50)
    seg_intensity_range: tuple = (10, 50)
    min_angle: float = 1.0
    min_linear_dark_area = 10
    max_linear_dark_area = 0.96*WIDTH
    min_nms: float = 0.001
    min_ring_seg_nms: float = 0.0
    p_ps_noise: float = 0.005
    poisson_range: tuple = (50, .78*WIDTH)
    a_coef: float = 3.5
    w_coef: float = 1#1.5
    add_hot_pixels: bool = False
    hot_pixels_range: tuple = (-2., 3.)
    hot_pixels_p: float = 0.001
    hot_pixels_prob: float = 0.2
    prob_single_obj: float = 0.1


class FastSimulation(object):
    def __init__(self, sim_config: SimulationConfig = None, device: torch.device = 'cuda'):

        self.background_img = None
        self.device = device
        self.x = torch.arange(WIDTH, device=device)[None, :, None]
        self.y = torch.arange(HEIGHT, device=device)[:, None, None]
        self.sim_config = sim_config or SimulationConfig()

        self.mask_coords = torch.flip((self.x * torch.cos(self.y / 1024 * pi / 2)).squeeze(), (0,))
        self.angle_limits = AngleLimits()

        self.kernel1 = torch.tensor(_SMOOTH_KERNEL, device=device).view(1, 1, 3, 3)

        self.quazipolar_dark_area = True
        self.polar_dark_area = True
        self.linear_dark_area = True
        self.linear_y_max = 10
        self.quazipolar_coef = 1.54

        #coords for black detector masks
        self.rs = 0
        self.ws = 0

    @torch.no_grad()
    def simulate_img(self, background_img = None):

        self.background_img = background_img

        if self.background_img is not None:
            self.sim_config.ring_intensity_range: tuple = (45, 50)
            self.sim_config.seg_intensity_range: tuple = (48, 50)
        else:
            self.sim_config.ring_intensity_range: tuple = (2, 50)
            self.sim_config.seg_intensity_range: tuple = (10, 50)

        global WIDTH
        if random.random() < 0.5:
            if WIDTH == 1024:
                WIDTH = 1024
            else:
                WIDTH = 1024
            self.__init__()

        self.detector_mask = False
        self.create_detector_mask()

        boxes, intensities, is_ring = self.simulate_labels()

        if random.random() <= self.sim_config.prob_single_obj:
            boxes = boxes[:1]
            intensities = intensities[:1]
            is_ring = is_ring[:1]

        if len(boxes) == 0:
            brekpoi = 0


        # create an image with 2D-Gaussian peaks
        img = self.img_from_labels(boxes, intensities, is_ring)
        if img.min() == img.max() or torch.any(img.isnan()):
            return self.simulate_img()

        img = mul_perlin(img)

        # add background
        #img = background_perlin(img)
     
        if self.background_img is None:
            img = add_glass(img, self.x, self.y)
            img = add_linear_background(img)

        # add noise
            img = apply_poisson_noise(img, self.sim_config.poisson_range)
        #img = add_perlin_noise(img)
        # img = apply_speckle_noise(img)
        #img = apply_stretch(img)
        if self.polar_dark_area:
            img = apply_stretch(img, (int(.04*WIDTH), int(.1*WIDTH)), (7, 10))

        # sometimes the image is empty, and we generate it again
        if img.min() == img.max():
            return self.simulate_img()

        #import matplotlib.pyplot as plt
        #fig, ax = plt.subplots()
        #ax.bar(x=torch.histogram(img)[1][:-1], height=torch.histogram(img)[0])
        #fig.savefig('/home/constantin/git_repos/DINO/hist.png')

        # add masks
        img, mask = self.add_dark_area(img, boxes)
        img, mask = self.apply_detector_gaps(img, mask)

        # deteriorate contrast
        if self.sim_config.add_hot_pixels:
            img = add_hot_pixels(
                img,
                self.sim_config.hot_pixels_p,
                self.sim_config.hot_pixels_range,
                prob=self.sim_config.hot_pixels_prob,
            )

        # add salt & pepper noise
        img = apply_salt_pepper_noise(img, self.sim_config.p_ps_noise)


        
        
        clahe_img = img
        # apply kernels & contrast correction
        clahe_img = apply_log(clahe_img)
        clahe_img = apply_he(clahe_img)
        clahe_img = apply_clip_img(clahe_img)
        clahe_img = apply_kernel(clahe_img, self.kernel1)
        clahe_img = digitalize_img(clahe_img)

        clahe_img = normalize(clahe_img)

        if self.background_img is not None:
            clahe_img = clahe_img + self.background_img
            clahe_img = normalize(clahe_img)
            boxes = torch.cat([boxes, Tensor([[116,0,128,512]]).cuda()])

        clahe_img, boxes, mask = flip_image(clahe_img, boxes, mask)        

        return clahe_img, boxes, mask

    @torch.no_grad()
    def simulate_boxes(self):
        boxes, intensities = self.simulate_labels()
        return boxes

    @torch.no_grad()
    def simulate_labels(self):
        self.angle_limits.update_params()

        sc = self.sim_config

        rings_or_seg_or_both = random.random()



        def simulate_and_process(obj_num, pos_c, width_c, width_std, a_pos_c, a_width_c, a_width_std, intensity_range, is_segment=False):
            pos, widths, a_pos, a_widths = simulate_labels(
                obj_num, sc.pos, width_c, width_std,
                sc.a_pos, a_width_c, a_width_std, self.device
            )
            if is_segment:
                a_widths = torch.maximum(a_widths, widths * (torch.rand_like(widths) + 1.0))
            pos, widths, a_pos, a_widths, _ = filter_nms(pos, widths, a_pos, a_widths, sc.min_nms)
            intensities = gen_intensities(pos, widths, a_pos, a_widths, intensity_range)
            return pos, widths, a_pos, a_widths, intensities

        ring_pos = ring_widths = ring_a_pos = ring_a_widths = ring_intensities = torch.empty(0, device=self.device)
        seg_pos = seg_widths = seg_a_pos = seg_a_widths = seg_intensities = torch.empty(0, device=self.device)

        if rings_or_seg_or_both < 1/3:
            ring_pos, ring_widths, ring_a_pos, ring_a_widths, ring_intensities = simulate_and_process(
                sc.obj_num, sc.ring_width_central, sc.ring_width_central, sc.widths_std,
                sc.a_ring_widths_central, sc.a_ring_widths_central, sc.a_ring_widths_std,
                sc.ring_intensity_range
            )
        elif rings_or_seg_or_both < 2/3:
            seg_pos, seg_widths, seg_a_pos, seg_a_widths, seg_intensities = simulate_and_process(
                sc.obj_num, sc.width_central, sc.width_central, sc.widths_std,
                sc.a_seg_widths_central, sc.a_seg_widths_central, sc.a_seg_widths_std,
                sc.seg_intensity_range, is_segment=True
            )
        else:
            ring_pos, ring_widths, ring_a_pos, ring_a_widths, ring_intensities = simulate_and_process(
                sc.obj_num, sc.ring_width_central, sc.ring_width_central, sc.widths_std,
                sc.a_ring_widths_central, sc.a_ring_widths_central, sc.a_ring_widths_std,
                sc.ring_intensity_range
            )
            seg_pos, seg_widths, seg_a_pos, seg_a_widths, seg_intensities = simulate_and_process(
                sc.obj_num, sc.width_central, sc.width_central, sc.widths_std,
                sc.a_seg_widths_central, sc.a_seg_widths_central, sc.a_seg_widths_std,
                sc.seg_intensity_range, is_segment=True
            )

        pos      = torch.cat([ring_pos, seg_pos])
        widths   = torch.cat([ring_widths, seg_widths])
        a_pos    = torch.cat([ring_a_pos, seg_a_pos])
        a_widths = torch.cat([ring_a_widths, seg_a_widths])
        intensities = torch.cat([ring_intensities, seg_intensities])

        is_ring= torch.cat([torch.ones(ring_pos.size()[0], dtype=torch.bool, device=self.device), torch.zeros(seg_pos.size()[0], dtype=torch.bool, device=self.device)])


        pos, widths, a_pos, a_widths, indices = filter_nms(
            pos, widths, a_pos, a_widths, is_ring , self.sim_config.min_ring_seg_nms
        )

        is_ring = is_ring[indices]
        intensities = intensities[indices]

        boxes = self._boxes_from_positions(pos, widths, a_pos, a_widths)
        #boxes = clamp_boxes(boxes)

        #filter peaks in detector gaps
    
        boxes_peaks = boxes[torch.logical_not(is_ring)]
        peaks_not_in_gap = self.filter_peaks_detector_gap(boxes_peaks)
        indices_detector_gap = torch.cat([torch.ones(torch.count_nonzero(is_ring), dtype=torch.bool, device=self.device),peaks_not_in_gap])
        boxes, indices_dark_area = self.filter_dark_area(pos, boxes)

        indices = indices_detector_gap & indices_dark_area
        
        boxes = boxes[indices]
        intensities = intensities[indices]
        pos = pos[indices]
        widths = widths[indices]
        a_pos = a_pos[indices]
        a_widths = a_widths[indices]
        is_ring = is_ring[indices]

        #add peaks on rings
        pos, widths, a_pos, a_widths, intensities_peaks =   self.add_peaks_on_rings(pos[is_ring], widths[is_ring], boxes[is_ring], intensities[is_ring])
        if pos is not None:
            boxes_peaks_on_rings = self._boxes_from_positions(pos, widths, a_pos, a_widths)
            idx_in_img = self.filter_peaks_detector_gap(boxes_peaks_on_rings)
            boxes_peaks_on_rings = boxes_peaks_on_rings[idx_in_img]
            intensities_peaks = intensities_peaks[idx_in_img]
            boxes = torch.cat([boxes, boxes_peaks_on_rings])
            intensities = torch.cat([intensities, intensities_peaks])
            is_ring = torch.cat([is_ring, torch.zeros((len(intensities_peaks)), device=self.device)])
        
        if not boxes.shape[0]:
            return self.simulate_labels()
        
        intensities = intensities[(boxes[:, 1] < boxes[:,3]) & (boxes[:, 0] < boxes[:,2])]
        is_ring = is_ring[(boxes[:, 1] < boxes[:,3]) & (boxes[:, 0] < boxes[:,2])].bool()        
        boxes = boxes[(boxes[:, 1] < boxes[:,3]) & (boxes[:, 0] < boxes[:,2])]

        clamp_boxes(boxes)

        



        # if random.random() <= 1:
        #     total_num = intensities.shape[0]
        #     if total_num <= 5:
        #         return
        #     bright_peaks = np.random.choice(np.arange(total_num), 4, replace=False)
        #     intensities[bright_peaks] *= 100



        return boxes, intensities, is_ring

    def _boxes_from_positions(self, pos, widths, a_pos, a_widths):
        boxes = torch.stack([pos - widths * self.sim_config.w_coef,
                             a_pos - a_widths * self.sim_config.a_coef,
                             pos + widths * self.sim_config.w_coef,
                             a_pos + a_widths * self.sim_config.a_coef], 1)
        return boxes
        
    

    def add_peaks_on_rings(self, x_position, widths, boxes, ring_intensities):
        #no peaks on rings
        if random.random() > .1:
            return None, None, None, None, None
        
        else:
            try:
                max_a_width = (boxes[:,3] - boxes[:,1])/2
                a_pos = (boxes[:,3] + boxes[:,1]) /2 - max_a_width

                #determine how many peaks on each ring
                num_peaks_one_dim = torch.tensor(np.random.randint(0,4, size=len(x_position)), device=self.device)
                num_peaks = torch.cat([torch.Tensor(num_peaks_one_dim == 0), torch.Tensor(num_peaks_one_dim > 0),torch.Tensor(num_peaks_one_dim > 1),torch.Tensor(num_peaks_one_dim > 2)]).reshape(4,-1)
                #no peaks on to small rings
                no_peaks = max_a_width < 100
                num_peaks[:,no_peaks] = 0

                #calculate variable azimuthal position of peaks
                rand_offsets = torch.randint(1, 100,(4,len(x_position)),device=self.device)
                sum_rand_offsets = torch.sum(rand_offsets, dim=0).to(device=self.device)
                percentage_offset = (rand_offsets / sum_rand_offsets)*num_peaks
                a_space_to_divide = max_a_width*2 * ((5-num_peaks_one_dim)/6)
                a_var_offsets = a_space_to_divide * percentage_offset
                a_var_offsets[1] = a_var_offsets[1] + a_var_offsets[0]
                a_var_offsets[2] = a_var_offsets[2] + a_var_offsets[1]
                a_var_offsets[3] = a_var_offsets[3] + a_var_offsets[2]

                a_widths = torch.randint(int(self.sim_config.a_seg_widths_central[0])+1, int(self.sim_config.a_seg_widths_central[1])+1,(4,len(x_position)),device=self.device)* num_peaks
                
                #when ring is very bright, peaks should be wider and not brighter than the ring
                ring_ishigh = torch.logical_or(ring_intensities > .5*torch.max(ring_intensities), ring_intensities > self.sim_config.ring_intensity_range[1])
                #widths[ring_ishigh] = widths[ring_ishigh] * (1 + .5+random.random()*.5)
                widths = widths*(1 + .5+random.random()*.5)
                #ring_intensities[is_high] = 0
                widths = widths.repeat(4).reshape(4,-1) * num_peaks

                ring_intensities = ring_intensities.repeat(4).reshape(4,-1) * num_peaks
                ring_ishigh = ring_ishigh.repeat(4).reshape(4,-1) * num_peaks

                a_fixed_offsets = torch.cat([torch.zeros((1,len(x_position)),device=self.device),torch.full((1,len(x_position)),1/6,device=self.device)*max_a_width*2,torch.full((1,len(x_position)),2/6,device=self.device)*max_a_width*2,torch.full((1,len(x_position)),3/6,device=self.device)*max_a_width*2])
                a_offsets = (a_fixed_offsets + a_var_offsets)* num_peaks
                a_positions = (a_offsets + a_pos.repeat(4).reshape(4,-1))*num_peaks
                max_a_width = max_a_width.repeat(4).reshape(4,-1) * num_peaks
                a_widths = max_a_width/40 * (torch.rand((4,len(boxes)), device=self.device) +1)
                x_positions = x_position.repeat(4).reshape(4,-1) * num_peaks
                positions = torch.stack((x_positions, a_positions, widths, a_widths, ring_intensities, ring_ishigh)).reshape(6,-1)
                positions = positions[:,positions[0]!=0].reshape(6,-1)
                peak_intensities = gen_intensities(positions[0],positions[2],positions[1],positions[3],tuple(ti for ti in self.sim_config.a_seg_widths_central))

                #if ring is high, peak intensity is the one of the ring
                peak_intensities = positions[4] + positions[5]*peak_intensities


                return positions[0], positions[2], positions[1], positions[3], peak_intensities
            except:
                return None, None, None, None, None
            #pos, widths, a_pos, a_widths, intensities_peaks

    
    def add_dark_area(self, img, boxes):

        def calculate_angle_limits_mask():
            dark_area_idx = (self.y <= self.angle_limits.min(self.x)) | (self.y >= self.angle_limits.max(self.x))
            level = random.uniform(-0.1, 0.5)
            #shift y_indices to the right if image different from 512 width            
            idx_dark_area = torch.where(dark_area_idx)
            y_shifted = (idx_dark_area[1] * (1 + (WIDTH-512)/512)).clamp(min=0, max=WIDTH-1).long()
            y_shifted1 = (y_shifted + 1).clamp(min=0, max=WIDTH-1).long()
            img_mask = torch.zeros((HEIGHT,WIDTH), device=self.device).bool()
            img_mask[idx_dark_area[0], y_shifted] = True
            img_mask[idx_dark_area[0], y_shifted1] = True
            return img_mask, level

        if self.polar_dark_area:
            mask_angle_limits, level = calculate_angle_limits_mask()
            img = normalize(img)
            img[mask_angle_limits > 0] = level
            return img, ~mask_angle_limits
        
        if self.quazipolar_dark_area:
            mask = torch.zeros_like(img, device=self.device)
            dark_area_idx = self.y > self.quazipolar_coef * (1 - (WIDTH - 512)/1024) * self.x
            mask[dark_area_idx.squeeze()] = 1            
            mask_angle_limits, level = calculate_angle_limits_mask()
            img = normalize(img)
            mask = torch.logical_or(mask_angle_limits, dark_area_idx.squeeze()).bool()
            img[mask > 0] = level
            mask = torch.logical_or(mask,self.idx_black)
            return normalize(img), ~mask

        if self.linear_dark_area:
            mask = torch.zeros_like(img, device=self.device)
            mask[ self.linear_y_max:, :] = 1
            return normalize(img * (1 - mask)), ~(mask.bool())
        
        else:
            return normalize(img),  torch.ones_like(img, device=self.device).bool()

        side = random.choice(["top", "bottom", "left", "right"])
    
        # Randomly choose a position along the selected side
        if side in ["top", "bottom"]:
            position = random.randint(0, img.shape[0] - 1)
        else:
            position = random.randint(0, img.shape[1] - 1)
        
        # Create a mask for the dark area
        mask = torch.zeros_like(img)
        if side == "top":
            mask[ :position, :] = 1
        elif side == "bottom":
            mask[ position:, :] = 1
        elif side == "left":
            mask[:, :position] = 1
        else:
            mask[:, position:] = 1
        
        # Apply a random rotation to the mask
        rotation_angle = random.uniform(-30, 30)  # Adjust the range as needed
        mask = mask[None, :,:]
        rotated_mask = TF.rotate(mask, angle=rotation_angle)
        rotated_mask = rotated_mask[0]




        
        mask_flat = rotated_mask.flatten()

        # Apply the dark area to the input tensor
        img = img * (1 - rotated_mask)
        h = 512
        w = 512

        rotated_mask = torch.ones(512,512,device='cuda')
        #boxes_as_masks = torch.zeros(h,w)
        boxes_as_masks = boxes_to_masks(boxes)
        boxes_and_masks =  boxes_as_masks * (1 -rotated_mask)
        boxes_to_keep = []
        for mask in boxes_and_masks:
            boxes_to_keep.append(masks_to_boxes(mask[None, :,:]))
        
        img = torchvision.utils.draw_bounding_boxes((img*255).to(torch.uint8)[None, :,:],torch.concat(boxes_to_keep))
            #intersecting_boxes = masks_to_boxes(boxes_and_masks)

        boxes_and_masks = torch.stack([boxes_as_masks,rotated_mask], dim=0)
        intersection_mask = torch.any(boxes_and_masks, dim=0)
        intersection_mask = intersection_mask[None, :,:]
        boxes_to_keep = masks_to_boxes(intersection_mask)
        torchvision.utils.save_image(img.float(),fp='/home/constantin/git_repos/object_detection/outputs/simulation/2.png')





        y_indices, x_indices = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        indices = y_indices * w + x_indices
        indices_flat = indices.flatten()

        box_indices = (boxes[:, None, 1] <= y_indices) & (y_indices < boxes[:, None, 3]) & \
              (boxes[:, None, 0] <= x_indices) & (x_indices < boxes[:, None, 2])
        

        box_within_mask = torch.all(mask_flat[indices_flat[box_indices]], dim=1)





        #level = random.uniform(-0.1, 0.5)

        #img[dark_area_idx.squeeze()] = level
        return img, rotated_mask
    
    

    def filter_dark_area(self, pos, boxes):

        #remove too small boxes and the ones outside the image
        boxes = clamp_boxes(boxes)
        widths = boxes[:, 3] - boxes[:, 1]
        widths_big_enough = widths > 1.6
        a_widths = boxes[:, 2] - boxes[:, 0]
        a_widths_big_enough = a_widths > 1.6
        indices_outside_image = widths_big_enough & a_widths_big_enough

        random_nr = random.random()
        #return without polar dark areas
        if random_nr > 0:
            self.polar_dark_area = True
            self.linear_dark_area = False
            self.quazipolar_dark_area = False
            pos = pos / (1 + (WIDTH-512)/512)

            

            angles = (boxes[:, 3] + boxes[:, 1]) / 2
            boxes[:, 3] = torch.minimum(boxes[:, 3], self.angle_limits.max(pos))
            boxes[:, 1] = torch.maximum(boxes[:, 1], self.angle_limits.min(pos))

            widths = boxes[:, 3] - boxes[:, 1]
            min_angle = self.sim_config.min_angle
            polar_indices = (widths >= min_angle) & (angles - boxes[:, 1] > - widths / 2) & (angles < boxes[:, 3])

            if random_nr > .5 and self.background_img is None:
                #remove boxes in quazipolar region
                self.polar_dark_area = False
                self.linear_dark_area = False
                self.quazipolar_dark_area = True
                self.quazipolar_coef = 1.54 + (-.2 + .4*random.random())
                #if rings reach into the quazipolar area, clamp them to the allowed area
                quazipolar_indices = (boxes[:, 3] >= self.quazipolar_coef * (1 - (WIDTH - 512)/1024) * boxes[:, 0]) & (boxes[:, 0] < (1/self.quazipolar_coef *(WIDTH)))
                boxes[quazipolar_indices, 3] =  self.quazipolar_coef * (1 - (WIDTH - 512)/1024) * boxes[quazipolar_indices, 0]
                indices = indices_outside_image & polar_indices

                return boxes, indices
            else:
                return boxes, polar_indices
     
        #return all boxes
        elif random_nr > 1:
            self.polar_dark_area = False
            self.linear_dark_area = False
            self.quazipolar_dark_area = False
            return boxes, boxes[indices_outside_image], indices_outside_image
        
        #remove boxes starting from the bottom
        else:
            self.polar_dark_area = False
            self.linear_dark_area = True
            self.quazipolar_dark_area = False
            self.linear_y_max = random.randint(int(self.sim_config.min_linear_dark_area),int(self.sim_config.max_linear_dark_area))
            indices = boxes[:,3] < self.linear_y_max
            indices_outside_image = indices & indices_outside_image

            return boxes[indices_outside_image], indices_outside_image



    def img_from_labels(self, boxes, intensities, is_ring):

        power = 2 if random.random() > 0.1 else 4
        x, y = self.x, self.y

        pos_peaks = (boxes[~is_ring, 0] + boxes[~is_ring, 2]) / 2
        a_pos_peaks = (boxes[~is_ring, 1] + boxes[~is_ring, 3]) / 2
        widths_peaks = (boxes[~is_ring, 2] - boxes[~is_ring, 0]) / self.sim_config.w_coef
        a_widths_peaks = (boxes[~is_ring, 3] - boxes[~is_ring, 1]) / self.sim_config.a_coef

        img =  (intensities[~is_ring][None, None] * (
            torch.exp(
                - torch.abs(x - pos_peaks[None, None]) ** power / widths_peaks[None, None] ** power / 2
                - (y - a_pos_peaks[None, None]) ** 2 / a_widths_peaks[None, None] ** 2 / 2
            )
        )).sum(-1)


        pos_rings = (boxes[is_ring, 0] + boxes[is_ring, 2]) / 2
        a_pos_rings = (boxes[is_ring, 1] + boxes[is_ring, 3]) / 2
        widths_rings = (boxes[is_ring, 2] - boxes[is_ring, 0]) / self.sim_config.w_coef
        a_widths_rings = torch.full((torch.count_nonzero(is_ring),),1000, device=self.device)

        img =  img + (intensities[is_ring][None, None] * (
            torch.exp(
                - torch.abs(x - pos_rings[None, None]) ** power / widths_rings[None, None] ** power / 2
                - (y - a_pos_rings[None, None]) ** 2 / a_widths_rings[None, None] ** 2 / 2
            )
        )).sum(-1)



    
        return img
    
    def create_detector_mask(self):
        self.detector_mask = True
        n = 2
        self.rs = np.random.uniform(80, 380, n)
        self.ws = np.random.uniform(1, 7, n)

        if n == 2 and abs(self.rs[1] - self.rs[0]) < 100:
            self.rs, self.ws = self.rs[:1], self.ws[:1]

        self.idx_black = torch.zeros(size=(HEIGHT, WIDTH),dtype=torch.bool, device=self.device)

        for r, w in zip(self.rs, self.ws):
            self.idx_black[(self.mask_coords <= (r + w)) & (self.mask_coords >= (r - w))] = 1

    def filter_peaks_detector_gap(self, boxes_peaks_on_rings):
        if self.detector_mask:
            boxes_as_masks = self.boxes_to_masks(boxes_peaks_on_rings)        
            return torch.logical_not(torch.any(self.idx_black & boxes_as_masks, dim=(1,2)))
        return torch.ones(size=(len(boxes_peaks_on_rings),), dtype=torch.bool ,device=self.device)


    def apply_detector_gaps(self, img, mask):
        if self.detector_mask:
            img[self.idx_black] = 0
            mask[self.idx_black] = 0
            return img, mask
        return img, mask
    
    def boxes_to_masks(self, boxes):
        """
        Convert bounding boxes to masks without using a for loop.
        
        Args:
            boxes (Tensor): Bounding boxes, shape (N, 4) where N is the number of boxes.
                            Each box is represented as (x1, y1, x2, y2).
            image_size (tuple): Size of the image (height, width).
        
        Returns:
            Tensor: Masks, shape (N, height, width).
        """
        N = boxes.shape[0]
        masks = torch.zeros((N, HEIGHT, WIDTH), dtype=torch.uint8, device=self.device)
        
        x1, y1, x2, y2 = boxes.unbind(1)
        
        # Create a grid of coordinates
        y = torch.arange(HEIGHT, dtype=torch.int64, device=self.device).view(1, HEIGHT, 1)
        x = torch.arange(WIDTH, dtype=torch.int64, device=self.device).view(1, 1, WIDTH)
        
        # Create masks using broadcasting
        masks = ((y >= y1.view(N, 1, 1)) & (y < y2.view(N, 1, 1)) &
                (x >= x1.view(N, 1, 1)) & (x < x2.view(N, 1, 1))).to(torch.uint8)
        
        return masks
    
@with_probability(0.5)
def apply_kernel(img, kernel):
    return F.conv2d(img[None, None], kernel, padding=1).squeeze()
    
def simulate_labels(
        obj_num,
        pos, width_central_range, widths_std,
        a_pos, a_widths_central_range, a_widths_std, device = 'cuda'
):
    lower_b, upper_b = obj_num
    n = int(max(lower_b, min(random.gauss(lower_b + 0.75 * (upper_b - lower_b), (upper_b - lower_b) / 1), upper_b)))

    width_central = random.uniform(*width_central_range)
    width_central = random.paretovariate(3)
    #clip to 1-3
    #randomly create width in range with a pareto distribution
    width_central = width_central_range[0] + (min(random.paretovariate(5),3)-1)/2 * (width_central_range[1] - width_central_range[0])

    pos = torch_uniform(*pos, n, device=device)

    widths = torch.poisson(
        torch.tensor([width_central] * n, device=device) * widths_std
    ) / widths_std

    widths = torch.clamp_(widths, width_central_range[0] / 2)

    a_pos = torch_uniform(*a_pos, n, device=device)

    a_widths_central = random.uniform(*a_widths_central_range)

    a_widths = torch.poisson(
        torch.tensor([a_widths_central] * n, device=device) * a_widths_std
    ) / a_widths_std

    a_widths = torch.clamp_(a_widths, a_widths_central_range[0] / 2)

    return pos, widths, a_pos, a_widths
    

def clamp_boxes(boxes):
    torch.clamp_(boxes[:, 0], min=1, max=WIDTH)
    torch.clamp_(boxes[:, 1], min=1, max=HEIGHT)
    torch.clamp_(boxes[:, 2], min=1, max=WIDTH)
    torch.clamp_(boxes[:, 3], min=1, max=HEIGHT)
    return boxes


""" def shift_coords(x, y, boxes, rotate_range):
    rotate_coef = random.uniform(*rotate_range)
    x = x - (y / y.max() - 0.5) * rotate_coef
    a_pos = (boxes[:, 1] + boxes[:, 3]) / 2
    y_shifts = (a_pos / y.max() - 0.5) * rotate_coef
    boxes[:, ::2] = boxes[:, ::2] + y_shifts[:, None]
    return x, y, boxes """





@with_probability(1)
def apply_he(img):
    return torch_he(img)


@with_probability(0.9)
def apply_log(img, coef_range: tuple = (50, 5000)):
    coef = random.uniform(*coef_range)
    return normalize(torch.log10(normalize(img) * coef + 1.))


@with_probability(0.05)
def apply_clip_img(img):
    m, s = img.mean().item(), img.std().item() * random.uniform(2, 4)
    return torch.clamp_(img, m - s, m + s)


@with_probability(0.9)
def mul_perlin(img):
    rates = (3, 4, 5)  # if random.random() > 0.3 else (3, 4, 5)
    weights = tuple(np.random.uniform(1, 3, len(rates)))
    p_weight = random.uniform(0.1, 0.5)
    return img * (p_weight * perlin(rates, weights, amp=1, device=img.device) + 1 - p_weight)


def _bernoulli(p, shape, device):
    return torch.bernoulli(torch.empty(*shape, device=device).uniform_(0, 2 * p)) == 1


@with_probability(0.2)
def background_perlin(img):
    background = perlin((3, 4, 5), device=img.device)

    return img + background / background.max() * img.max() * 0.1


@with_probability(0.5)
def add_perlin_noise(img):
    rates = (4, 5, 6)
    weights = tuple(np.random.uniform(1, 3, len(rates)))
    perlin_img = perlin(rates, weights, amp=1, device=img.device)
    perlin_img = torch.poisson(perlin_img / perlin_img.max() * 2)
    weight = random.uniform(0.05, 0.2)
    return img + perlin_img / perlin_img.max() * img.max() * weight


@with_probability(0.5)
def apply_salt_pepper_noise(img, p: float = 0.05):
    num = int(np.prod(img.shape) * p)
    y = torch.randint(HEIGHT, (num,))
    x = torch.randint(WIDTH, (num,))

    img[y[:num // 2], x[:num // 2]] = img.min()
    img[y[num // 2:], x[num // 2:]] = img.max()

    return img


@with_probability(0.1)
def add_hot_pixels(
        img: Tensor,
        p: float = 0.001,
        intensity_range: tuple = (-1, 2.)
):
    num = int(np.prod(img.shape) * p)
    y = torch.randint(HEIGHT, (num,))
    x = torch.randint(WIDTH, (num,))
    intensity = torch.rand(num, device=img.device, dtype=img.dtype) * (
            intensity_range[1] - intensity_range[0]) + intensity_range[0]

    img = normalize(img)
    img[y, x] = img.max() * intensity

    return normalize(img)


@with_probability(0.5)
def apply_speckle_noise(img):
    var = random.uniform(0.1, 0.25)
    noise = torch.normal(0, var, img.shape, device=img.device)
    return img + img * noise


@with_probability(1)
def apply_poisson_noise(img, poisson_range):
    coef = random.uniform(*poisson_range)
    img.min()
    normalize(img)
    return torch.poisson(coef * normalize(img))


@with_probability(1)
def apply_poisson_noise_not_normalized(img):
    coef = random.random() * 50 + 50
    m = img.max()
    return torch.poisson(coef * normalize(img)) / coef * m


@with_probability(0.4)
def digitalize_img(img):
    channels = random.randint(16, 64)
    return (normalize(img) * channels).round()


@with_probability(0.5)
def add_glass(img, x, y, pos_range: tuple = (40, 300)):
    power = 2 if random.random() > 0.5 else 1

    r = random.uniform(*pos_range)
    w = random.uniform(50, 140)
    a = random.uniform(50, 450)
    aw = random.uniform(250, 1050)
    weight = random.uniform(0.5, 1.2)

    if power == 1:
        weight *= 2
        w *= 2

    gauss = torch.exp(- torch.abs(x - r) ** power / 2 / w ** power - (y - a) ** 2 / 2 / aw ** 2).squeeze()
    return normalize(img) + gauss * weight


@with_probability(1)
def add_glass_not_normalized(img, x, y, pos_range: tuple = (40, 300)):
    power = 2 if random.random() > 0.5 else 1

    r = random.uniform(*pos_range)
    w = random.uniform(50, 140)
    a = random.uniform(50, 450)
    aw = random.uniform(250, 1050)
    weight = random.uniform(0.5, 1.2)

    if power == 1:
        weight *= 2
        w *= 2

    gauss = torch.exp(- torch.abs(x - r) ** power / 2 / w ** power - (y - a) ** 2 / 2 / aw ** 2).squeeze()
    return img + gauss * weight * img.max()


@with_probability(0.9)
def add_linear_background(img):
    start, end = np.random.uniform(0, 0.1, 2)
    #dark_area = torch.zeros(HEIGHT, WIDTH, device=img.device)
    #dark_area[:int(np.clip(HEIGHT*random.random()*1.5, 0, HEIGHT)), :] = 1
    noise = torch.linspace(start, end, WIDTH, device=img.device)[None].repeat(HEIGHT, 1)# *dark_area
    return normalize(img) + noise



@with_probability(1)
def add_linear_background_no_normalization(img):
    start, end = np.random.uniform(0, 0.1, 2)
    return img + torch.linspace(start, end, WIDTH, device=img.device)[None].repeat(HEIGHT, 1) * img.max()


#@with_probability(0.8)
def apply_stretch(img, x_range: tuple = (50, 150), step_range: tuple = (3, 6)):
    x_max = random.randint(*x_range)
    step = random.randint(*step_range)
    stretched = torch.nn.functional.interpolate(
        img[::step, :x_max][None, None],
        (512, x_max)
    ).squeeze()
    img[:, :x_max] = stretched
    return img


def gen_intensities(pos, widths, a_pos, a_widths, intensity_range: tuple):
    intensities = torch.rand(pos.shape[0], device=pos.device) * (
            intensity_range[1] - intensity_range[0]
    ) + intensity_range[0]
    #increase intensities
    amp_indices = (pos < 160) | (widths < 2.) | (a_widths < 5.)
    intensities[amp_indices] = intensities[amp_indices] * 2.
    #rescale to desired range
    intensities = (intensities - torch.min(intensities))/(torch.max(intensities) - torch.min(intensities)) *(
            intensity_range[1] - intensity_range[0]) + intensity_range[0]
    return intensities


def flip_image(img, boxes, mask):
    if np.random.rand() > 0.5:
        img = torch.flip(img, dims=(0,))
        boxes = flip_boxes(boxes, 0, img.shape)
        mask = torch.flip(mask, dims=(0,))
    if np.random.rand() > 0.5:
        img = torch.flip(img, dims=(1,))
        boxes = flip_boxes(boxes, 1, img.shape)
        mask = torch.flip(mask, dims=(1,))

    return img, boxes, mask


def flip_boxes(boxes, ax, shape):
    if ax == 0:
        boxes[:, 1::2] = shape[0] - torch.flip(boxes[:, 1::2], dims=(1,))
    if ax == 1:
        boxes[:, ::2] = shape[1] - torch.flip(boxes[:, ::2], dims=(1,))
    return boxes


def filter_nms(pos, widths, a_pos, a_widths, is_ring, min_nms: float = 0.001):
    idx_boxes = torch.stack(
        [
            pos - widths * 2.5,
            a_pos - a_widths * 3.5,
            pos + widths * 2.5,
            a_pos + a_widths * 3.5
        ], 1
    )
    indices = nms(idx_boxes, torch.ones(idx_boxes.shape[0], device=idx_boxes.device, dtype=torch.float), min_nms)
    return pos[indices], widths[indices], a_pos[indices], a_widths[indices], indices





def torch_uniform(low=0, high=1, *sizes, device='cuda'):
    return torch.rand(*sizes, device=device) * (high - low) + low


_SMOOTH_KERNEL = [[1., 1., 1.],
                  [1., 0.3, 1.],
                  [1., 1., 1.]]