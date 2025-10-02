from typing import Iterator, List
from torch import Tensor
from .matchers import get_matcher
from .metrics import FullMetrics, FalsePositive, FalseNegative, MatchedPair
import numpy as np
import torch
import pandas as pd

def _get_nonselected_indices(selected_indices: np.ndarray, size: int):
    return np.array(list(set(np.arange(size)).difference(selected_indices)))

def _df_from_idx(
    idx, 
    name,
    recalls, 
    precisions, 
    accuracies, 
    scores, 
    av_precision,
    recalls_levels,
    fp_nums,
):
    
    res = {
        'fp': [fp_nums[idx]],
        'recall_total': [recalls[idx]],
        'recall_high': [recalls_levels[1.][idx]],
        'recall_medium': [recalls_levels[0.5][idx]],
        'recall_low': [recalls_levels[np.float32(0.1)][idx]],
        'min_score': [scores[idx]],
    }
    
    return pd.DataFrame(data=res, index=[name])

def _get_av_precision(recalls: List[float], precisions: List[float]):
    if not precisions:
        return 0
    recalls = [0.] + recalls
    precisions = [precisions[0]] + precisions

    recalls = np.array(recalls)
    precisions = np.array(precisions)

    av_precision = ((recalls[1:] - recalls[:-1]) * (precisions[1:] + precisions[:-1]) / 2).sum()

    return av_precision

def _interpolate_precisions(precisions: List[float]):
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])


def recall_precision_curve_with_intensities(metrics: FullMetrics, interpolate_precision: bool = True):
    scores = np.concatenate([metrics.matched_scores, metrics.fp_scores])
    labels = np.concatenate([np.ones_like(metrics.matched_scores), np.zeros_like(metrics.fp_scores)])
    intensities = np.concatenate([metrics.matched_intensities, np.zeros_like(metrics.fp_scores)])

    if not labels.size:
        return [], [], [], [], 0, {}, []

    num_fn = len(metrics.false_negatives) + len(metrics.matched_ious)
    num_fp, matched = 0, 0

    matched_levels = {}
    num_fn_levels = {}

    indices = np.argsort(scores)[::-1]  # start with high scores; fp = 0, fn = total.
    recalls, precisions, accuracies, fp_nums = [], [], [], []
    recalls_levels = {}

    keys = np.float32(np.unique(np.concatenate([
        metrics.missed_intensities, metrics.matched_intensities
    ]),))

    for key in keys:

        matched_levels[key] = 0
        num_fn_levels[key] = (
                (metrics.missed_intensities == key).sum() +
                (metrics.matched_intensities == key).sum()
        )
        recalls_levels[key] = []

    labels = labels[indices]
    scores = scores[indices]
    intensities = intensities[indices]

    for label, intensity in zip(labels, intensities):
        if label:
            matched += 1
            num_fn -= 1.
            matched_levels[np.float32(intensity)] += 1.
            num_fn_levels[np.float32(intensity)] -= 1.
        else:
            num_fp += 1.


        for key in keys:
            recalls_levels[key].append(
                matched_levels[key] / (matched_levels[key] + num_fn_levels[key])
            )

        fp_nums.append(num_fp)
        recalls.append(matched / (matched + num_fn))
        precisions.append(matched / (matched + num_fp))
        accuracies.append(matched / (matched + num_fp + num_fn))

    if interpolate_precision:
        _interpolate_precisions(precisions)

    av_precision = _get_av_precision(recalls, precisions)

    fp_nums = np.array(fp_nums) / (len(metrics.false_negatives) + len(metrics.matched_ious))

    return recalls, precisions, accuracies, scores, av_precision, recalls_levels, fp_nums



class Evaluator():
    
    def __init__(self, match_criterion: str = 'q', match_threshold: float = 0.1) -> None:
        self.match_criterion = match_criterion
        self.match_threshold = match_threshold
        self.matcher = get_matcher(match_criterion=match_criterion, min_iou=match_threshold)
        self.metrics = FullMetrics()
        pass

    @torch.no_grad()
    def get_full_metrics(self,
            target: Tensor,
            predicted: Tensor,
            scores: Tensor,
            matcher ,
            intensities: np.ndarray = None,
            **kwargs
    ):
        if intensities is None:
            intensities = - np.ones(target.shape[0])

        intensities = torch.from_numpy(intensities).to(torch.float32)

        try:  # if predicted is empty, matcher throws an IndexError. Fixing it here is simpler than in every of the matcher classes
            iou_mtx, row_ind, col_ind = matcher(target, predicted)
        except IndexError:
            return FullMetrics()

        missed_indices = _get_nonselected_indices(row_ind, target.shape[0])
        fp_indices = _get_nonselected_indices(col_ind, predicted.shape[0])

        matched_pairs = [
            MatchedPair(t_box, p_box, iou, score, intensity)
            for t_box, p_box, iou, score, intensity in zip(
                target[row_ind].tolist(),
                predicted[col_ind].tolist(),
                iou_mtx[row_ind, col_ind].tolist(),
                scores[col_ind].tolist(),
                intensities[row_ind].tolist()
            )
        ]

        if fp_indices.size == 0:
            false_positives = []
        else:
            false_positives = [
                FalsePositive(p_box, score) for p_box, score in zip(
                    predicted[fp_indices].tolist(), scores[fp_indices].tolist()
                )
            ]

        false_negatives = [
            FalseNegative(t_box, intensity) for t_box, intensity in zip(
                target[missed_indices].tolist(), intensities[missed_indices].tolist()
            )
        ]

        return FullMetrics(
            matched_pairs, false_positives, false_negatives,
            [len(matched_pairs)], [len(false_positives)], [len(false_negatives)]
        )

    def get_exp_metrics(self, predictions: Tensor, scores: Tensor, targets: Tensor, confidences: Tensor):

        self.metrics += self.get_full_metrics(
            target=targets,
            predicted=predictions,
            scores=scores,
            matcher=self.matcher,
            intensities=confidences,
        )


def get_full_conf_results(exp_metrics, name: str = 'exp metrics', max_fp: float = 0.1, min_recall_high: float = 0.95):
    rpres = (
        recalls, precisions, accuracies, scores, av_precision, recalls_levels, fp_nums
    ) = recall_precision_curve_with_intensities(exp_metrics)
    
    df_ap = pd.DataFrame(data={
        'ap_high': _get_av_precision(recalls_levels[1.], precisions),
        'ap_med': _get_av_precision(recalls_levels[0.5], precisions),
        'ap_low': _get_av_precision(recalls_levels[np.float32(0.1)], precisions),
        'ap_total': av_precision,
    }, index=[name])
        
    df = pd.DataFrame()
        
    max_idx = np.argmax(accuracies)
    
    best_score = scores[max_idx]
    
    df = pd.concat([df,_df_from_idx(max_idx, f'score = {best_score:.1f}', *rpres)])
    
    idx = np.argmin(np.abs(np.array(fp_nums) - max_fp))
    
    df = pd.concat([df,_df_from_idx(idx, f'fp <= {int(100 * max_fp)}', *rpres)])
    
    idx = np.argmin(np.abs(np.array(recalls_levels[1.]) - min_recall_high))
    
    df = pd.concat([df,_df_from_idx(idx, f'high_recall >= {int(100 * min_recall_high)}', *rpres)])
    
    return df, df_ap