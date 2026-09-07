"""Contrast settings and their canonical labels.

Kept separate from diagnostics/sweep_contrast.py so anything that needs to name or look up a
contrast (the sweep, the image dumper) shares ONE definition of the grid and ONE label
spelling -- the labels are the join key between the sweep CSV and everything downstream.

A setting is a dict: {'clip': None|(lo_pct, hi_pct), 'log': bool, 'gamma': None|float,
'he': bool, 'clahe': None|(clip_limit, tile_h, tile_w)}. util.exp_preprocess.apply_contrast
is what turns one into an image.
"""
import itertools

DEFAULT_SETTING = {'clip': (5.0, 99.5), 'log': True, 'gamma': None, 'he': True, 'clahe': None}

#clipping percentile pairs. EXTRA_CLIPS are the aggressive/asymmetric ones added later; they
#are also available on their own as the 'clips_extra' grid so they can be appended to an
#existing sweep CSV without re-running the whole thing.
BASE_CLIPS = [None, (0, 100), (1, 99), (5, 99.5), (10, 99), (2, 99.9), (5, 99.99)]
EXTRA_CLIPS = [(25, 90), (5, 50), (75, 99.9)]


def label(s: dict) -> str:
    c = s.get('clip')
    parts = ['clip=' + ('none' if c is None else f'{c[0]:g}/{c[1]:g}')]
    parts.append('log' if s.get('log') else 'nolog')
    if s.get('gamma'):
        parts.append(f"gamma{s['gamma']:g}")
    if s.get('he'):
        parts.append('he')
    elif s.get('clahe'):
        l, th, tw = s['clahe']
        parts.append(f'clahe{l:g}@{th}x{tw}')
    else:
        parts.append('nohe')
    return '_'.join(parts)


def _clip_block(clips):
    return [{'clip': c, 'log': lg, 'gamma': None, 'he': he, 'clahe': None}
            for c, lg, he in itertools.product(clips, [True, False], [True, False])]


def build_grid(name: str):
    """Named sweep grids. Every grid contains the deployed default as a reference row."""
    grids = {}

    #factorial: clipping percentiles x log x global HE  (the three deployed knobs)
    grids['main'] = _clip_block(BASE_CLIPS + EXTRA_CLIPS)
    grids['clips_extra'] = _clip_block(EXTRA_CLIPS)
    #power-law compression as an alternative to log/HE
    grids['gamma'] = [
        {'clip': (5, 99.5), 'log': lg, 'gamma': g, 'he': False, 'clahe': None}
        for lg, g in itertools.product([True, False], [0.3, 0.5, 0.7, 1.5, 2.0])
    ]
    #local contrast instead of the global histogram
    grids['clahe'] = [
        {'clip': (5, 99.5), 'log': lg, 'gamma': None, 'he': False, 'clahe': (l, th, tw)}
        for lg, (l, (th, tw)) in itertools.product(
            [True, False], itertools.product([1.0, 2.0, 4.0, 8.0], [(8, 8), (4, 16), (16, 16)]))
    ]
    grids['base'] = [dict(DEFAULT_SETTING)]
    grids['all'] = grids['main'] + grids['gamma'] + grids['clahe']

    if name not in grids:
        raise SystemExit(f'unknown grid {name!r}; choose from {sorted(grids)}')
    out, seen = [], set()
    for s in [dict(DEFAULT_SETTING)] + grids[name]:
        k = label(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def by_label(name: str) -> dict:
    """Resolve one canonical label (as written in the sweep CSV) back to its setting dict."""
    for s in build_grid('all'):
        if label(s) == name:
            return s
    raise SystemExit(f'unknown contrast label {name!r}')
