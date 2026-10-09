"""One-page PDF explaining the heatmap detector (models/heatmap_head.py).

  python heatmap/make_architecture_pdf.py --out <dir>/heatmap_architecture.pdf [--png]

Pure matplotlib + numpy (no torch, no data). The example image / heatmap / boxes are a synthetic illustration, not model output.
The numbers in the text are those of the default head (stride 2, dim 128): 195,486,084 frozen swin parameters, 1,228,582 trained."""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

PAGE_W, PAGE_H = 11.69, 8.27                       # A4 landscape, inches
UW, UH = PAGE_W / PAGE_H * 100, 100.0              # drawing units: 1 unit = same length in x and y
INK, MUTED, BG = '#18222f', '#5d6b7a', '#fbfaf7'
FROZEN = dict(fc='#e3ebf4', ec='#4d6a8a')
TRAIN = dict(fc='#fdebd0', ec='#d6861a')
IO = dict(fc='#f1f3f5', ec='#8993a0')
SEG, RING = '#12968a', '#e0522b'


def box(ax, x, y, w, h, style, lw=1.5, r=1.4, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=style['fc'], ec=style['ec'], lw=lw, zorder=z))


def arrow(ax, p, q, color=INK, lw=1.8, rad=0.0, ms=15):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw, connectionstyle=f'arc3,rad={rad}', zorder=4,
                                 shrinkA=0, shrinkB=0))


def line(ax, pts, color=INK, lw=1.8):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, solid_capstyle='round', zorder=4)


def badge(ax, x, y, text, style):
    ax.text(x, y, text, ha='center', va='center', fontsize=6.5, fontweight='bold', color='white', zorder=6,
            bbox=dict(boxstyle='round,pad=0.28,rounding_size=0.6', fc=style['ec'], ec='none'))


def header(ax, x, y, w, title, style, tag):
    ax.text(x + 1.8, y, title, ha='left', va='center', fontsize=10.5, fontweight='bold', color=INK, zorder=6)
    badge(ax, x + w - 5.6, y + 2.6, tag, style)      # sits on the top edge of the box, never on the title


def body(ax, x, y, text, size=8.2, color=INK, ha='left', va='top', weight='normal', ls=1.35):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color, linespacing=ls, zorder=6, fontweight=weight)


def scene(seed=3):
    rng = np.random.default_rng(seed)
    H, W = 128, 256
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    spots = [(34, 30), (40, 36), (22, 78), (30, 100), (94, 40), (85, 92), (60, 110), (105, 130), (45, 150), (88, 170),
             (28, 195), (70, 205), (101, 226), (50, 238)]
    ring_x = 70
    img = np.exp(-xx / 120.0) * 0.5 + rng.gamma(2.0, 0.045, (H, W))
    heat_s = np.zeros((H, W)); heat_r = np.zeros((H, W))
    for (cy, cx) in spots:
        a = rng.uniform(0.5, 1.0)
        img += a * np.exp(-((xx - cx) ** 2 / (2 * 2.2 ** 2) + (yy - cy) ** 2 / (2 * 2.6 ** 2)))
        heat_s += np.exp(-((xx - cx) ** 2 / (2 * 1.8 ** 2) + (yy - cy) ** 2 / (2 * 1.8 ** 2))) * min(1.0, a + 0.25)
    img += 0.55 * np.exp(-((xx - ring_x) ** 2) / (2 * 1.6 ** 2)) * (0.8 + 0.2 * np.sin(yy / 9))
    heat_r = 0.9 * np.exp(-((xx - ring_x) ** 2) / (2 * 1.8 ** 2)) * np.exp(-((yy - 64) ** 2) / (2 * 22 ** 2))
    img = np.log1p(6 * img); img = (img - img.min()) / (img.max() - img.min())
    return img, np.clip(heat_s, 0, 1), np.clip(heat_r, 0, 1), spots, ring_x


def thumb(fig, x, y, w, h, im, cmap, vmin=0, vmax=1):
    a = fig.add_axes([x / UW, y / UH, w / UW, h / UH]); a.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', interpolation='bilinear')
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_edgecolor('#8993a0'); s.set_linewidth(0.8)
    return a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--png', action='store_true')
    a = ap.parse_args()
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, UW); ax.set_ylim(0, UH); ax.axis('off'); ax.set_facecolor(BG)
    img, hs, hr, spots, ring_x = scene()

    # ---- title
    ax.text(3, 95.2, 'Heatmap detector for GIWAXS Bragg peaks', fontsize=21, fontweight='bold', color=INK, va='center')
    ax.text(3, 90.4, 'A frozen swin backbone and a small convolutional head. For every 2 x 2 pixel cell the network says how much it looks like the '
                     'centre of a peak,\nand what box sits around it. Peaks are then read off the map, with no queries, matcher or decoder.',
            fontsize=9, color=MUTED, va='center', linespacing=1.4)

    # ---- row A: input -> backbone -> FPN
    box(ax, 3, 56, 24, 24, IO)
    body(ax, 4.8, 78.0, 'Input', size=10.5, weight='bold', va='center')
    t = thumb(fig, 4.6, 66.2, 20.8, 10.4, img, 'magma')
    body(ax, 15, 64.4, 'polar image, 512 x 1024\n(q along x, chi along y)\nlog + histogram-equalised', size=7.2, color=MUTED, ha='center', ls=1.3)
    arrow(ax, (27, 68), (32, 68))

    box(ax, 32, 56, 26, 24, FROZEN)
    header(ax, 32, 77.4, 26, 'Swin-L backbone', FROZEN, 'FROZEN')
    body(ax, 35.0, 74.7, 'stride', size=5.6, color=MUTED, ha='center', va='center')
    body(ax, 55.2, 74.7, 'channels', size=5.6, color=MUTED, ha='center', va='center')
    for i, (wd, c, st, ch) in enumerate(zip([15, 12, 9, 6], ['#9db4cf', '#86a1c2', '#6f8fb6', '#587ba8'], ['4', '8', '16', '32'], ['192', '384', '768', '1536'])):
        yb = 71.4 - i * 2.55
        ax.add_patch(Rectangle((45 - wd / 2, yb), wd, 2.1, fc=c, ec='white', lw=0.8, zorder=5))
        body(ax, 35.0, yb + 1.05, st, size=6.2, color=MUTED, ha='center', va='center')
        body(ax, 55.2, yb + 1.05, ch, size=6.2, color=MUTED, ha='center', va='center')
    body(ax, 45, 61.0, 'four feature maps\nSimMIM pre-trained, window 48 x 6\n195.5 M parameters', size=6.8, color=INK, ha='center', ls=1.3)
    arrow(ax, (58, 68), (63, 68))

    box(ax, 63, 56, 26, 24, TRAIN)
    header(ax, 63, 77.4, 26, 'Feature pyramid (FPN)', TRAIN, 'TRAINED')
    body(ax, 64.8, 74.6, 'merges the four maps top-down:\n1 x 1 conv to 128 channels,\nupsample, add, 3 x 3 conv\n+ GroupNorm + ReLU', size=7.6)
    box(ax, 65.2, 57.6, 21.6, 4.6, dict(fc='#fff6e6', ec=TRAIN['ec']), lw=1.0, r=0.9)
    body(ax, 76, 59.9, '128-channel map, stride 4', size=7.2, ha='center', va='center', weight='bold')

    # ---- stat cards (right of row A)
    box(ax, 94, 66, 20, 14, dict(fc='#eef3f9', ec=FROZEN['ec']), lw=1.2)
    body(ax, 104, 74.8, '195.5 M', size=19, ha='center', va='center', weight='bold', color=FROZEN['ec'])
    body(ax, 104, 69.6, 'frozen parameters\n(the backbone)', size=7.8, ha='center', va='center', color=INK)
    box(ax, 118, 66, 20, 14, dict(fc='#fef5e6', ec=TRAIN['ec']), lw=1.2)
    body(ax, 128, 74.8, '1.23 M', size=19, ha='center', va='center', weight='bold', color=TRAIN['ec'])
    body(ax, 128, 69.6, 'trained parameters\n(FPN, stem, towers)', size=7.8, ha='center', va='center', color=INK)
    box(ax, 94, 56.0, 44, 8.0, IO, lw=1.0, r=1.0)
    body(ax, 96, 60.0, 'Frozen: weights copied from self-supervised pre-training,\nnever updated.\nTrained: starts random, learns from simulated images only.', size=6.9, color=MUTED, va='center', ls=1.3)

    # ---- row B: stem, fuse, towers, outputs
    box(ax, 32, 30, 26, 20, TRAIN)
    header(ax, 32, 47.4, 26, 'Image stem', TRAIN, 'TRAINED')
    body(ax, 33.8, 44.2, 'looks at the raw image again,\nfor fine position detail:\n2 x (3 x 3 conv + norm + ReLU)\n32 channels, stride 2', size=7.6)
    line(ax, [(15, 56), (15, 40)]); arrow(ax, (15, 40), (32, 40))
    body(ax, 16.4, 48.3, 'same image\ngoes to the stem', size=6.8, color=MUTED)

    box(ax, 63, 30, 26, 20, TRAIN)
    header(ax, 63, 47.4, 26, 'Fuse', TRAIN, 'TRAINED')
    body(ax, 64.8, 44.2, 'concatenate\n128 (FPN) + 32 (stem) = 160 ch\n3 x 3 conv + norm + ReLU\n-> 128 ch at stride 2\n(256 x 512 cells)', size=7.6)
    arrow(ax, (58, 40), (63, 40))
    arrow(ax, (76, 56), (76, 50))
    body(ax, 77.6, 53.0, 'upsample to stride 2', size=6.8, color=MUTED, va='center')

    box(ax, 94, 41, 23, 10, TRAIN)
    header(ax, 94, 48.6, 23, 'Heat tower', TRAIN, 'TRAINED')
    body(ax, 95.8, 45.4, '2 x (3 x 3 conv, 64 ch)\nthen 1 x 1 conv', size=7.6)
    box(ax, 94, 28, 23, 10, TRAIN)
    header(ax, 94, 35.6, 23, 'Box tower', TRAIN, 'TRAINED')
    body(ax, 95.8, 32.4, '2 x (3 x 3 conv, 64 ch)\nthen 1 x 1 conv', size=7.6)
    line(ax, [(89, 40), (91.5, 40), (91.5, 46)]); arrow(ax, (91.5, 46), (94, 46))
    line(ax, [(91.5, 40), (91.5, 33)]); arrow(ax, (91.5, 33), (94, 33))

    box(ax, 121, 41, 17, 10, IO)
    body(ax, 129.5, 49.6, 'Heatmap, 2 channels', size=7.4, ha='center', va='center', weight='bold')
    thumb(fig, 122.2, 42.0, 14.6, 6.1, np.maximum(hs, hr), 'Blues')
    arrow(ax, (117, 46), (121, 46))
    box(ax, 121, 28, 17, 10, IO)
    body(ax, 129.5, 36.2, 'Box maps, 4 channels', size=7.4, ha='center', va='center', weight='bold')
    body(ax, 129.5, 32.2, 'dx, dy: centre offset\nlog width, log height', size=7.0, ha='center', va='center', color=INK, ls=1.3)
    arrow(ax, (117, 33), (121, 33))
    body(ax, 123.5, 52.0, 'segment | ring', size=6.6, color=MUTED, ha='center', va='center')

    # ---- row C: decoding
    body(ax, 3, 25.0, 'Decoding  (no learned part): reads both output maps above', size=11, weight='bold', va='center')
    chips = [('3 x 3 max-pool\npeak picking', 'a cell is a peak if it\nis the largest nearby'),
             ('keep the top 225\npeaks', 'score = heatmap value\nat the peak'),
             ('read the box', 'centre = cell + offset\nsize = exp(log w, log h)'),
             ('class-aware NMS\nscore > 0.1', 'removes duplicates\n(segments, rings apart)')]
    for i, (t1, t2) in enumerate(chips):
        x0 = 3 + i * 23.8
        box(ax, x0, 10.5, 20.4, 11.6, IO)
        body(ax, x0 + 10.2, 19.6, t1, size=8.0, ha='center', va='center', weight='bold', ls=1.25)
        body(ax, x0 + 10.2, 13.6, t2, size=6.6, ha='center', va='center', color=MUTED, ls=1.25)
        if i < 3:
            arrow(ax, (x0 + 20.4, 16.3), (x0 + 23.8, 16.3), lw=1.6, ms=12)
    arrow(ax, (92.4, 16.3), (99.5, 16.3), lw=1.6, ms=12)

    # result thumbnail (boxes on the image)
    body(ax, 119, 24.0, 'Result: boxes on the image', size=8.2, ha='center', va='center', weight='bold')
    r = thumb(fig, 100, 4.2, 38, 18.2, img, 'magma')
    for (cy, cx) in spots:
        r.add_patch(Rectangle((cx - 6, cy - 5), 12, 10, fill=False, ec=SEG, lw=1.0))
    r.add_patch(Rectangle((ring_x - 3, 6), 6, 116, fill=False, ec=RING, lw=1.2))
    r.set_xlim(-0.5, 255.5); r.set_ylim(127.5, -0.5)
    body(ax, 100.4, 2.8, 'teal = segment (spot)    orange = ring', size=6.8, color=MUTED, va='center')

    # ---- training line
    box(ax, 3, 1.3, 90, 8.2, dict(fc='#f6f4ee', ec='#c9c4b5'), lw=1.0, r=1.0)
    body(ax, 4.6, 5.4, 'Training: simulated GIWAXS images only (the same generator as the DINO detector).\n'
                       'Targets: a small Gaussian bump per segment, a tall ridge per ring.\n'
                       'Loss: focal loss (heatmap) + 4 x L1 (box maps). AdamW, lr 3e-4, 60 epochs of 1000 images. The backbone is never updated.',
         size=6.9, color=INK, va='center', ls=1.4)
    ax.text(UW - 1.5, 0.9, 'illustration is synthetic, not model output  |  models/heatmap_head.py', fontsize=5.8, color=MUTED, ha='right', va='center')

    fig.savefig(a.out, format='pdf', facecolor=BG)
    if a.png:
        fig.savefig(a.out.rsplit('.', 1)[0] + '.png', dpi=130, facecolor=BG)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
