"""Render the simulator pipeline document to a multi-page PDF (matplotlib only)."""
import os, sys, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

W = '/mnt/lustre/work/schreiber/szb389/tmp_diag/sim2'
OUT = f'{W}/doc/simulator_pipeline.pdf'
A4 = (8.27, 11.69)
INK, MUTE, ACC, BOX = '#1a1a1a', '#5a5a5a', '#8c2d19', '#f2efe9'

class Page:
    """A page with a running cursor in figure coordinates (1.0 = top, 0.0 = bottom)."""
    def __init__(self, pdf, title=None, kicker=None):
        self.fig = plt.figure(figsize=A4)
        self.pdf = pdf
        self.y = 0.945
        self.L, self.R = 0.085, 0.915
        if kicker:
            self.fig.text(self.L, self.y, '  '.join(kicker.upper()), fontsize=7.0, color=ACC,
                          family='DejaVu Sans', weight='bold', va='top')
            self.y -= 0.018
        if title:
            self.fig.text(self.L, self.y, title, fontsize=17, color=INK,
                          family='DejaVu Serif', weight='bold', va='top')
            self.y -= 0.045
    def gap(self, h=0.012):
        self.y -= h
    def para(self, text, size=9.2, color=INK, width=96, indent=0.0, leading=0.0148):
        lines = []
        for block in text.strip().split('\n'):
            lines += textwrap.wrap(block, width) or ['']
        self.fig.text(self.L+indent, self.y, '\n'.join(lines), fontsize=size, color=color,
                      family='DejaVu Sans', va='top', linespacing=1.45)
        self.y -= leading*len(lines) + 0.006
    def head(self, text, size=11):
        self.gap(0.008)
        self.fig.text(self.L, self.y, text, fontsize=size, color=ACC, family='DejaVu Sans',
                      weight='bold', va='top')
        self.y -= 0.024
    def mono(self, text, size=8.2, box=True):
        lines = text.strip('\n').split('\n')
        h = 0.0135*len(lines) + 0.012
        if box:
            self.fig.patches.append(FancyBboxPatch(
                (self.L-0.012, self.y-h), (self.R-self.L)+0.024, h,
                boxstyle='round,pad=0.006,rounding_size=0.008', transform=self.fig.transFigure,
                facecolor=BOX, edgecolor='none', zorder=0))
        self.fig.text(self.L, self.y-0.010, '\n'.join(lines), fontsize=size, color=INK,
                      family='DejaVu Sans Mono', va='top', linespacing=1.45)
        self.y -= h + 0.008
    def table(self, rows, widths, size=8.4, header=True):
        x0 = self.L
        for i, r in enumerate(rows):
            x = x0
            bold = 'bold' if (header and i == 0) else 'normal'
            for cell, w in zip(r, widths):
                self.fig.text(x, self.y, str(cell), fontsize=size,
                              color=(MUTE if (header and i == 0) else INK),
                              family='DejaVu Sans Mono', weight=bold, va='top')
                x += w
            self.y -= 0.0155
            if header and i == 0:
                self.fig.add_artist(plt.Line2D([self.L, self.R], [self.y+0.008]*2,
                                    color='#cfc9bf', lw=0.8, transform=self.fig.transFigure))
                self.y -= 0.006
        self.y -= 0.008
    def image(self, path, h=None, caption=None, hmax=0.52):
        if not os.path.exists(path):
            return
        img = plt.imread(path)
        w = self.R - self.L
        # figure is A4, so a width fraction w spans w*A4[0] inches; match the image aspect so the
        # axes box has no letterboxing. If that would be too tall, cap the height and narrow the
        # box instead, keeping it centred.
        asp = img.shape[0]/img.shape[1]
        x0 = self.L
        if h is None:
            h = w*A4[0]*asp/A4[1]
            if h > hmax:
                h = hmax
                w = h*A4[1]/(A4[0]*asp)
                x0 = self.L + ((self.R-self.L)-w)/2
        ax = self.fig.add_axes([x0, self.y-h, w, h])
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor('#cfc9bf'); s.set_linewidth(0.6)
        self.y -= h + 0.010
        if caption:
            cl = textwrap.wrap(caption, 118)
            self.fig.text(self.L, self.y, '\n'.join(cl), fontsize=7.6, color=MUTE,
                          family='DejaVu Sans', va='top', style='italic', linespacing=1.4)
            self.y -= 0.013*len(cl) + 0.006
    def close(self, num=None):
        if num is not None:
            self.fig.text(0.5, 0.035, str(num), fontsize=8, color=MUTE, ha='center',
                          family='DejaVu Sans')
        if os.environ.get('PAGE_PNG'):
            self.fig.savefig(f'{W}/doc/page_{num or 0:02d}.png', dpi=110)
        self.pdf.savefig(self.fig); plt.close(self.fig)


# ----------------------------------------------------------------------------------- content
pdf = PdfPages(OUT)

# ---------- cover
f = plt.figure(figsize=A4)
f.text(0.085, 0.80, 'How a training image', fontsize=30, family='DejaVu Serif', color=INK)
f.text(0.085, 0.755, 'is made', fontsize=30, family='DejaVu Serif', color=INK)
f.add_artist(plt.Line2D([0.085, 0.36], [0.735]*2, color=ACC, lw=2.5, transform=f.transFigure))
f.text(0.085, 0.695, 'The GIWAXS simulator, end to end, with every number it uses',
       fontsize=11.5, family='DejaVu Sans', color=MUTE, va='top')
f.text(0.085, 0.63,
       'Real peak-free detector frames are cut into tiles and reassembled into a\n'
       'background; physics-derived diffraction peaks are drawn on top; only the\n'
       'peaks a human could see get a box.',
       fontsize=10.5, family='DejaVu Sans', color=INK, va='top', linespacing=1.6)
rows = [('Output image', '512 x 1024 polar, x = q, y = chi'),
        ('Background source', '90 reviewed peak-free Lambda frames, DESY P03'),
        ('Peak source', '526,131 entries from 58,459 organic CIFs'),
        ('Box convention', 'full extent = coef x sigma, coef = (2.80 chi, 1.30 q)'),
        ('Speed', '0.35 s per frame, single core'),
        ('Code', 'realbkg_simulation.py + realbkg_sim/mosaic_background.py')]
y = 0.44
for k, v in rows:
    f.text(0.085, y, k, fontsize=8.6, family='DejaVu Sans', color=ACC, weight='bold')
    f.text(0.33, y, v, fontsize=8.6, family='DejaVu Sans Mono', color=INK)
    y -= 0.026
f.text(0.085, 0.09, 'mlgidDETECT_DINO  ·  branch multi-channel-analysis  ·  2026-09-17',
       fontsize=8, family='DejaVu Sans', color=MUTE)
if os.environ.get('PAGE_PNG'):
    f.savefig(f'{W}/doc/page_01.png', dpi=110)
pdf.savefig(f); plt.close(f)

# ---------- overview
p = Page(pdf, 'The whole pipeline at a glance', 'overview')
p.para('Fourteen steps, in three groups. Steps 1-8 build a background out of real detector '
       'pixels. Steps 9-12 draw diffraction peaks on top of it. Steps 13-14 decide which of '
       'those peaks the network is told about, and convert the raw counts into the image the '
       'network sees. Everything before step 14 is in real photon counts.')
p.gap(0.004)
STAGES = [('BACKGROUND', '#2f5d50', ['1  pick donors', '2  cut tiles', '3  flatten each tile',
                                     '4  feather seams', '5  crop', '6  radial envelope',
                                     '7  detector mask', '8  measure its noise']),
          ('PEAKS', '#8c2d19', ['9  draw from CIF bank', '10 assign brightness',
                                '11 render pseudo-Voigt', '12 add peak photon noise']),
          ('LABELS', '#3b4a7a', ['13 visibility gate', '14 contrast chain'])]
ax = p.fig.add_axes([p.L, p.y-0.30, p.R-p.L, 0.30]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
x = 0.0
for name, col, items in STAGES:
    w = 0.32 if name != 'LABELS' else 0.28
    ax.add_patch(FancyBboxPatch((x, 0.06), w, 0.86, boxstyle='round,pad=0.012,rounding_size=0.03',
                                facecolor=col, alpha=0.08, edgecolor=col, lw=1.0))
    ax.text(x+w/2, 0.86, name, ha='center', fontsize=9, color=col, weight='bold',
            family='DejaVu Sans')
    yy = 0.76
    for it in items:
        ax.text(x+0.022, yy, it, fontsize=8.0, color=INK, family='DejaVu Sans Mono', va='top')
        yy -= 0.085
    ax.arrow_x = x + w
    x += w + 0.04
for xa in (0.335, 0.695):
    ax.add_patch(FancyArrowPatch((xa, 0.49), (xa+0.028, 0.49), arrowstyle='-|>',
                                 mutation_scale=13, color=MUTE, lw=1.3))
p.y -= 0.305
p.head('The peak funnel, measured over 30 frames at training settings')
p.para('Not every peak the physics produces ends up with a box -- roughly half do. The rest are '
       'real photons in the image that no annotator could have marked, which is exactly the '
       'situation real labelled data is in. Rings are rarer still: 0 in a median frame and at '
       'most 14, because only 15% of frames contain a powder entry at all.', size=8.8)
ax = p.fig.add_axes([p.L, p.y-0.135, p.R-p.L, 0.125]); ax.axis('off')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
FUN = [('drawn from the CIF bank', 55, 8, 146, '#3b4a7a'),
       ('rendered into the image', 53, 8, 135, '#2f5d50'),
       ('labelled with a box', 26, 7, 43, '#8c2d19')]
for i, (name, med, lo, hi, col) in enumerate(FUN):
    yy = 0.80 - i*0.32
    w = 0.34*med/55.0
    ax.add_patch(Rectangle((0.30, yy-0.095), w, 0.19, facecolor=col, alpha=0.20,
                           edgecolor=col, lw=1.0))
    ax.text(0.285, yy, name, ha='right', va='center', fontsize=8.4, family='DejaVu Sans',
            color=INK)
    ax.text(0.30+w+0.014, yy, f'median {med}    range {lo}-{hi}', va='center',
            fontsize=8.0, family='DejaVu Sans Mono', color=MUTE)
p.y -= 0.145
p.head('Why the background is real pixels and not a model')
p.para('A modelled background can only contain what we thought to put in it. A real frame '
       'already carries the detector\'s noise correlations, its module seams, its dead channels, '
       'the beamstop shadow and the way intensity falls off with q. The catch is that a real '
       'frame usually also carries real diffraction, and an earlier attempt that removed those '
       'peaks always left residue -- every such frame taught the network that a real peak is '
       'background. These 90 donors are bare silicon with no sample on it: they never contained '
       'a peak, so nothing has to be removed and nothing can be left behind.')
p.head('Why the donors are cut up instead of used whole')
p.para('A 500,000-image training set would reuse 90 frames about 5,500 times each. Cutting them '
       'into 128 x 128 tiles and reassembling a fresh canvas per image turns 90 frames into '
       'effectively unlimited variety, while every pixel stays a real measurement.')
p.close(2)


# ---------- steps 1-2
p = Page(pdf, 'Steps 1-2  ·  The donors, and cutting them up', 'background')
p.head('1. Pick the donor frames')
p.para('Collaborators delivered 212 candidate raw frames from six beamtimes. Each was rendered '
       'as a numbered contact sheet and reviewed by eye; 90 survived. Removed were frames '
       'containing the direct beam and beamstop, frames with visible diffraction arcs, and the '
       'entire ESRF ID10 set (IDs 108-211), which carries a distinct ring across the whole '
       'beamtime. Dropping ID10 also removed the only 75 um and 200 um detectors, so one canvas '
       'can no longer mix pixel sizes.')
p.table([['beamtime', 'kept', 'samples', 'material', 'detector', 'pixel', 'counts/px'],
         ['2023_04 DESY P03 Ivan', '72', '9', 'bare Si', 'Lambda', '55 um', '16 - 496'],
         ['2023_04 DESY P03 Elena', '18', '2', 'bare Si', 'Lambda', '55 um', '20 - 633'],
         ['removed', '122', '18 by eye, plus all 104 ESRF ID10 frames', '', '', '', '']],
        [0.215, 0.055, 0.075, 0.085, 0.095, 0.075, 0.13], size=8.0)
p.para('Each donor is 516 x 1556 pixels. Two repairs are applied when it is loaded, because the '
       'mosaic\'s x axis is q and its y axis is chi -- so a donor COLUMN becomes a line of '
       'constant q spanning the whole frame, which is exactly what a powder ring looks like.')
p.mono("""repair_lines     per-column and per-row flat field, profile smoothed at sigma 32
                 gain correction applied if 0.5 < g < 2.0, else pixel marked invalid
repair_defects   5x5 median reference (immune to lines up to 2 px wide)
                 pixel replaced if it deviates by more than 6 Poisson sigma
                 replacement = reference + freshly drawn noise, not the reference
                 pixels touched: 0.051% median, 0.101% p90, 0.166% worst frame
                 valid pixels after both repairs: 99.69%""")
p.head('2. Cut tiles, and screen every one')
p.para('Each canvas first picks one exposure class -- donors whose median count rate is within a '
       'factor of two of a randomly chosen target. Relative noise is set by count rate, so a tile '
       'from a 20-count frame is visibly grainier than one from a 500-count frame even after both '
       'are normalised, and mixing them leaves block structure at the seams.')
p.para('Screening donors frame by frame is not enough. A frame can be flat overall and still '
       'contain a shadow edge or a module border somewhere in it, and a 128 px tile that lands '
       'there gets transplanted whole. Every tile is therefore tested on its own: blur the '
       'flattened tile at sigma 8 and compare to its own noise. Detector noise is nearly white '
       'and all but vanishes under that blur; structure does not.')
p.table([['structure ratio  std(blur sigma 8) / std', 'p50', 'p90', 'p95', 'p99', 'max'],
         ['measured over 3,868 random tiles', '0.055', '0.121', '0.155', '0.527', '0.755'],
         ['accepted if below 0.12  ->  rejects 10.3% of draws', '', '', '', '', '']],
        [0.44, 0.09, 0.09, 0.09, 0.09, 0.09])
p.image(f'{W}/images/04_mosaic/tiles_worst.png', hmax=0.185,
        caption='What the screen throws out: the twelve most structured tiles, flattened. Six are '
                'the one donor whose broad diagonal shadow survived the whole-frame review; the '
                'rest are module edges. Laid into a mosaic, a tile like this is a hard-edged '
                'rectangle with no box on it.')
p.close(3)

# ---------- steps 3-5
p = Page(pdf, 'Steps 3-5  ·  Flatten, feather, crop', 'background')
p.head('3. Flatten each tile by its own smooth field')
p.para('A tile is divided by a Gaussian blur of itself at sigma 24. Dividing by the tile\'s '
       'MEDIAN is not enough and the first version proved it: intensity varies strongly within a '
       'frame because of the radial falloff, so a tile from a bright low-q region landed beside '
       'one from a dark high-q region and the step survived as a visible checkerboard -- hard '
       'block edges, narrow in q, exactly the shape the detector is trained to find. Dividing by '
       'a blur removes the level AND the internal gradient. It does not average anything, so the '
       'noise amplitude is untouched.')
p.head('4. Lay the tiles down with feathered seams')
p.para('Tiles are placed on a 768 x 1536 canvas on a 112 px grid, so each overlaps its '
       'neighbours by 16 px, and the overlap is cross-faded with a separable raised cosine rather '
       'than butted together. 98 tiles make one canvas. The blend band is deliberately narrow: a '
       'wide one makes every pixel an average of four independent tiles and suppresses the '
       'detector noise far below what Poisson allows at that brightness.')
p.table([['overlap (of a 128 px tile)', 'noise / Poisson', 'fraction of px >20% over background'],
         ['64', '0.58', '0.0014'],
         ['32', '0.86', '0.0003'],
         ['16   <- in use', '0.87', '0.0001'],
         ['8', '0.96', '0.0002'],
         ['real donors, for reference', '1.00', '0.0009 - 0.0027']],
        [0.30, 0.18, 0.34])
p.head('5. Crop')
p.para('A random 512 x 1024 window is taken from the 768 x 1536 canvas. Two images built from '
       'the same donors still land on different tiles, in a different arrangement, at a different '
       'crop.')
p.image(f'{W}/images/04_mosaic/tiles_passing2.png', h=0.235,
        caption='The twelve most structured tiles that still pass the screen, flattened and '
                'stretched 1-99%. What remains is a few-percent difference in noise texture at '
                'chip boundaries -- far below the 1.5x contrast a peak needs to be labelled.')
p.close(4)

# ---------- steps 6-8
p = Page(pdf, 'Steps 6-8  ·  Shape, mask, noise', 'background')
p.head('6. Put the large-scale shape back')
p.para('Once every tile is individually flattened the mosaic is flat by construction, and a real '
       'GIWAXS background is not -- it is bright at low q and falls away. That envelope is taken '
       'from a real polar frame and blurred at sigma 64, then multiplied in, and the result is '
       'scaled to the count level the tiles were actually cut at, so the graininess matches the '
       'brightness.')
p.mono("""envelope blur   sigma 64 px          typical peak width  sigma_q ~ 4 px
                                     -> the envelope is 16x too broad to be a peak

first attempt   envelope taken from a Lambda module, which lives in DETECTOR space:
                several mosaics came out dark at low q and bright at high q, backwards.
                It must come from a POLAR frame.""")
p.head('7. Apply a detector mask')
p.para('A mosaic has no geometry of its own, and real polar frames have a characteristic masked '
       'high-q wedge and curved module gaps. Masks are taken from the old donor bank: a mask is '
       'pure geometry, so it carries none of the peak contamination that made that bank unusable '
       'as an image source.')
p.head('8. Measure the background\'s own noise')
p.para('Everything about the noise is measured on this background, not assumed -- these are real '
       'detector pixels, so their noise is real detector noise.')
p.mono("""noise map    residual after subtracting a sigma-16 local mean, masked-aware
coefficient  c = noise / sqrt( blur(background, sigma 16) )
             this is the sqrt(I) law measured on THIS frame, used in step 12
             so peak photons end up exactly as grainy as the pixels they land on""")
p.head('What the finished background measures')
p.para('Twenty backgrounds, no peaks drawn, against the same amplitude test used to accept the '
       'donors in the first place:')
p.table([['', 'p50', 'p90', 'max', 'real donors', 'frames with real diffraction'],
         ['fraction of px >20% over local bkg', '0.0000', '0.0000', '0.0000',
          '0.0009-0.0027', '0.024 - 0.052'],
         ['brightest 0.1% of px, over local bkg', '+7.8%', '+11.0%', '+11.3%', '-', '-'],
         ['dynamic range p99.9 / median', '2.39', '8.40', '13.9', '1.67-2.50', '3.8 - 15.0']],
        [0.31, 0.065, 0.065, 0.065, 0.13, 0.20])
p.para('Not one pixel in twenty frames sits more than 20% above its local background. The '
       'dynamic range exceeds the 2.5 used for donors, but that threshold was calibrated on flat '
       'detector modules; a full polar frame has a radial envelope by construction and REAL polar '
       'frames sit at dynamic range 17.3 median. 2.4-13.9 means the envelope is present and '
       'weaker than reality, not that structure crept in.', size=8.8, color=MUTE)
p.close(5)

# ---------- steps 9-10
p = Page(pdf, 'Steps 9-10  ·  Which peaks, and how bright', 'peaks')
p.head('9. Draw peaks from the CIF bank')
p.para('Positions are physics, not invention. A bank precomputed from organic crystal structures '
       'supplies, for every entry, a list of reflections with their q, their azimuth chi and '
       'their relative intensity. The simulator composes a frame from a few entries.')
p.table([['bank contents', '', ''],
         ['unique CIF structures', '58,459', ''],
         ['oriented entries (single-crystal-like, spots)', '467,672', '82-200 reflections each'],
         ['powder entries (rings)', '58,459', '20-191 reflections each'],
         ['total reflections stored', '98,333,366', '']],
        [0.40, 0.14, 0.34])
p.gap(0.004)
p.table([['per frame', 'value', 'meaning'],
         ['oriented crystals', '1 to 3', 'uniform; config realbkg_n_oriented'],
         ['spots kept per crystal', '8 to 60', 'uniform cap, brightest first'],
         ['probability the frame has rings', '0.15', 'config realbkg_p_ring'],
         ['powder entries when it does', '1', 'config n_powder'],
         ['rings kept per powder entry', '3 to 15', 'uniform cap, brightest first'],
         ['reflections dropped', 'q >= 0.995 q_max', 'off the right edge of the frame'],
         ['frame rejected if', '< 3 peaks inside mask', 'redrawn from scratch']],
        [0.28, 0.24, 0.36], size=8.0)
p.mono("""placement   x = q / q_max * 1024          y = chi / 90 * 512
            rings are placed at y = 256 with sigma_chi = 1e4, i.e. full height,
            and their box is stretched to the full 512 rows""")
p.head('10. Assign brightness: pygidSIM orders them, real data sets the values')
p.para('The CIF bank gives a physically correct ORDERING of peak intensities but not values in '
       'detector counts. Those come from the distribution fitted to real labelled frames: peak '
       'amplitude divided by the local noise. The brightest peaks are rank-matched onto that '
       'lognormal, so the labelled set reproduces the real spread and the real count per frame '
       'once the visibility gate has cut the faint end.')
p.table([['fitted from real labelled data', 'value', 'source sample'],
         ['labelled peaks per frame', '18 - 63, median 35', '57 real frames'],
         ['amplitude / local noise  (lognormal)', 'mu 1.364, sigma 1.002', '214 real peaks'],
         ['   -> median amplitude', '3.9 x local noise', 'real p50 3.12'],
         ['   -> real range', '0.43 - 442 x noise', '']],
        [0.36, 0.24, 0.26])
p.para('Peaks fainter than the rank-matched set are not thrown away. They are placed just below '
       'the labelling threshold, at contrast_min x exp(-|N(0, 0.8)|) -- rendered, because those '
       'photons really are in the frame, but unlabelled, because nobody could mark them. That is '
       'deliberate: real data contains such peaks too, and a simulator that omits them would '
       'train the network to expect an unnaturally clean image.')
p.close(6)

# ---------- steps 11-12
p = Page(pdf, 'Steps 11-12  ·  Drawing the peaks', 'peaks')
p.head('11. Shape: a pseudo-Voigt with tapered wings')
p.para('A pure Gaussian stops too abruptly -- real peaks at two half-widths are 4 to 5 times '
       'brighter than a Gaussian predicts. The profile is a mix of a Gaussian core and Lorentzian '
       'wings, both normalised to the SAME full width at half maximum, so the mixing weight '
       'cannot move the peak\'s visible width. That is what keeps the box convention exact for '
       'any mixing weight.')
p.mono("""mixing weight eta     uniform 0.6 - 1.0     (1.0 = pure Gaussian core weight)
wing taper            exp( -(u^2/uc^2)^2 ),  uc^2 = voigt_cut^2 * 2 ln2
voigt_cut             3.0 half-widths
render window         2.0 x voigt_cut = 6 sigma around each peak
   at that radius the Lorentzian term is 0.037 and the taper is 2.4e-4,
   together under 1e-5 of the peak amplitude -- truncating there is invisible

why a window   full-frame evaluation cost 4.16 s per simulated frame; per-peak
               windows cost 0.30 s, a 14x speed-up, and is what makes generating
               images during training feasible at all""")
p.head('Widths, also fitted to real peaks')
p.table([['', 'distribution', 'median', 'clipped to', 'real reference (1,926 peaks)'],
         ['sigma_q  (radial)', 'lognormal 1.359 / 0.395', '3.9 px', '0.7 - 25 px', 'p50 3.81, range 0.26-18.2'],
         ['sigma_chi (azimuth)', 'lognormal 3.010 / 0.710', '20.3 px', '2.0 - 160 px', 'p50 22.5, range 1.5-302']],
        [0.17, 0.25, 0.09, 0.14, 0.31], size=8.0)
p.para('A second lognormal of width 0.53 (q) and 0.83 (chi) scatters each individual peak around '
       'its frame\'s value, so peaks within one frame are not all identical.', size=8.8)
p.head('Arcs: elongation')
p.table([['probability a frame elongates at all', '0.30'],
         ['probability a given peak in that frame elongates', '0.30'],
         ['elongation factor', '2.0 - 5.0 x'],
         ['elongated along chi (an arc) rather than q', '0.80'],
         ['amplitude divided by the factor', 'yes - same reflection, longer footprint']],
        [0.46, 0.34], header=False)
p.head('12. Add counting noise to the peak photons only')
p.para('The background already carries its own real noise; adding it again would double-count '
       'what the real frame already has. So noise is added only to the rendered peaks, using the '
       'coefficient measured on this background in step 8.')
p.mono("""peaks = peaks + c * sqrt(peaks) * N(0,1)        c from step 8, per pixel

before rendering, peaks with amp < 0.2 x local noise are skipped entirely:
their brightest pixel would sit a fifth of a sigma above background -- nothing
a render would show, and nothing the gate would ever keep.""")
p.close(7)

# ---------- steps 13-14
p = Page(pdf, 'Steps 13-14  ·  Labels, and the image the network sees', 'labels')
p.head('13. Decide which peaks get a box')
p.para('This is the most important step in the whole pipeline, because it defines the ground '
       'truth. A peak is labelled only if a human annotator could have seen it. Two conditions, '
       'both required, both computed from the peak\'s OWN amplitude so a bright neighbour cannot '
       'lend a faint peak its signal.')
p.table([['condition', 'threshold', 'what it means'],
         ['contrast = amp / local noise', '>= 1.5', 'the peak stands above the grain'],
         ['matched-filter SNR', '>= 6.0', 'and has enough area to be believed'],
         ['   SNR = contrast x sqrt(n_eff)', '', ''],
         ['   n_eff, spot', 'pi x sigma_q x min(sigma_chi, 512)', 'effective pixel count'],
         ['   n_eff, ring', 'L x sigma_q x sqrt(pi)', 'L = unmasked column height'],
         ['ring de-duplication', 'IoU > 0.10 dropped', 'keeps the brighter of two rings'],
         ['frame rejected if', '0 boxes survive', 'redrawn from scratch']],
        [0.27, 0.27, 0.33])
p.head('Box convention')
p.mono("""FULL box extent = coefficient x rendered sigma

    coefficient = 2.80 in chi (height)      1.30 in q (width)

box = [ x - 1.30*sigma_q/2 ,  y - 2.80*sigma_chi/2 ,
        x + 1.30*sigma_q/2 ,  y + 2.80*sigma_chi/2 ]

rings override the vertical extent to the full frame: y from 0 to 512
boxes are clipped to the frame and dropped if they collapse to zero area""")
p.head('14. Contrast chain')
p.para('Everything up to here is in real photon counts. The last step converts counts into the '
       'image the network actually receives, using the identical chain the real evaluation frames '
       'go through -- so simulated and real images are processed the same way.')
p.mono("""1  clip to the 5th and 99.5th percentile of the unmasked pixels
2  log10
3  global histogram equalisation
4  masked pixels zeroed

the review HDF5 is written BEFORE this step, so what you inspect is raw counts""")
p.para('A frame is discarded and redrawn if the result is not finite -- a constant image makes '
       'the normalisation 0/0, which would otherwise reach the matcher as NaN and kill the run '
       'mid-epoch.', size=8.8, color=MUTE)
p.close(8)

# ---------- freshness + cost
p = Page(pdf, 'Freshness, cost, and what runs when', 'in training')
p.head('Nothing is cached between runs')
p.para('A saved background bank would mean every run trains on the same pixels in the same '
       'arrangement, which is the reuse problem the mosaic exists to solve. Instead:')
p.table([['at run start', 'a pool of 48 backgrounds is assembled from the 90 donors'],
         ['during the run', 'one pool slot is rebuilt every 64 simulated frames'],
         ['over 500k frames', 'the pool turns over roughly 160 times'],
         ['cost of a rebuild', '~1.0 s, i.e. ~15 ms per frame amortised'],
         ['cost of a frame', '0.35 s measured, single core'],
         ['what IS cached', 'only the repaired donor PIXELS, keyed to the donor list;'],
         ['', 'the tiling is redone per image, so freshness is unaffected'],
         ['seed', 'left unseeded on purpose; set realbkg_mosaic_seed to reproduce']],
        [0.22, 0.60], header=False)
p.head('Is generating images during training feasible?')
p.para('Yes, with margin. At 0.35 s per frame a single dataloader worker produces 2.9 images per '
       'second; four workers produce about 11. The detector consumes roughly 1.9 images per '
       'second. Before the per-peak render window this was 4.16 s per frame, i.e. 1.9 images per '
       'second at eight workers -- exactly break-even and therefore no margin at all.')
p.head('Config switches')
p.mono("""config/DINO/DINO_4scale_swin_realbkg.py

use_realbkg_sim        = True     image source is this simulator, exclusively
realbkg_mosaic         = True     backgrounds are mosaics; donor bank NOT read
realbkg_mosaic_pool    = 48       backgrounds held in memory
realbkg_mosaic_refresh = 64       rebuild one slot every N frames
realbkg_n_oriented     = (1, 3)   crystals per frame
realbkg_p_ring         = 0.15     probability a frame contains powder rings
box_coef_override      = (2.80, 1.30)     (chi, q)""")
p.head('One number worth revisiting later')
p.para('realbkg_p_ring = 0.15 is the one lever already known to separate the two evaluation sets: '
       'the legacy simulator\'s ring rate matches the 41-set\'s composition, the physics-CIF rate '
       'matches the organic set\'s. It is left at 0.15 for the first mosaic run so that the '
       'background change is the only variable being tested.', color=MUTE, size=8.8)
p.close(9)

# ---------- results
p = Page(pdf, 'What comes out', 'result')
p.para('Twenty frames written to datasets/mosaicsim_raw.h5, raw counts, in the same layout as '
       'the ground-truth set. Peak content is stratified rather than left to chance -- sparse, '
       'medium and crowded spot counts, with and without rings -- and each frame\'s background is '
       'cut from its own disjoint group of 4-5 donors, so no two frames share a single source '
       'pixel. Training does the opposite, drawing from the widest pool it can, so this file is a '
       'strict worst case for background variety rather than the typical one.')
p.image(f'{W}/images/05_sim_output/review20_p1.png', h=0.46,
        caption='Five of the twenty, log + histogram equalised. Red = spot boxes, cyan = ring '
                'boxes. Title gives the stratum, the donor IDs behind that background, its level '
                'in counts per pixel, and the raw intensity range.')
p.table([['file', 'datasets/mosaicsim_raw.h5   (96.5 MB, 20 frames)'],
         ['boxes per frame', '19 to 59, median 27'],
         ['frames with rings', '8 of 20  (forced high for review; training uses 0.15)'],
         ['entry_simNN/polar/image', 'the raw polar frame, lossless, pre-contrast'],
         ['entry_simNN/data/img_gid_q', 'reciprocal-space resampling (loses the high-q wedge)'],
         ['  .../analysis/frame00000/fitted_peaks', 'ground-truth boxes, pygid PEAK_DTYPE'],
         ['entry_simNN/process/settings', 'which donor IDs that background came from']],
        [0.345, 0.50], header=False, size=8.0)
p.close(10)

# ---------- limits
p = Page(pdf, 'What this does not do', 'honest limits')
p.para('Every one of these is a known, deliberate choice rather than an oversight.')
p.head('The background has no q-dependent texture')
p.para('Tiles are shuffled freely, so a tile cut at high q can land at low q. That destroys the '
       'physical correlation between position and texture; only the smooth envelope restores the '
       'intensity gradient. The alternative is implemented but off: q_locked=True keeps each tile '
       'at its own q and shuffles only along chi.')
p.head('One detector, one material, one facility')
p.para('All 90 donors are bare silicon on a Lambda detector at DESY P03. That uniformity is '
       'deliberate -- uniform texture beats variety we cannot vouch for -- but it does mean the '
       'network sees one detector\'s noise character. The ESRF ID10 frames are still in '
       'master_donors.json under their IDs if they are ever wanted back for an A/B test.')
p.head('q_max is borrowed')
p.para('A mosaic has no geometry of its own, so the q at the right edge of the frame is taken '
       'from a real polar frame. That only fixes where peaks land in q, which is what has to '
       'match the evaluation data\'s convention.')
p.head('The masks come from the rejected bank')
p.para('That bank is unusable as an image source because of its unremoved peaks, but a mask is '
       'pure geometry and carries none of that contamination.')
p.head('Unlabelled faint peaks are present on purpose')
p.para('Peaks below the visibility gate are drawn but not boxed. This mirrors real data, where '
       'such peaks also exist unlabelled -- but it does mean the ground truth is not a complete '
       'inventory of every photon in the frame, and precision measured against it has a floor.')
p.head('Two metrics that were wrong, recorded so they are not repeated')
p.para('Blankness was twice judged by a statistical significance score, which asks "is this '
       'structure real?" and therefore scales with exposure. The same bare wafer scored 2.9e-3 at '
       '3.8 counts/px and 4.5e-2 at 396 counts/px -- so the metric rejected the BEST data twice. '
       'The right question is amplitude over local background, which is exposure-free. Separately, '
       'a dynamic-range cut of 2.5 calibrated on flat detector modules must never be applied to '
       'full polar frames, which carry an envelope and sit at 17.3 in reality.', color=MUTE,
       size=8.8)
p.close(11)

pdf.close()
print('wrote', OUT, os.path.getsize(OUT)/1e6, 'MB')
