"""Pick the learning rate for the long run from the lr-test run's final +nms scores.
  python heatmap/pick_lr.py <evaluate_final.txt of the lr-1e-4 run>
Prints the lr for the long run (batch 8; sqrt-scaled from the winner). Rule: the lr-1e-4 run wins if the mean of its
+nms ap_total (organic, 41) beats the ridge run's epoch-59 NO-FLOOR mean (organic 0.6067, 41 0.7112 -> 0.6590); otherwise 3e-4.
Any parse problem falls back to 3e-4. Everything it decides is also printed to stderr."""
import re, sys
BASE = (0.6067 + 0.7112) / 2        # ridge run, +nms, NO score floor (evaluate_final_nofloor.txt)
try:
    txt = open(sys.argv[1]).read()
    ap = {m.group(1): float(m.group(2)) for m in re.finditer(r'\+nms\] (organic|41): ap_total ([0-9.]+)', txt)}
    mean = (ap['organic'] + ap['41']) / 2
    win = mean > BASE
    print(f'lr-1e-4 run +nms: {ap}, mean {mean:.4f} vs ridge baseline {BASE:.4f} -> {"1e-4" if win else "3e-4"}', file=sys.stderr)
except Exception as e:
    win = False
    print(f'pick_lr fallback to 3e-4 ({type(e).__name__}: {e})', file=sys.stderr)
print('1.4e-4' if win else '4.2e-4')
