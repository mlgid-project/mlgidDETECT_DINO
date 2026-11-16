"""Run generate_dataset.main() with the settings the user decided on 2026-11-02 for the training datasets (so the vendored generate_dataset.py stays byte-identical):
iomin_sr = 0.5 (as in the notebook; the file's DEFAULT_CFG has 0.0), min_snr = (1.1, 1.1) (= file default). Everything else is the file's DEFAULT_CFG.
Usage identical to generate_dataset.py:  python gen_v2.py --out DIR --n N [--workers W] [--shard-size S]
The effective config is written to <out>/config.json by generate_dataset.main()."""
import pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_dataset as G

G.DEFAULT_CFG.update(iomin_sr=0.5, min_snr=[1.1, 1.1])

if __name__ == '__main__':
    G.main()
