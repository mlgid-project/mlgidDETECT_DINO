"""Embed frames stored as uint8 arrays with the latent-map backbone (same vector as embed_corpus.py / embed_sim.py), for translate_eval.py output (GPU, mlgpu_clean env like the other embeddings).

  HM_DATA_DIR=<dir> PYTHONPATH=. python heatmap/embed_arrays.py --frames <translated.npy> --masks <mask.npy> --name <sim_name> --out <file.npz>

Output npz as embed_sim.py (feat, set = 'sim_<name>', frame, n_boxes = 0), readable by heatmap/latent_sim.py."""
import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from heatmap.embed_corpus import embed, DATA, DinoV3Backbone, torch, np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames', required=True); p.add_argument('--masks', required=True); p.add_argument('--name', required=True); p.add_argument('--out', required=True)
    p.add_argument('--bb', default='dinov3_vith16plus'); p.add_argument('--weights', default=f'{DATA}/pretrain_runs/vith16plus_3k_sk2/backbone_teacher.pth')
    a = p.parse_args()
    F, M = np.load(a.frames, mmap_mode='r'), np.load(a.masks, mmap_mode='r')
    bb = DinoV3Backbone(a.bb); bb.net.load_state_dict(torch.load(a.weights, map_location='cpu', mmap=True), strict=True); bb = bb.to('cuda').eval()
    feats = [embed(bb, np.asarray(F[i], np.float32) / 255 * (np.asarray(M[i]) > 0), np.asarray(M[i]) > 0, 'cuda').astype(np.float16) for i in range(len(F))]
    np.savez(a.out, feat=np.stack(feats), set=np.array([f'sim_{a.name}'] * len(F)), frame=np.arange(len(F)), n_boxes=np.zeros(len(F), int), bb=a.bb, weights=a.weights)
    print('saved', a.out, len(F), flush=True)


if __name__ == '__main__':
    main()
