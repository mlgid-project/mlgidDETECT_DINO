"""Produce real-background simulator frames in a separate process, so the training step does not wait for the 0.19 s/frame simulator.

One worker holds the simulator (13 GB resident, mostly the hkl bank: two do not fit in colorbox1's 30 GB); the training process never builds it.
Frames come back as numpy arrays through a bounded queue and are turned into tensors by the consumer. The worker starts its own CUDA context
(about 0.3-0.5 GB of GPU memory), so use it with frozen-swin runs; an unfrozen run at 11.5 of 12.2 GB has no room for it.
"""
import os, sys, random
import multiprocessing as mp
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _produce(q, box_coef, overrides, seed):
    sys.path.insert(0, ROOT)
    try:
        random.seed(seed); np.random.seed(seed % (2 ** 32)); torch.manual_seed(seed)
        from realsim.factory import make_realbkg_sim
        sim = make_realbkg_sim(device='cuda', box_coef=box_coef, overrides=overrides)
        q.put(('ready', None))
        while True:
            try:
                out = sim.simulate_img()
            except Exception:
                continue
            if out is None:
                continue
            img, boxes, mask, is_ring = out
            q.put(('frame', (img.float().cpu().numpy(), boxes.float().cpu().numpy(), is_ring.long().cpu().numpy(), mask.cpu().numpy())))
    except BaseException as e:                      # report the reason instead of letting the consumer wait forever
        import traceback
        q.put(('error', traceback.format_exc()))
        raise


class SimWorker:
    def __init__(self, box_coef, overrides, seed, qsize=64, start_timeout=900):
        ctx = mp.get_context('spawn')
        self.q = ctx.Queue(qsize)
        self.p = ctx.Process(target=_produce, args=(self.q, tuple(box_coef), overrides, int(seed) + 12345), daemon=True)
        self.p.start()
        kind, payload = self.q.get(timeout=start_timeout)
        if kind != 'ready':
            raise RuntimeError(f'sim worker failed to start:\n{payload}')
        print(f'[sim_worker] worker pid {self.p.pid} ready', flush=True)

    def sample(self, timeout=600):
        kind, payload = self.q.get(timeout=timeout)
        if kind == 'error':
            raise RuntimeError(f'sim worker died:\n{payload}')
        img, boxes, ring, mask = payload
        return torch.from_numpy(img)[None], torch.from_numpy(boxes), torch.from_numpy(ring), torch.from_numpy(mask)
