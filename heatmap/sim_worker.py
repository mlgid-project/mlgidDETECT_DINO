"""Produce real-background simulator frames in a separate process, so the training step does not wait for the 0.19 s/frame simulator.

Each worker holds the simulator (13 GB peak resident, mostly the hkl bank: on colorbox1 (30 GB) only one fits; on galvani use --mem accordingly); the training process never builds it.
Frames come back as numpy arrays through a bounded queue and are turned into tensors by the consumer. The worker starts its own CUDA context
(about 0.3-0.5 GB of GPU memory), so use it with frozen-swin runs; an unfrozen run at 11.5 of 12.2 GB has no room for it.
"""
import os, sys, random
import multiprocessing as mp
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _produce(q, rq, box_coef, overrides, seed, preset='r6'):
    sys.path.insert(0, ROOT)
    try:
        random.seed(seed); np.random.seed(seed % (2 ** 32)); torch.manual_seed(seed)
        from realsim.factory import make_realbkg_sim
        sim = make_realbkg_sim(device='cuda', box_coef=box_coef, overrides=overrides, preset=preset)
        rq.put(('ready', None))
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
        tb = traceback.format_exc()
        rq.put(('error', tb)); q.put(('error', tb))
        raise


class SimWorker:
    """n worker processes feeding one queue. They are started one after the other (each is ready before the next starts), because loading the bank
    peaks at ~13 GB per process and the peaks should not coincide."""
    def __init__(self, box_coef, overrides, seed, n=1, qsize=64, start_timeout=900, preset='r6'):
        ctx = mp.get_context('spawn')
        self.q = ctx.Queue(qsize)
        self.rq = ctx.Queue()                      # start-up handshake only: with n > 1 the frame queue already holds frames of the first worker while the next one starts
        self.ps = []
        for k in range(int(n)):
            p = ctx.Process(target=_produce, args=(self.q, self.rq, tuple(box_coef), overrides, int(seed) + 12345 + 1000 * k, preset), daemon=True)
            p.start(); self.ps.append(p)
            kind, payload = self.rq.get(timeout=start_timeout)
            if kind != 'ready':
                raise RuntimeError(f'sim worker {k} failed to start:\n{payload}')
            print(f'[sim_worker] worker {k + 1}/{n} pid {p.pid} ready', flush=True)

    def sample(self, timeout=600):
        kind, payload = self.q.get(timeout=timeout)
        if kind == 'error':
            raise RuntimeError(f'sim worker died:\n{payload}')
        if kind == 'ready':                          # cannot happen after start-up; ignore defensively
            return self.sample(timeout)
        img, boxes, ring, mask = payload
        return torch.from_numpy(img)[None], torch.from_numpy(boxes), torch.from_numpy(ring), torch.from_numpy(mask)
