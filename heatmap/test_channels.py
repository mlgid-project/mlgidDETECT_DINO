"""build_channels (batched, models/heatmap_head.py) == the per-image reference of branch multi-channel-analysis
util/channels.py (pass its path as argv[1]). Tiny CPU tensors only."""
import ast, sys, importlib.util, torch

src = open('models/heatmap_head.py').read()
tree = ast.parse(src)
keep = [n for n in tree.body if (isinstance(n, ast.Assign) and n.targets[0].id == 'CHAN_N')
        or (isinstance(n, ast.FunctionDef) and n.name == 'build_channels')]
ns = {'torch': torch}
exec(compile(ast.Module(keep, []), 'hm', 'exec'), ns)
bc = ns['build_channels']

spec = importlib.util.spec_from_file_location('ref', sys.argv[1]); ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
g = torch.Generator().manual_seed(0)
B, H, W = 3, 40, 24
img = torch.rand(B, H, W, generator=g)
mask = torch.rand(B, H, W, generator=g) > 0.3
mask[1, :, 5] = False                       # an entirely invalid column
mask[2, :, 7] = False; mask[2, 3:, 9] = False
img = img.masked_fill(~mask, 0.)
full = bc(img, mask, 'full')
for b in range(B):
    r = ref.build_channels(img[b], mask[b])
    assert torch.allclose(full[b], r, atol=1e-6), (b, (full[b] - r).abs().max())
assert torch.equal(bc(img, mask, 'he')[:, 0], img)
hm = bc(img, mask, 'he_mask'); assert hm.shape == (B, 2, H, W) and torch.equal(hm[:, 1], mask.float())
assert torch.equal(hm[:, 0], full[:, 0])
print('build_channels == reference (full, he_mask, he); shapes', tuple(full.shape), tuple(hm.shape))
