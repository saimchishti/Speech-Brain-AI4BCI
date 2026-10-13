"""Per-trial targets: HiFi-GAN mel80 (N,80,126), Mimi pre-quant latents (N,512,25), Mimi codes (N,8,25) -> data/features/{pt}_targets.npz"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import codecs_util as cu

ROOT = cu.ROOT
if __name__ == '__main__':
    mi, hg = cu.Mimi(), cu.HiFiGAN()
    for pt in (sys.argv[1:] or [f'sub-{i:02d}' for i in range(1, 11)]):
        d = np.load(f'{ROOT}/data/features/{pt}_trials.npz')
        a16, a24 = torch.tensor(d['audio16']), torch.tensor(d['audio24'])
        mel = torch.cat([hg.mel(a16[i:i + 32]).cpu() for i in range(0, len(a16), 32)])
        lat = torch.cat([mi.latents(a24[i:i + 32]).cpu() for i in range(0, len(a24), 32)])
        codes = torch.cat([mi.codes(a24[i:i + 32]).cpu() for i in range(0, len(a24), 32)])
        np.savez_compressed(f'{ROOT}/data/features/{pt}_targets.npz', mel80=mel.numpy(), mimi_lat=lat.numpy(),
                            mimi_codes=codes.numpy().astype(np.int16))
        print(pt, mel.shape, lat.shape, codes.shape, flush=True)
