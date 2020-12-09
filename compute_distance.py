import gc
import logging
import argparse
import random
import pickle
from pathlib import Path
from time import time, sleep

import numpy as np
import torch
import torch.nn as nn
# from sklearn.manifold import MDS
# import matplotlib.pyplot as plt

def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size(), obj.device)

def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class PIPS(nn.Module):
    def __init__(self):
        super(PIPS, self).__init__()
        ckpt = torch.load('alex.pth', map_location='cpu')
        chs = [64,192,384,256,256]
        self.L = len(chs)
        lins = []
        for i, ch in enumerate(chs):
            setattr(self, f'lin{i}', NetLinLayer(ch, use_dropout=True))
            lins.append(getattr(self, f'lin{i}'))
        
        self.load_state_dict(ckpt)
        self.lins = nn.ModuleList(lins)
        
    def forward(self, diffs, b1, b2):
        res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        distance = torch.cat(res, dim=1).sum(dim=[1,2,3])

        return distance.view(b1,b2)

    

if __name__ == "__main__":
    N_LAYER = 5
    TOTAL = 70000
    DATA_BS = 16
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_bb', type=int)
    parser.add_argument('--end_bb', type=int)
    parser.add_argument('--row_idx', type=int, default=0)
    parser.add_argument('--col_idx', type=int, default=0)
    parser.add_argument('--batch_step', type=int, default=6, help="how many data to process in this run")
    parser.add_argument('--debug', type=bool, action='store_true')
    args = parser.parse_args()
    
    r = TOTAL % DATA_BS
    n_batch = TOTAL // DATA_BS if r == 0 else TOTAL // DATA_BS + 1
    
    r_b = n_batch % args.batch_step
    n_batch_batch = n_batch // args.batch_step if r_b == 0 else n_batch // args.batch_step + 1
    total_bb = (n_batch_batch + 1) * n_batch_batch / 2
    
    print(r, n_batch, r_b, n_batch_batch, total_bb)
    
    assert 0 <= args.start_bb < args.end_bb <= total_bb
    ffhq = sorted(list(Path('~/data/FFHQ_feat/feat').expanduser().glob('*.pkl')))
    assert len(ffhq) == n_batch
    
    model = PIPS().cuda()
    model.eval()
    
    idx = 0
    distances = []
    out_name = f'dists/dist-{args.start_bb}-{args.end_bb}.pkl'
    s = time()
    for bi in range(n_batch_batch):
        featX = None

        for bj in range(n_batch_batch):
            if bj < bi:
                continue
                
            if args.start_bb <= idx < args.end_bb:
                print("---"*15)
                dist = dict(idx=idx)
                diffs = []
                
                if featX is None:
                    stepX = r_b if bi == (n_batch_batch - 1) else args.batch_step
                    print(f"({bi},{bj},{idx}) load featX from ffhq[{bi*args.batch_step}] to ffhq[{bi * args.batch_step + stepX}]")

                    featX = [[],[],[],[],[]]
                    for i in range(bi*args.batch_step, bi*args.batch_step+stepX):
                        feat = pickle.loads(ffhq[i].read_bytes())
                        for l in range(N_LAYER):
                            featX[l].append(feat[l])
                    featX = [normalize_tensor(torch.cat(x, dim=0)) for x in featX]
                    del feat
                
                stepY = r_b if bj == (n_batch_batch - 1) else args.batch_step
                print(f"({bi},{bj},{idx}) load featY from ffhq[{bj*args.batch_step}] to ffhq[{bj * args.batch_step + stepY}]")
                
                featY = [[],[],[],[],[]]
                for i in range(bj*args.batch_step, bj*args.batch_step+stepY):
                    feat = pickle.loads(ffhq[i].read_bytes())
                    for l in range(N_LAYER):
                        featY[l].append(feat[l])
                featY = [normalize_tensor(torch.cat(x, dim=0)) for x in featY]
                del feat
                
                with torch.no_grad():
                    s = time()
                    b1 = featX[0].shape[0]
                    b2 = featY[0].shape[0]
                    for l in range(N_LAYER):
                        diff = (featX[l][None, :, :, :, :] - featY[l][:, None, :, :, :]) ** 2
                        diffs.append(diff.view(b1*b2, *diff.shape[2:]))
                    diff = None

                    output = model(diffs, b1, b2).detach().cpu().numpy()
                    dist['diff'] = output.copy()
                    del output

                    distances.append(dist)
            idx += 1
    
    pickle.dump(distances, open(out_name, 'wb'))
    print(f"save result to {out_name}")
    print(f"total: {time()-s :.2f} sec")