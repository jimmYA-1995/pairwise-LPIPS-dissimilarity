import gc
import logging
import argparse
import random
import pickle
from pathlib import Path
from time import time

import numpy as np
import skimage.io as io
import torch
import torch.nn as nn
import sys
sys.path.append('pytorch_ssim')
from pytorch_ssim import SSIM

def debug_only(func, debug=False):
    def wrap_func(*args, **kwargs):
        if debug:
            return func(*args, **kwargs)
    return wrap_func

def memReport():
    # https://discuss.pytorch.org/t/how-pytorch-releases-variable-garbage/7277
    # https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/32
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            logging.info(f"{type(obj)}, {obj.size()}, {obj.device}")
            
def load_img(p, cuda=True):
    img = io.imread(p).astype(np.float32) / 255.
    img = img.transpose(2,0,1)[None, ...]
    img = torch.from_numpy(img)
    if cuda:
        img = img.cuda()
    return img


if __name__ == "__main__":
    N_LAYER = 1
    TOTAL = 70000
    DATA_BS = 1
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_bb', type=int)
    parser.add_argument('--end_bb', type=int)
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--batch_step', type=int, default=6, help="how many data to process in this run")
    parser.add_argument('--img_dir', type=str, default='~/data/FFHQ/images1024x1024')
    parser.add_argument('--out_dir', type=str, default='dists')
    parser.add_argument('--debug', default=False, action='store_true')
    args = parser.parse_args()
    
    loglevel = 'DEBUG' if args.debug else 'INFO'
    logging.basicConfig(
        filename=args.log_path,
        level=loglevel,
        format="%(asctime)s;%(levelname)s;%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    memReport = debug_only(memReport, args.debug)
    
    r = TOTAL % DATA_BS
    n_batch = TOTAL // DATA_BS if r == 0 else TOTAL // DATA_BS + 1
    
    r_b = n_batch % args.batch_step
    n_batch_batch = n_batch // args.batch_step if r_b == 0 else n_batch // args.batch_step + 1
    total_bb = (n_batch_batch + 1) * n_batch_batch / 2
    
    logging.info(f"{r}, {n_batch}, {r_b}, {n_batch_batch}, {total_bb}")
    
    assert 0 <= args.start_bb < args.end_bb <= total_bb
    img_paths = sorted(list(Path(args.img_dir).expanduser().glob('*/*.png')))
    assert len(img_paths) == n_batch
    
    model = SSIM(size_average=False).cuda()
    model.eval()
    
    idx = 0
    distances = []
    out_name = f'{args.out_dir}/dist-{args.start_bb}-{args.end_bb}.pkl'
    s = time()
    for bi in range(n_batch_batch):
        featX = None

        for bj in range(n_batch_batch):
            if bj < bi:
                continue
                
            if args.start_bb <= idx < args.end_bb:
                logging.debug("-"*10 + "begging" + "-"*10)
                memReport()
                dist = dict(idx=idx)
                diffs = []
                
                if featX is None:
                    stepX = r_b if bi == (n_batch_batch - 1) else args.batch_step
                    logging.info(f"({bi},{bj},{idx}) load featX from ffhq[{bi*args.batch_step}] to ffhq[{bi * args.batch_step + stepX}]")

                    imgsX = []
                    for i in range(bi*args.batch_step, bi*args.batch_step+stepX):
                        img = load_img(img_paths[i])
                        imgsX.append(img)
                    imgsX = torch.cat(imgsX, dim=0)
                    del img
                
                logging.debug("-"*10 + "featX loaded" + "-"*10)
                memReport()
                
                stepY = r_b if bj == (n_batch_batch - 1) else args.batch_step
                logging.info(f"({bi},{bj},{idx}) load featY from ffhq[{bj*args.batch_step}] to ffhq[{bj * args.batch_step + stepY}]")
                
                imgsY = []
                for i in range(bj*args.batch_step, bj*args.batch_step+stepY):
                    img = load_img(img_paths[i])
                    imgsY.append(img)
                imgsY = torch.cat(imgsY, dim=0)
                del img
                
                logging.debug("-"*10 + "featY loaded" + "-"*10)
                memReport()
                
                with torch.no_grad():
                    b1 = imgsX.shape[0]
                    b2 = imgsY.shape[0]
                    featX = []
                    featY = []
                    for i in range(b1):
                        for j in range(b2):
                            featX.append(imgsX[i].unsqueeze(0))
                            featY.append(imgsY[j].unsqueeze(0))
                    featX = torch.cat(featX, dim=0)
                    featY = torch.cat(featY, dim=0)
                    
                    # print(featX.shape, featY.shape, featX.max(), featX.min(), featX.data.type())
                    ssim_out = model(featX, featY).view(b1,b2).cpu().numpy()
                    
                    dist['diff'] = ssim_out.copy()
                    del ssim_out

                    distances.append(dist)
               
                logging.debug("-"*10 + "loop bottom" + "-"*10)
                memReport()
            idx += 1
    
    pickle.dump(distances, open(out_name, 'wb'))
    logging.info(f"save result to {out_name}")
    logging.info(f"total: {time()-s :.2f} sec to compute {len(distances)} batchOfbatch")