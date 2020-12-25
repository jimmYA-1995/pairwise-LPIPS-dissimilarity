#!/usr/bin/env python

"""Given AlexNet feat(sequence of torch.cuda.Tensor), get LPIPS dissimilarity.
   It will take about 50 minutes."""
# coding: utf-8
# Author: Chun-Ting, Ye

import subprocess
import asyncio
import math
import sys
import os

# Only leave machine idx in cmd args.
FEAT_DIR = '~/data/FFHQ/feat_face_crop'
LOG_DIR = 'logs'
OUT_DIR = 'dists'
N_RECORDS = 70000 # FFHQ
SPLITS = 80
N_GPU = 8
BATCH_SIZE=16 # depends on feat.pkl
BLOCK_SIZE=6 # tune by yct (~30G gpu mem.)

b = math.ceil(N_RECORDS / (BATCH_SIZE * BLOCK_SIZE)) # b=730 in FFHQ
total_blocks = (b+1) * b / 2
r = total_blocks % SPLITS
n = total_blocks // SPLITS
        
async def run(machine_idx):
    fs = []
    procs = []
    
    for gpu_idx, bb_idx in enumerate(range(N_GPU*machine_idx, N_GPU*(machine_idx+1))):
        print(f"process {n*bb_idx} ~ {n*(bb_idx+1)}")
        log_name = f'{LOG_DIR}/machine_{machine_idx}_gpu{gpu_idx}.log'
        procs.append(await asyncio.create_subprocess_shell(
            f"CUDA_VISIBLE_DEVICES={gpu_idx} " \
            f"python compute_distance.py --start_bb {n*bb_idx} --end_bb {n*(bb_idx+1)} " \
            f"--log_path {log_name} --feat_dir {FEAT_DIR} --out_dir {OUT_DIR}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        ))
        
    for i, proc in enumerate(procs):
        await proc.wait()
        print(f'proc {i} [exited with {proc.returncode}]')
        
        
def main():
    machine_idx = int(sys.argv[1])
    asyncio.run(run(machine_idx))
    
if __name__ == "__main__":
    main()