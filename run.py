import subprocess
import asyncio
import sys
import os

N_GPU = 8
total = (731+1) * 730 / 2
r = total % 80
n = int(total // 80)
        
async def run(machine_idx):
    fs = []
    procs = []
    for gpu_idx, bb_idx in enumerate(range(N_GPU*machine_idx, N_GPU*(machine_idx+1))):
        print(f"process {n*bb_idx} ~ {n*(bb_idx+1)}")
        log_name = f'logs/machine_{machine_idx}_gpu{gpu_idx}.log'
        procs.append(await asyncio.create_subprocess_shell(
            f"CUDA_VISIBLE_DEVICES={gpu_idx} python compute_distance.py --start_bb {n*bb_idx} --end_bb {n*(bb_idx+1)} --log_path {log_name}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        ))
    for i, proc in enumerate(procs):
        await proc.wait()
#         stdout, stderr = await proc.communicate()
        print(f'proc {i} [exited with {proc.returncode}]')
#         if stdout:
#             print(f'[stdout]\n{stdout.decode()}', file=f)
#         if stderr:
#             print(f'[stderr]\n{stderr.decode()}', file=f)
        
        
def main():
    machine_idx = int(sys.argv[1])
    asyncio.run(run(machine_idx))
    
if __name__ == "__main__":
    main()