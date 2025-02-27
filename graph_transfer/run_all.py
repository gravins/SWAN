import os
import tqdm
import sys

root='./'
gpus = ['0'] # gpu ids used for the experiment
models = ['swan'] # the model name defined in  conf.py
distances = [50, 10, 5, 3]
pbar = tqdm.tqdm(total= len(models)*len(distances))

for model in models:
    for d in distances: #, 100]):
        g = 3
        p = max(1, len(gpus) * g)
        ngpus = 1/g if len(gpus) else 0.0 # at least 1/g gpus per task
        ncpus = 5   # at least 1 cpu per task

        print()
        print('-----------------')
        print(f'--     {d} - {model}    --')
        print('-----------------')
        print()
        
        batch = 512
        cmd = f'export CUDA_VISIBLE_DEVICES={",".join([str(x) for x in gpus])}; '
        cmd += f'python3 -u main.py --m {model} --batch {batch} --ngpus {ngpus} '\
               f'--ncpus {ncpus} --distance {d} --root {root} --epochs 2000 --parallelism {p} '\
               f'> {os.path.join(root, f"{model}_{d}_out")} 2> {os.path.join(root, f"{model}_{d}_err")}'
        print(cmd) 
        os.system(cmd) 
        pbar.update(1)

        f = open(f'{os.path.join(root, f"{model}_{d}_err")}', 'r')
        lines = f.readlines()
        f.close()
        for l in lines[::-1]:
            if 'Error' in l:
                print(f"{model}_{d}_err: {l}", file=sys.stderr)
                break