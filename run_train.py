import glob
import subprocess
import os
import itertools

loss = ['mae', 'mse']
normalization = ['time', 'frequency']
quantile_scaler = [True,False]

for l, n, q in itertools.product(loss, normalization, quantile_scaler):
    template = f"""python -u train.py --normalization {n} --epochs 20 --batch_size 64 {'--quantile_scaler ' if q else ' '}--loss {l} --augmentations"""
    print(template)
    # with open("tmp.sh", "w") as f:
    #     f.write(template)
    # result = subprocess.run(["sbatch", "tmp.sh"])
    # print("The exit code was: %d" % result.returncode)
