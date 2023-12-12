import glob
import subprocess
import os
import itertools

loss = ['mae', 'mse']
normalization = ['time', 'frequency']
quantile_scaler = [True,False]

for l, n, q in itertools.product(loss, normalization, quantile_scaler):
    template = f"""#!/bin/bash
#SBATCH --account=def-miranska
#SBATCH --gres=gpu:a100:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=16   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64G       # memory per node
#SBATCH --time=00-6:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn 
module load python/3.9

source venv/bin/activate

python -u train.py --normalization {n} --epochs 20 --batch_size 64 {'--quantile_scaler ' if q else ' '}--loss {l} --augmentations
    """
    with open("tmp.sh", "w") as f:
        f.write(template)
    result = subprocess.run(["sbatch", "tmp.sh"])
    print("The exit code was: %d" % result.returncode)
