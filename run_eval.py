import glob
import subprocess
import os

test = glob.glob('models/model_*')
for t in test:
    name = t.split(os.sep)[1]
    print('Starting: '+name)
    template = f"""#!/bin/bash
#SBATCH --account=def-miranska
#SBATCH --cpus-per-task=16   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64G       # memory per node
#SBATCH --time=00-6:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load python/3.9

source venv/bin/activate

python -u evaluate.py --model_name '{name}' --train_loss
    """
    with open("tmp.sh", "w") as f:
        f.write(template)
    result = subprocess.run(["sbatch", "tmp.sh"])
    print("The exit code was: %d" % result.returncode)
