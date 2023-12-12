import glob
import subprocess
import os

def main():
    test = glob.glob('models/model_*')
    for t in test:
        name = t.split(os.sep)[1]
        print('Starting: '+name)
        template = f"#!/bin/bash\nsource venv/bin/activate\npython -u evaluate.py --model_name '{name}' --train_loss"
        with open("tmp.sh", "w") as f:
            f.write(template)
        result = subprocess.run(["sbatch", "tmp.sh"])
        print("The exit code was: %d" % result.returncode)

if __name__ == '__main__':
    main()
