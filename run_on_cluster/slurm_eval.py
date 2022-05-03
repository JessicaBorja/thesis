from pathlib import Path
import subprocess
import sys

import numpy as np

from lfp.utils.utils import get_all_checkpoints

import argparse
import datetime
import os
from pathlib import Path
import stat
import subprocess

import git
from git import Repo
import numpy as np
from setuptools import sandbox

default_log_dir = os.getcwd()

def parse_args():
    parser = argparse.ArgumentParser(description="Parse slurm parameters and hydra config overrides")
    parser.add_argument("--script", type=str, default="sbatch_eval.sh")
    parser.add_argument("--eval_file", type=str, default="../thesis/evaluation/evaluate_policy.py")
    parser.add_argument("-l", "--log_dir", type=str, default=default_log_dir)
    parser.add_argument("-j", "--job_name", type=str, default="eval")
    parser.add_argument("-g", "--gpus", type=int, default=1)
    parser.add_argument("--mem", type=int, default=0)  # 0 means no memory limit
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("-v", "--venv", type=str)
    parser.add_argument("-p", "--partition", type=str, default="alldlc_gpu-rtx2080")
    parser.add_argument("--train_folder", type=str, default="~")
    args, unknownargs = parser.parse_known_args()

    assert np.all(["gpu" not in arg for arg in unknownargs])
    assert np.all(["hydra.run.dir" not in arg for arg in unknownargs])
    assert np.all(["log_dir" not in arg for arg in unknownargs])
    assert np.all(["hydra.sweep.dir" not in arg for arg in unknownargs])
    return args, unknownargs

def submit_job(job_info, script):
    # Construct sbatch command
    slurm_cmd = ["sbatch"]
    for key, value in job_info.items():
        # Check for special case keys
        if key == "cpus_per_task":
            key = "cpus-per-task"
        if key == "job_name":
            key = "job-name"
        slurm_cmd.append("--%s=%s" % (key, value))
    slurm_cmd.append(script)
    print("Generated slurm batch command: '%s'" % slurm_cmd)

    # Run sbatch command as subprocess.
    try:
        sbatch_output = subprocess.check_output(slurm_cmd)
    except subprocess.CalledProcessError as e:
        # Print error message from sbatch for easier debugging, then pass on exception
        if sbatch_output is not None:
            print("ERROR: Subprocess call output: %s" % sbatch_output)
        raise e

    print(sbatch_output.decode("utf-8"))


def main():
    """
    This script calls the evaluate.sh script of the specified training_dir 8 times with different checkpoints
    """
    args, unknownargs = parse_args()
    training_dir = Path(args.train_folder).resolve()
    log_dir = "%s/evaluation" % training_dir.as_posix()
    os.makedirs(log_dir)
    log_dir = args.train_folder
    args.script = Path(args.script).resolve()
    args.eval_file = Path(args.eval_file).resolve()
    
    aff_str = "" if "--aff_lmp" in unknownargs else "aff"

    job_opts = {
        "partition": args.partition,
        "mem": args.mem,
        "ntasks-per-node": args.gpus,
        "cpus_per_task": args.cpus,
        "gres": f"gpu:{args.gpus}",
        "output": os.path.join(log_dir, "%x.%N.%j_%sEval.out" % aff_str),
        "error": os.path.join(log_dir, "%x.%N.%j_%sEval.err" % aff_str),
        "job_name": args.job_name,
        "mail-type": "FAIL",
        "time": f"{args.days}-00:00",
    }

    script = f"{args.script.as_posix()} {args.venv} {args.eval_file.as_posix()} {' '.join(unknownargs)}"
    script += f" --train_folder {training_dir.as_posix()} --log_dir {log_dir}"
    # max_epoch = int(sys.argv[2]) if len(sys.argv) > 2 else np.inf
    if not "--checkpoint" in unknownargs:
        checkpoints = get_all_checkpoints(training_dir)
        epochs = [str(int(chk.stem.split("=")[1])) for chk in checkpoints] #if (e := int(chk.stem.split("=")[1])) <= max_epoch]
        split_epochs = np.array_split(epochs, 8)
        epoch_args = [",".join(arr) for arr in split_epochs if len(arr)]
        for epoch_arg in epoch_args:
            # cmd = [(training_dir / "evaluate.sh").as_posix(), "--checkpoints", epoch_arg]
            script += f"--checkpoints {epoch_arg}"
            submit_job(job_opts, script)
    else:
        submit_job(job_opts, script)
        


if __name__ == "__main__":
    main()
