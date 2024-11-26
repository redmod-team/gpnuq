""" Scheduling runs on a HPC cluster with SLURM

* targeted towards aCluster@tugraz.at
* each run is submitted as a job using a slurm batch script
* run arrays are submitted as a slurm job array
* by default completed runs are recognised by the interface, but the scheduler is polled as a fallback (less often)
"""

import subprocess
from time import sleep, time
import os
import json
import re

from .runner import Runner


# === Slurm Runner === #


class SlurmRunner(Runner, label="slurm"):
    """Runner which submits each run as a job to the SLURM scheduler on a cluster

    - generates a slurm batch script with the given configuration
    - can also be used with a custom script
    - supports OpenMP
    - tries to minimize overhead by using job arrays
    - polls the scheduler only at longer intervals
    """

    def __init__(
        self,
        *,
        interface="zeromq",
        cpus=1,
        openmp=False,
        custom=False,
        path="slurm.bash",
        options=None,
        command="srun profit-worker",
        **kwargs,
    ):
        self.cpus = cpus
        self.openmp = openmp
        self.custom = custom
        self.path = path
        self.options = {"job-name": "profit"}
        if options is not None:
            self.options.update(options)
        self.command = command

        super().__init__(interface=interface, **kwargs)

        with self.change_work_dir():
            if self.custom:
                if not os.path.exists(self.path):
                    self.logger.error(
                        f"flag for custom script is set, but could not be found at "
                        f"specified location {self.path}"
                    )
                    self.logger.debug(f"cwd = {os.getcwd()}")
                    self.logger.debug(f"ls = {os.listdir(os.path.dirname(self.path))}")
                    raise FileNotFoundError(f"could not find {self.path}")
            else:
                self.generate_script()

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} (" + f", {self.cpus} cpus" + ", OpenMP"
            if self.openmp
            else (
                "" + ", debug"
                if self.debug
                else "" + ", custom script" if self.custom else "" + ")>"
            )
        )

    @property
    def config(self):
        config = {}
        if not self.custom:
            config.update(
                {
                    "cpus": self.cpus,
                    "openmp": self.openmp,
                    "options": self.options,
                    "command": self.command,
                }
            )
        config.update(
            {
                "custom": self.custom,
                "path": self.path,
            }
        )
        return {**super().config, **config}  # super().config | config in python3.9

    def spawn(self, params=None, wait=False):
        super().spawn(params, wait)  # fill data with params
        self.logger.info(f"schedule run {self.next_run_id:03d} via Slurm")
        self.logger.debug(f"wait = {wait}, params = {params}")
        env = os.environ.copy()
        env["PROFIT_RUN_ID"] = str(self.next_run_id)
        env["PROFIT_WORKER"] = json.dumps(self.worker)
        env["PROFIT_INTERFACE"] = json.dumps(self.interface.config)
        env["SBATCH_EXPORT"] = "ALL"
        submit = subprocess.run(
            ["sbatch", "--parsable", self.path],
            cwd=self.work_dir,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        job_id = submit.stdout.split(";")[0].strip()
        self.runs[self.next_run_id] = job_id
        if wait:
            self.wait(self.next_run_id)
        self.next_run_id += 1

    def spawn_array(self, params_array, wait=False, progress=False):
        import tqdm

        self.logger.info(
            f"schedule array {self.next_run_id} - {self.next_run_id + len(params_array) - 1} via Slurm"
        )
        if progress:
            progressbar = tqdm.tqdm(params_array, desc="submitted")
        self.fill(params_array, offset=self.next_run_id)
        env = os.environ.copy()
        env["PROFIT_RUN_ID"] = str(self.next_run_id)
        env["PROFIT_WORKER"] = json.dumps(self.worker)
        env["PROFIT_INTERFACE"] = json.dumps(self.interface.config)
        env["SBATCH_EXPORT"] = "ALL"
        array_str = f"--array=0-{len(params_array) - 1}"
        if self.parallel > 0:
            array_str += f"%{self.parallel}"
        submit = subprocess.run(
            ["sbatch", "--parsable", array_str, self.path],
            cwd=self.work_dir,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        job_id = submit.stdout.split(";")[0].strip()
        for i in range(len(params_array)):
            self.runs[self.next_run_id + i] = f"{job_id}_{i}"
        self.next_run_id += len(params_array)
        if progress:
            progressbar.update(progressbar.total)
            progressbar.close()
        if wait:
            self.wait_all(progress=progress)

    def poll(self, run_id):
        self.logger.warning(
            "`poll(run_id)` is not supported, calling `poll_all` instead"
        )
        self.poll_all()

    def poll_all(self):
        acct = subprocess.run(
            ["sacct", f'--name={self.options["job-name"]}', "--brief", "--parsable2"],
            capture_output=True,
            text=True,
            check=True,
        )
        lookup = {job: run for run, job in self.runs.items()}
        for line in acct.stdout.split("\n"):
            if len(line) < 2:
                continue
            job_id, state = line.split("|")[:2]
            if job_id in lookup:
                run_id = lookup[job_id]
                if not (state.startswith("RUNNING") or state.startswith("PENDING")):
                    self.failed[run_id] = self.runs.pop(run_id)

        # remove slurm-logs for completed runs which did not fail
        if not self.debug:
            with self.change_work_dir():
                for path in os.listdir():
                    match = re.fullmatch(r"slurm-([\d_]+)\.out", path)
                    if match is not None:
                        job_id = match.groups()[0]
                        if (
                            job_id not in self.runs.values()
                            and job_id not in self.failed.values()
                        ):
                            os.remove(path)

    def cancel(self, run_id):
        subprocess.run(["scancel", self.runs[run_id]])
        self.failed = self.runs.pop(run_id)

    def cancel_all(self):
        ids = set()
        for run_id in self.runs:
            ids.add(re.split(r"[_.]", self.runs[run_id])[0])
        for job_id in ids:
            subprocess.run(["scancel", job_id])
        self.failed.update(self.runs)
        self.runs = {}

    def clean(self):
        """remove generated scripts and any slurm-stdout-files which match ``slurm-*.out``"""
        super().clean()
        if not self.custom and os.path.exists(self.path):
            os.remove(self.path)
        with self.change_work_dir():
            for path in os.listdir():
                if re.fullmatch(r"slurm-([\d_]+)\.out", path):
                    os.remove(path)

    def generate_script(self):
        text = f"""\
#!/bin/bash
# automatically generated SLURM batch script for running simulations with proFit
# see https://github.com/redmod-team/profit
"""
        for key, value in self.options.items():
            if value is not None:
                text += f"\n#SBATCH --{key}={value}"

        text += """
#SBATCH --ntasks=1
"""
        if self.cpus == "all" or self.cpus == 0:
            text += """
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=$SLURM_CPUS_ON_NODE"""
        elif self.cpus > 1:
            text += f"""
#SBATCH --cpus-per-task={self.cpus}"""

        if self.openmp:
            text += """
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export OMP_PLACES=threads"""

        text += f"""
if [[ -n $SLURM_ARRAY_TASK_ID ]]; then
    export PROFIT_ARRAY_ID=$SLURM_ARRAY_TASK_ID
fi
export PROFIT_RUNNER_ADDRESS=$SLURM_SUBMIT_HOST

{self.command}
"""
        with open(self.path, "w") as file:
            file.write(text)
