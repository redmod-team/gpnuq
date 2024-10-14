""" Local Runner & Memory-map Interface

 * LocalRunner: start Workers locally via the shell (subprocess.Popen)
 * ForkRunner: start Workers locally with forking (multiprocessing.Process)
 * MemmapInterface: share date using a memory-mapped, structured array (using numpy)
"""

import subprocess
from multiprocessing import Process
from time import sleep
import logging
import numpy as np
import os
from shutil import rmtree
import json

from .interface import RunnerInterface, WorkerInterface
from .runner import Runner
from .worker import Worker


# === Local Runner === #


class LocalRunner(Runner, label="local"):
    """start Workers locally via the shell"""

    def __init__(self, command="profit-worker", parallel="all", **kwargs):
        if parallel == "all":  # parallel: 'all' infers the number of available CPUs
            parallel = len(os.sched_getaffinity(0))
        self.command = command
        super().__init__(parallel=parallel, **kwargs)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} (" + ", debug"
            if self.debug
            else (
                "" + f", {self.command}"
                if self.command != "profit-worker"
                else "" + ")>"
            )
        )

    @property
    def config(self):
        config = {
            "command": self.command,
        }
        return {**super().config, **config}  # super().config | config in python3.9

    def spawn(self, params=None, wait=False):
        super().spawn(params, wait)
        env = os.environ.copy()
        env["PROFIT_RUN_ID"] = str(self.next_run_id)
        env["PROFIT_WORKER"] = json.dumps(self.worker)
        env["PROFIT_INTERFACE"] = json.dumps(self.interface.config)
        self.runs[self.next_run_id] = subprocess.Popen(
            self.command, shell=True, env=env, cwd=self.work_dir
        )
        if wait:
            self.wait(self.next_run_id)
        self.next_run_id += 1

    def poll(self, run_id):
        if self.runs[run_id].poll() is not None:
            self.logger.info(f"run {run_id} failed")
            self.failed[run_id] = self.runs.pop(run_id)

    def cancel(self, run_id):
        self.runs[run_id].terminate()
        self.failed[run_id] = self.runs.pop(run_id)


# === Fork Runner === #


class ForkRunner(Runner, label="fork"):
    """start Workers locally using forking (multiprocessing.Process)"""

    def __init__(self, parallel="all", **kwargs):
        if parallel == "all":  # parallel: 'all' infers the number of available CPUs
            parallel = len(os.sched_getaffinity(0))
        super().__init__(parallel=parallel, **kwargs)

    def spawn(self, params=None, wait=False):
        super().spawn(params, wait)

        def work():
            with self.change_work_dir():
                worker = Worker.from_config(
                    self.worker, self.interface.config, self.next_run_id
                )
                worker.work()
                worker.clean()

        process = Process(target=work)
        self.runs[self.next_run_id] = process
        process.start()
        if wait:
            self.wait(self.next_run_id)
        self.next_run_id += 1

    def poll(self, run_id):
        if self.runs[run_id].exitcode is not None:
            self.logger.info(f"run {run_id} failed")
            self.failed[run_id] = self.runs.pop(run_id)

    def cancel(self, run_id):
        self.runs[run_id].terminate()
        self.failed[run_id] = self.runs.pop(run_id)


# === Numpy Memmap Interface === #


class MemmapRunnerInterface(RunnerInterface, label="memmap"):
    """Runner-Worker Interface using a memory mapped numpy array

    - expected to be very fast with the *local* Runner as each Worker can access the array directly (unverified)
    - expected to be inefficient if used on a cluster with a shared filesystem (unverified)
    - reliable
    - known issue: resizing the array (to add more runs) is dangerous, needs a workaround
      (e.g. several arrays in the same file)
    """

    def __init__(
        self,
        size,
        input_config,
        output_config,
        *,
        path: str = "interface.npy",
        logger_parent: logging.Logger = None,
    ):
        super().__init__(size, input_config, output_config, logger_parent=logger_parent)
        self.path = path

        init_data = np.zeros(
            size, dtype=self.input_vars + self.internal_vars + self.output_vars
        )
        np.save(self.path, init_data)
        self.logger.debug(f"init memmap <{self.path}, size {size}, {init_data.dtype}>")

        try:
            self._memmap = np.load(self.path, mmap_mode="r+")
        except FileNotFoundError:
            self.runner.logger.error(
                f"{self.__class__.__name__} could not load {self.path} (cwd: {os.getcwd()})"
            )
            raise

        # should return views on memmap
        self.input = self._memmap[[v[0] for v in self.input_vars]]
        self.output = self._memmap[[v[0] for v in self.output_vars]]
        self.internal = self._memmap[[v[0] for v in self.internal_vars]]

    @property
    def config(self):
        return {
            **super().config,
            "path": self.path,
        }  # super().config | config in python3.9

    def resize(self, size):
        """Resizing the Interface

        Attention: this is dangerous and may lead to unexpected errors!
        The problem is that the memory mapped file is overwritten.
        Any Workers which have this file mapped will run into severe problems.
        Possible future workarounds: multiple files or multiple headers in one file.
        """
        if size <= self.size:
            self.logger.warning("shrinking RunnerInterface is not supported")
            return

        self.logger.warning("resizing MemmapRunnerInterface is dangerous")
        self.clean()
        init_data = np.zeros(
            size, dtype=self.input_vars + self.internal_vars + self.output_vars
        )
        np.save(self.path, init_data)

        try:
            self._memmap = np.load(self.path, mmap_mode="r+")
        except FileNotFoundError:
            self.runner.logger.error(
                f"{self.__class__.__name__} could not load {self.path} (cwd: {os.getcwd()})"
            )
            raise

        self.input = self._memmap[[v[0] for v in self.input_vars]]
        self.output = self._memmap[[v[0] for v in self.output_vars]]
        self.internal = self._memmap[[v[0] for v in self.internal_vars]]

    def clean(self):
        if os.path.exists(self.path):
            os.remove(self.path)


class MemmapWorkerInterface(WorkerInterface, label="memmap"):
    """Runner-Worker Interface using a memory mapped numpy array

    counterpart to :py:class:`MemmapRunnerInterface`
    """

    def __init__(
        self, run_id: int, *, path="interface.npy", logger_parent: logging.Logger = None
    ):
        self.path = path
        self._memmap = None

        super().__init__(run_id, logger_parent=logger_parent)

    @property
    def config(self):
        return {
            **super().config,
            "path": self.path,
        }  # super().config | config in python3.9

    @property
    def time(self):
        if self._memmap is None:
            return None
        return self._data["TIME"]

    @time.setter
    def time(self, value):
        if self._memmap is not None:
            self._data["TIME"] = value

    def retrieve(self):
        try:
            self._memmap = np.load(self.path, mmap_mode="r+")
        except FileNotFoundError:
            self.logger.error(
                f"{self.__class__.__name__} could not load {self.path} (cwd: {os.getcwd()})"
            )
            raise

        # should return views on memmap
        inputs, outputs = [], []
        k = 0
        for k, key in enumerate(self._memmap.dtype.names):
            if key == "DONE":
                break
            inputs.append(key)
        for key in self._memmap.dtype.names[k:]:
            if key not in ["DONE", "TIME"]:
                outputs.append(key)
        self.input = self._memmap[inputs][self.run_id]
        self.output = self._memmap[outputs][self.run_id]
        self._data = self._memmap[self.run_id]

    def transmit(self):
        # signal the Worker has completed
        self._data["DONE"] = True
        # ensure the data is written to disk
        self._memmap.flush()

    def clean(self):
        if self._memmap is not None:
            # ensure the data is written to disk
            self._memmap.flush()
            # close the connection
            self._memmap = None
            del self._data
            del self.input
            del self.output
