""" run flow components mainly used in testing

for calling a worker within the same process (label='internal')
* InternalRunner
* InternalRunnerInterface (req. InternalRunner)
* InternalInterface (req. InternalRunner)

no pre- or postprocessing needed (label='null'~None)
* NoPreprocessor
* NoPostprocessor

a worker for testing purposes without a simulation
* MockupWorker
"""

from .worker import Worker, Interface, Preprocessor, Postprocessor
from .runner import Runner, RunnerInterface

import asyncio


def mockup_f(r, u, v, a, b):
    def rosenbrock(x, y):
        return (a - x)**2 + b * (y - x**2)**2
    return rosenbrock((r - 0.5) + u - 5, 1 + 3 * (v - 0.6))


@Runner.register('internal')
class InternalRunner(Runner):
    def __init__(self, interface_class, config, base_config, worker_cls=Worker):
        super().__init__(interface_class, config, base_config)
        self.worker_cls = worker_cls

    async def spawn_run(self, params=None, wait=False):
        self.runs[self.next_run_id] = dict(worker=self.worker_cls.from_config(self.run_config, self.next_run_id))
        if self.run_config['interface']['class'] == 'internal':
            self.runs[self.next_run_id]['worker'].interface.connect(self.interface)
        self.runs[self.next_run_id]['task'] = asyncio.create_task(self.runs[self.next_run_id]['worker'].main())
        if wait:
            await self.runs[self.next_run_id]['task']
        self.next_run_id += 1

    def check_runs(self):
        for run_id in list(self.runs.keys()):
            if not self.interface.internal['DONE'][run_id]:
                self.logger.warning(f'run_{run_id:03d} likely crashed')
            del self.runs[run_id]


@RunnerInterface.register('internal')
class InternalRunnerInterface(RunnerInterface):
    def clean(self):
        self.logger.info('clean')

    def cancel(self, run_id=None):
        self.logger.info(f'cancel {run_id}')
        return False  # not possible


@Interface.register('internal')
class InternalInterface(Interface):
    def __init__(self, *args):
        super().__init__(*args)
        # full initialization requires an external call to connect()
        self.ready = False

    def connect(self, parent: RunnerInterface):
        self.parent = parent
        self.ready = True

    @property
    def input(self):
        return self.parent.input[self.worker.run_id]

    @property
    def output(self):
        return self.parent.output[self.worker.run_id]

    @property
    def time(self):
        return self.parent.internal['TIME'][self.worker.run_id]

    @time.setter
    def time(self, value):
        self.parent.internal['TIME'][self.worker.run_id] = value

    def done(self):
        self.parent.internal['DONE'][self.worker.run_id] = True


@Preprocessor.register(None)
class NoPreprocessor(Preprocessor):
    no_run_dir = True

    def pre(self):
        self.logger.info('preprocessing')
        self.logger.debug(f'input {self.worker.interface.input}')

    def post(self):
        self.logger.info('no cleanup for preprocessor necessary')
        pass

    @classmethod
    def handle_config(cls, config, base_config):
        pass


@Postprocessor.register(None)
class NoPostprocessor(Postprocessor):
    def post(self):
        self.logger.info('postprocessing')

    @classmethod
    def handle_config(cls, config, base_config):
        pass


class MockupWorker(Worker):
    def run(self):
        kwargs = {'r': 0.25, 'u': 0.5, 'v': 0.5, 'a': 1, 'b': 3}
        for x in kwargs:
            if x in self.interface.input.dtype.names:
                kwargs[x] = self.interface.input[x]
        if 'f' in self.interface.output.dtype.names:
            self.interface.output['f'] = mockup_f(**kwargs)
        self.logger.info('RUN: running simulation (mockup)')
        self.logger.debug(f'RUN: stdout={self.config["stdout"]}, stderr={self.config["stderr"]}')
        self.logger.debug(f'RUN: command={self.config["command"]}')