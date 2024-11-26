""" mockup 1D example with a custom postprocessor """

from profit.run import Postprocessor, Worker
import numpy as np


class MockupPostprocessor(Postprocessor, label="mockup_post1"):
    """almost identical copy of NumpytxtPostprocessor"""

    def retrieve(self, data):
        raw = np.loadtxt("mockup.out")
        data["f"] = raw


@Postprocessor.wrap("mockup_post2")
def post(self, data):
    """shorthand for mockup_post1"""
    raw = np.loadtxt("mockup.out")
    data["f"] = raw


class Mockup(Worker, label="mockup_worker2"):
    """directly calling the wanted python function"""

    def work(self):
        self.interface.retrieve()
        u = self.interface.input["u"]
        self.interface.output["f"] = np.cos(10 * u) + u
        self.interface.transmit()


@Worker.wrap("mockup_worker4", "f")
def mockup4(u):
    """shorthand for mockup_worker2"""
    return np.cos(10 * u) + u
