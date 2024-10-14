"""
For computationally expensive simulations or experiments it is crucial to get the most information out of every
training point. This is not the case in the standard procedure of randomly selecting the training points.
In order to get the most out of the least number of training points, the next point is inferred by calculating an
acquisition function like the minimization of local variance or expected improvement.
"""

import numpy as np
from abc import abstractmethod
from warnings import warn

from profit.util.base_class import CustomABC
from profit.defaults import active_learning as defaults


class ActiveLearning(CustomABC):
    """Active learning base class.

    Parameters:
        runner (profit.run.Runner): Runner to dynamically start runs.
        variables (profit.util.variable.VariableGroup): Variables.
        ntrain (int): Total number of training points.
        nwarmup (int): Number of warmup (random) initialization points.
        batch_size (int): Number of training samples learned in parallel.
        convergence_criterion (float): AL is stopped when the loss of the acquisition function is lower than this
            criterion. Not implemented yet.
        nsearch (int): Number of possible candidate points in each dimension.
        make_plot (bool): Flat indicating if the AL progress is plotted.

    Attributes:
        krun (int): Current training cycle.
    """

    labels = {}

    def __init__(
        self,
        runner,
        variables,
        ntrain,
        nwarmup=defaults["nwarmup"],
        batch_size=defaults["batch_size"],
        convergence_criterion=defaults["convergence_criterion"],
        nsearch=defaults["nsearch"],
        make_plot=defaults["make_plot"],
    ):
        self.runner = runner
        self.variables = variables

        self.ntrain = ntrain
        self.nwarmup = min(nwarmup, ntrain)

        if nwarmup > ntrain:
            message = "nwarmup > ntrain. Setting nwarmup=ntrain."
            warn(message)

        self.batch_size = batch_size
        if (ntrain - nwarmup) % batch_size:
            raise RuntimeError(
                "Number of learning points ({}) and batch size ({}) for AL not compatible!".format(
                    ntrain - nwarmup, batch_size
                )
            )

        self.convergence_criterion = convergence_criterion
        self.make_plot = make_plot
        self.krun = 0

    @abstractmethod
    def warmup(self, save_intermediate=defaults["save_intermediate"]):
        """Warmup cycle before the actual learning starts."""
        pass

    @abstractmethod
    def learn(
        self,
        resume_from=defaults["resume_from"],
        save_intermediate=defaults["save_intermediate"],
    ):
        """Main loop for active learning."""
        pass

    def update_run(self, candidates):
        """Run a batch of simulations with the new candidates.

        Parameters:
            candidates (np.array): Input points to run the simulation on.
        """

        params_array = [{} for _ in range(self.batch_size)]

        for key, values in zip(self.variables.named_input.dtype.names, candidates.T):
            for idx, value in enumerate(values):
                params_array[idx][key] = value
        # Start batch
        self.runner.spawn_array(params_array, wait=True)

    def update_data(self):
        """Update the variables with the runner data."""
        from profit.util import check_ndim

        for key in self.runner.input_data.dtype.names:
            self.variables[key].value = check_ndim(self.runner.interface.input[key])
        for key in self.runner.output_data.dtype.names:
            self.variables[key].value = check_ndim(self.runner.interface.output[key])

    @abstractmethod
    def save(self, path):
        """Save the AL model.

        Parameters:
            path (str): Path where the model is saved.
        """
        pass

    def save_intermediate(self, model_path=None, input_path=None, output_path=None):
        from profit.util.file_handler import FileHandler

        if model_path:
            self.save(model_path)
        if input_path:
            FileHandler.save(input_path, self.variables.named_input)
        if output_path:
            formatted_output_data = (
                self.variables.formatted_output
                if output_path.endswith(".txt")
                else self.variables.named_output
            )
            FileHandler.save(output_path, formatted_output_data)
        print("Saved intermediate results.")

    @abstractmethod
    def plot(self):
        """Plot the progress of the AL learning."""
        pass

    @classmethod
    def from_config(cls, runner, variables, config, base_config):
        """Instantiates an ActiveLearning object from the configuration parameters.

        Parameters:
            runner (profit.run.runner.Runner): Runner instance.
            variables (profit.util.variable.VariableGroup): Variables.
            config (dict): Only the 'active_learning' part of the base_config.
            base_config (dict): The whole configuration parameters.

        Returns:
            profit.al.active_learning.ActiveLearning: AL instance.
        """

        child = cls[config["algorithm"]["class"]]
        child_instance = child.from_config(runner, variables, config, base_config)
        return child_instance
