import numpy as np

from profit.al import ActiveLearning
from profit.defaults import active_learning as base_defaults
from profit.defaults import al_algorithm_mcmc as defaults

np.random.seed(43)

two = False


@ActiveLearning.register("mcmc")
class McmcAL(ActiveLearning):
    """Markov-chain Monte-Carlo active learning algorithm.

    Parameters:
        reference_data (np.ndarray): Observed experimental data points. This is not the simulated model data!
        warmup_cycles (int): Number of warmup cycles with `nwarmup` iterations each.
        target_acceptance_rate (float): Target rate with which probability new points are accepted.
        sigma_n (float): Estimated standard deviation of the experimental data.
        initial_points (list of float): Starting points for the MCMC.
        delayed_acceptancd (bool): Whether to use delayed acceptance with a surrogate model for the likelihood.

    Attributes:
        Xpred (np.ndarray): Matrix of the candidate points built with np.meshgrid.
        dx (np.ndarray): Distance between iterations in parameter space.
        ndim (int): Dimension of input parameters.
        Xtrain (np.ndarray): Array of sampled MCMC points.
        log_likelihood (np.ndarray): Array of the log likelihood during training.
        accepted (np.ndarray[bool]): Boolean array of accepted/rejected sample MCMC points.
    """

    labels = {}

    def __init__(
        self,
        runner,
        variables,
        reference_data,
        ntrain,
        warmup_cycles=defaults["warmup_cycles"],
        nwarmup=base_defaults["nwarmup"],
        batch_size=base_defaults["batch_size"],
        target_acceptance_rate=defaults["target_acceptance_rate"],
        convergence_criterion=base_defaults["convergence_criterion"],
        nsearch=base_defaults["nsearch"],
        sigma_n=defaults["sigma_n"],
        make_plot=base_defaults["make_plot"],
        initial_points=defaults["initial_points"],
        save=defaults["save"],
        last_percent=defaults["last_percent"],
        delayed_acceptance=defaults["delayed_acceptance"],
    ):
        super().__init__(
            runner,
            variables,
            ntrain,
            nwarmup,
            batch_size,
            convergence_criterion,
            nsearch,
            make_plot,
        )
        self.reference_data = reference_data.reshape(1, -1)
        self.warmup_cycles = warmup_cycles
        self.target_acceptance_rate = target_acceptance_rate
        self.sigma_n = sigma_n
        self.initial_points = initial_points
        self.save_path = save
        self.last_index = int(last_percent * ntrain)

        al_keys = [
            var.name for var in variables.list if var.kind.lower() == "activelearning"
        ]
        Xpred = [
            (
                np.linspace(*var.constraints, nsearch)
                if var.name in al_keys
                else np.unique(var.value)
            )
            for var in variables.input_list
        ]
        self.Xpred = np.hstack(
            [xi.flatten().reshape(-1, 1) for xi in np.meshgrid(*Xpred)]
        )

        self.ndim = self.Xpred.shape[-1]
        self.dx = np.full((self.ntrain, self.ndim), np.nan)
        self.Xtrain = np.full((self.ntrain, self.ndim), np.nan)
        self.ytrain = np.full((self.ntrain, self.reference_data.shape[-1]), np.nan)

        self.log_likelihood = np.full((self.ntrain, 1), np.nan)
        self.accepted = np.zeros((self.ntrain, self.ndim), dtype=bool)
        self.log_random = np.log(np.random.random((self.ntrain, self.ndim)))

        self.runner.interface.resize(ntrain * self.ndim + 1)

        if delayed_acceptance:
            from profit.sur import Surrogate

            self.delayed_acceptance_surrogate = Surrogate["GPy"]()
            self.accepted_sur = np.zeros((self.ntrain, self.ndim), dtype=bool)
            self.log_random_sur = np.log(np.random.random((self.ntrain, self.ndim)))
        else:
            self.delayed_acceptance_surrogate = None

    def cost(self, y):
        return np.sum((y - self.reference_data) ** 2) / (
            y.shape[-1] * 2 * self.sigma_n**2
        )

    def f(self, x):
        nt = 250
        t = np.linspace(0, 2 * (1 - 1 / nt), nt)
        y_model = x[:, 0] * np.sin((t - x[:, 1]) ** 3)
        return y_model

    def warmup(self, save_intermediate=base_defaults["save_intermediate"]):
        """Warmup MCMC."""
        from time import time

        self.Xtrain[0] = (
            self.Xpred[np.random.choice(self.Xpred.shape[0])]
            if self.initial_points is None
            else self.initial_points
        )

        # TODO: Implement warmup with default AL model
        """
        if self.delayed_acceptance_surrogate:
            from profit.al.default_al import DefaultAL
            from profit.util.variable import OutputVariable
            log_likelihood_variable = OutputVariable('ll', 'Output', (self.ntrain, 1))
            self.variables.add(log_likelihood_variable)
            warmup_al = DefaultAL(self.runner, self.variables, self.delayed_acceptance_surrogate,
                                  self.nwarmup, 3, acquisition_function='log_likelihood')
            warmup_al.acquisition_function.reference_data = self.reference_data
            warmup_al.warmup()
            warmup_al.learn()
        """
        if two:
            self.ytrain[0] = self.f(self.Xtrain[[0]])
        else:
            self.update_run(self.Xtrain[[0]])
            self.ytrain[0] = self.runner.flat_output_data[0]

        self.log_likelihood[0] = -self.cost(self.ytrain[[0]])

        self.dx = (
            np.random.randn(self.ntrain, self.ndim) * self.sigma_n * self.Xtrain[0]
        )

        st = time()
        for cycle in range(self.warmup_cycles):
            self.krun = 1
            self.runner.next_run_id = 1
            self.accepted[:] = 0
            self.log_random[:] = np.log(np.random.random((self.ntrain, self.ndim)))

            self.do_mcmc(range(1, self.nwarmup + 1))

            if self.make_plot:
                self.plot_mcmc("warmup")

            acceptance_rate = np.mean(self.accepted[: self.nwarmup], axis=0)
            print(f"Acceptance rate for warmup cycle {cycle+1}: {acceptance_rate}")
            self.dx = self.dx * np.exp(
                acceptance_rate / self.target_acceptance_rate - 1
            )

            if self.delayed_acceptance_surrogate:
                self.delayed_acceptance_surrogate.train(
                    self.Xtrain[: self.nwarmup + 1],
                    self.log_likelihood[: self.nwarmup + 1],
                )

            if save_intermediate:
                self.save_intermediate(**save_intermediate)

            if cycle + 1 < self.warmup_cycles:
                self.Xtrain[0] = self.Xtrain[self.nwarmup]
                self.ytrain[0] = self.ytrain[self.nwarmup]
                self.log_likelihood[0] = self.log_likelihood[self.nwarmup]
        print("Runtime warmup: {}".format(time() - st))

    def learn(
        self,
        resume_from=base_defaults["resume_from"],
        save_intermediate=base_defaults["save_intermediate"],
    ):
        from time import time

        st = time()

        kstart = resume_from if resume_from is not None else self.nwarmup
        self.do_mcmc(range(kstart, self.ntrain))
        if self.delayed_acceptance_surrogate:
            print(
                "Surrogate acceptance: ",
                np.mean(self.accepted_sur[kstart + 1 :], axis=0),
            )

        if self.make_plot:
            self.plot_mcmc("learn")

        last_Xtrain = self.Xtrain[-self.last_index :]
        mean_Xtrain = np.mean(last_Xtrain, axis=0)
        std_Xtrain = np.std(last_Xtrain, axis=0)
        print("Best parameters: {} +- {}".format(mean_Xtrain, std_Xtrain))

        self.update_data()

        if save_intermediate:
            self.save_intermediate(**save_intermediate)

        print("Runtime main loop: {}".format(time() - st))
        if self.make_plot:
            from matplotlib.pyplot import show

            show()

    def do_mcmc(self, rng):
        from tqdm import tqdm

        for i in tqdm(rng):
            self.Xtrain[i] = self.Xtrain[i - 1]
            self.ytrain[i] = self.ytrain[i - 1]
            self.log_likelihood[i] = self.log_likelihood[i - 1]
            for col in range(self.ndim):
                Xguess = self.Xtrain[[i]].copy()
                Xguess[:, col] += self.dx[i - 1, col]

                if (
                    self.delayed_acceptance_surrogate
                    and self.delayed_acceptance_surrogate.trained
                ):
                    (
                        log_likelihood_guess_sur,
                        _,
                    ) = self.delayed_acceptance_surrogate.predict(Xguess)
                    A = log_likelihood_guess_sur - self.log_likelihood[i]
                    if A >= self.log_random_sur[i, col]:
                        self.accepted_sur[i - 1, col] = True
                    else:
                        continue

                if two:
                    y_guess = self.f(Xguess)
                else:
                    self.update_run(Xguess)
                    y_guess = self.runner.flat_output_data[
                        [self.runner.next_run_id - 1]
                    ].copy()
                log_likelihood_guess = -self.cost(y_guess)
                A = log_likelihood_guess - self.log_likelihood[i]
                if A >= self.log_random[i, col]:
                    self.Xtrain[i] = Xguess
                    self.ytrain[i] = y_guess
                    self.log_likelihood[i] = log_likelihood_guess
                    self.accepted[i - 1, col] = True
            self.krun += 1

    def update_run(self, candidates):
        super().update_run(candidates)

    def update_data(self):
        """Update the variables with the runner data."""

        for idx, key in enumerate(self.runner.input_data.dtype.names):
            self.variables[key].value = self.Xtrain[:, [idx]]
        for idx, key in enumerate(self.runner.output_data.dtype.names):
            # TODO: Multi Output
            self.variables[key].value = self.ytrain

    def save(self, path):
        from profit.util.file_handler import FileHandler
        from os.path import dirname, join

        dirpath = dirname(path)
        FileHandler.save(join(dirpath, "log_likelihood.txt"), self.log_likelihood)
        self.save_stats(join(dirpath, "mcmc_stats.txt"))

    def save_stats(self, path):
        """Save mean and std of X values"""
        from profit.util.file_handler import FileHandler

        last_Xtrain = self.Xtrain[-self.last_index :]
        mean_Xtrain = np.mean(last_Xtrain, axis=0)
        std_Xtrain = np.std(last_Xtrain, axis=0)
        params = np.empty((2, 1), dtype=[("Xmean", "float"), ("Xstd", "float")])
        params["Xmean"] = mean_Xtrain.reshape(-1, 1)
        params["Xstd"] = std_Xtrain.reshape(-1, 1)
        FileHandler.save(path, params)

    def plot(self):
        import matplotlib.pyplot as plt

        plt.plot(self.reference_data.T, color="r")
        plt.plot(self.ytrain[-self.last_index :].T)
        plt.show()

    def plot_mcmc(self, phase):
        from matplotlib.pyplot import subplots

        fig, ax = subplots(3, 2)
        if phase == "warmup":
            ind = range(0, self.nwarmup)
        elif phase == "learn":
            ind = range(self.nwarmup, self.ntrain)
        else:
            ind = -1
        ax[0, 0].plot(self.ytrain[ind].T, "b")
        ax[0, 0].plot(self.reference_data.T, "r")
        ax[1, 0].hist(self.dx[ind])

        if self.ndim == 1:
            ax[0, 1].scatter(self.Xtrain[ind], self.log_likelihood[ind])
        elif self.ndim == 2:
            ax[0, 1].scatter(
                self.Xtrain[ind, 0], self.Xtrain[ind, 1], c=self.log_likelihood[ind]
            )

        ax[1, 1].plot(self.log_likelihood[ind])
        ax[2, 0].hist(self.Xtrain[ind])
        ax[2, 1].plot(self.Xtrain[ind])

    @classmethod
    def from_config(cls, runner, variables, config, base_config):
        from profit.util.file_handler import FileHandler

        reference_data = FileHandler.load(config["algorithm"]["reference_data"])
        reference_data = np.hstack(
            [reference_data[key] for key in reference_data.dtype.names]
        )
        return cls(
            runner,
            variables,
            reference_data,
            base_config["ntrain"],
            warmup_cycles=config["algorithm"]["warmup_cycles"],
            nwarmup=config["nwarmup"],
            batch_size=config["batch_size"],
            target_acceptance_rate=config["algorithm"]["target_acceptance_rate"],
            convergence_criterion=config["convergence_criterion"],
            nsearch=config["nsearch"],
            sigma_n=config["algorithm"]["sigma_n"],
            make_plot=config["make_plot"],
            initial_points=config["algorithm"]["initial_points"],
            save=config["algorithm"]["save"],
            last_percent=config["algorithm"]["last_percent"],
            delayed_acceptance=config["algorithm"]["delayed_acceptance"],
        )
